#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cassert>
#include <chrono>
#include <cstring>

#include <cuda_runtime.h>
#include <nccl.h>

#define CHECK_CUDA(cmd) do { \
    cudaError_t e = cmd; \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(1); \
    } \
} while(0)

#define CHECK_NCCL(cmd) do { \
    ncclResult_t res = cmd; \
    if (res != ncclSuccess) { \
        fprintf(stderr, "NCCL error %s:%d: %s\n", __FILE__, __LINE__, ncclGetErrorString(res)); \
        exit(1); \
    } \
} while(0)

struct Config {
    int H = 8;                  // hidden size
    int I = 16;                 // intermediate size
    int T_local = 4;            // local tokens per GPU
    int num_experts = 4;        // routed experts
    int num_shared_experts = 1; // shared experts
    int topk = 2;
    int warmup = 1;
    int iters = 3;
};

struct ExpertWeights {
    float* d_w1 = nullptr; // [H, I]
    float* d_w2 = nullptr; // [I, H]
};

struct RankBuffers {
    float* d_x = nullptr;               // [T_local, H]
    float* d_shared_out = nullptr;      // [T_local, H]
    float* d_logits = nullptr;          // [T_local, E]
    float* d_routed_out = nullptr;      // [T_local, H]
    float* d_final_out = nullptr;       // [T_local, H]

    // first dispatch
    float* d_send_hidden = nullptr;     // [total_slots, H]
    float* d_recv_hidden = nullptr;     // [total_slots, H]
    int*   d_send_tok = nullptr;        // [total_slots]
    int*   d_recv_tok = nullptr;        // [total_slots]
    int*   d_send_eid = nullptr;        // [total_slots]
    int*   d_recv_eid = nullptr;        // [total_slots]
    int*   d_send_src = nullptr;        // [total_slots]
    int*   d_recv_src = nullptr;        // [total_slots]
    float* d_send_w = nullptr;          // [total_slots]
    float* d_recv_w = nullptr;          // [total_slots]

    // local expert compute
    float* d_local_out = nullptr;       // [total_slots, H]

    // send back
    float* d_back_send_hidden = nullptr;
    float* d_back_recv_hidden = nullptr;
    int*   d_back_send_tok = nullptr;
    int*   d_back_recv_tok = nullptr;
};

static inline float rand_uniform_from_seed(int seed, int idx) {
    unsigned int x = (unsigned int)(seed * 1315423911u + idx * 2654435761u);
    x ^= (x >> 16);
    x *= 2246822519u;
    x ^= (x >> 13);
    x *= 3266489917u;
    x ^= (x >> 16);
    return ((x % 20001) - 10000) / 10000.0f * 0.05f; // [-0.05, 0.05]
}



int get_int_arg(int argc, char** argv, const char* key, int default_val) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (strcmp(argv[i], key) == 0) return atoi(argv[i + 1]);
    }
    return default_val;
}

void sync_and_check(cudaStream_t stream, const char* tag) {
    CHECK_CUDA(cudaStreamSynchronize(stream));
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel error after %s: %s\n", tag, cudaGetErrorString(err));
        exit(1);
    }
}

__global__ void zero_kernel(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = 0.0f;
}

__global__ void add_inplace_kernel(float* a, const float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] += b[i];
}

__global__ void linear_kernel(const float* x, const float* w, float* y, int N, int in_dim, int out_dim) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < out_dim) {
        float sum = 0.0f;
        for (int k = 0; k < in_dim; ++k) sum += x[row * in_dim + k] * w[k * out_dim + col];
        y[row * out_dim + col] = sum;
    }
}

__global__ void gelu_kernel(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        x[i] = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
    }
}

__global__ void scale_rows_kernel(float* x, const float* row_scale, int N, int H) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < H) x[row * H + col] *= row_scale[row];
}

__global__ void gather_rows_kernel(const float* src, const int* row_idx, float* dst, int nrows, int H, int max_src_rows) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nrows && col < H) {
        int src_row = row_idx[row];
        float v = 0.0f;
        if (src_row >= 0 && src_row < max_src_rows) v = src[src_row * H + col];
        dst[row * H + col] = v;
    }
}

__global__ void scatter_rows_kernel(const float* src, const int* row_idx, float* dst, int nrows, int H, int max_dst_rows) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nrows && col < H) {
        int dst_row = row_idx[row];
        if (dst_row >= 0 && dst_row < max_dst_rows) dst[dst_row * H + col] = src[row * H + col];
    }
}

__global__ void combine_routed_kernel(const float* returned_out, const int* returned_tok, float* routed_out, int total_slots, int H, int T_local) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < total_slots && col < H) {
        int tok = returned_tok[row];
        if (tok >= 0 && tok < T_local) atomicAdd(&routed_out[tok * H + col], returned_out[row * H + col]);
    }
}

void init_matrix_device(float*& d_ptr, int rows, int cols, int seed, int device) {
    std::vector<float> h(rows * cols);
    for (int i = 0; i < rows * cols; ++i) h[i] = rand_uniform_from_seed(seed, i);
    CHECK_CUDA(cudaSetDevice(device));
    CHECK_CUDA(cudaMalloc(&d_ptr, rows * cols * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_ptr, h.data(), rows * cols * sizeof(float), cudaMemcpyHostToDevice));
}

void init_input_device(float*& d_ptr, int rows, int cols, int rank) {
    std::vector<float> h(rows * cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) h[i * cols + j] = 0.01f * (rank * 100 + i * cols + j);
    }
    CHECK_CUDA(cudaSetDevice(rank));
    CHECK_CUDA(cudaMalloc(&d_ptr, rows * cols * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_ptr, h.data(), rows * cols * sizeof(float), cudaMemcpyHostToDevice));
}

void linear_forward(const float* d_x, const float* d_w, float* d_y, int N, int in_dim, int out_dim, cudaStream_t stream) {
    dim3 block(256);
    dim3 grid((out_dim + block.x - 1) / block.x, N);
    linear_kernel<<<grid, block, 0, stream>>>(d_x, d_w, d_y, N, in_dim, out_dim);
}

void expert_forward_gpu(const float* d_x, const ExpertWeights& ew, float* d_out, int N, int H, int I, cudaStream_t stream) {
    if (N == 0) return;
    float* d_hidden = nullptr;
    CHECK_CUDA(cudaMalloc(&d_hidden, N * I * sizeof(float)));
    linear_forward(d_x, ew.d_w1, d_hidden, N, H, I, stream);
    gelu_kernel<<<(N * I + 255) / 256, 256, 0, stream>>>(d_hidden, N * I);
    linear_forward(d_hidden, ew.d_w2, d_out, N, I, H, stream);
    CHECK_CUDA(cudaFree(d_hidden));
}

int expert_owner(int expert_id, int experts_per_rank) { return expert_id / experts_per_rank; }
int local_expert_id(int expert_id, int experts_per_rank) { return expert_id % experts_per_rank; }

void router_topk_host(const std::vector<float>& logits, int T_local, int E, int topk,
                      std::vector<int>& topk_idx, std::vector<float>& topk_w) {
    topk_idx.resize(T_local * topk);
    topk_w.resize(T_local * topk);
    std::vector<float> probs(E);
    for (int t = 0; t < T_local; ++t) {
        const float* row = &logits[t * E];
        float maxv = row[0];
        for (int e = 1; e < E; ++e) maxv = std::max(maxv, row[e]);
        float sum = 0.0f;
        for (int e = 0; e < E; ++e) {
            probs[e] = expf(row[e] - maxv);
            sum += probs[e];
        }
        for (int e = 0; e < E; ++e) probs[e] /= sum;
        std::vector<int> idx(E);
        for (int e = 0; e < E; ++e) idx[e] = e;
        std::sort(idx.begin(), idx.end(), [&](int a, int b) { return probs[a] < probs[b]; });
        float wsum = 0.0f;
        for (int k = 0; k < topk; ++k) {
            int eid = idx[E - topk + k]; // week7 style ascending argsort then take last K
            topk_idx[t * topk + k] = eid;
            topk_w[t * topk + k] = probs[eid];
            wsum += probs[eid];
        }
        for (int k = 0; k < topk; ++k) topk_w[t * topk + k] /= wsum;
    }
}

void all_to_all_emulated_float(std::vector<float*>& d_send, std::vector<float*>& d_recv,
                               int total_send_count, int nranks,
                               std::vector<ncclComm_t>& comms,
                               std::vector<cudaStream_t>& streams,
                               int per_peer_chunk) {
    std::vector<float*> d_gather(nranks, nullptr);
    for (int rank = 0; rank < nranks; ++rank) {
        CHECK_CUDA(cudaSetDevice(rank));
        CHECK_CUDA(cudaMalloc(&d_gather[rank], total_send_count * nranks * sizeof(float)));
    }

    CHECK_NCCL(ncclGroupStart());
    for (int rank = 0; rank < nranks; ++rank) {
        CHECK_NCCL(ncclAllGather(d_send[rank], d_gather[rank], total_send_count, ncclFloat, comms[rank], streams[rank]));
    }
    CHECK_NCCL(ncclGroupEnd());
    for (int rank = 0; rank < nranks; ++rank) {
        CHECK_CUDA(cudaSetDevice(rank));
        sync_and_check(streams[rank], "rank sync");
    }

    for (int recv_rank = 0; recv_rank < nranks; ++recv_rank) {
        CHECK_CUDA(cudaSetDevice(recv_rank));
        std::vector<float> h_gather(total_send_count * nranks);
        std::vector<float> h_recv(total_send_count);
        CHECK_CUDA(cudaMemcpy(h_gather.data(), d_gather[recv_rank], h_gather.size() * sizeof(float), cudaMemcpyDeviceToHost));

        // gathered layout: [src0 full_send][src1 full_send]...[srcN-1 full_send]
        // each full_send layout: [dst0 chunk][dst1 chunk]...[dstN-1 chunk]
        for (int src_rank = 0; src_rank < nranks; ++src_rank) {
            const float* src_base = &h_gather[src_rank * total_send_count + recv_rank * per_peer_chunk];
            float* dst_base = &h_recv[src_rank * per_peer_chunk];
            memcpy(dst_base, src_base, per_peer_chunk * sizeof(float));
        }
        CHECK_CUDA(cudaMemcpy(d_recv[recv_rank], h_recv.data(), h_recv.size() * sizeof(float), cudaMemcpyHostToDevice));
    }

    for (int rank = 0; rank < nranks; ++rank) {
        CHECK_CUDA(cudaSetDevice(rank));
        CHECK_CUDA(cudaFree(d_gather[rank]));
    }
}

void all_to_all_emulated_int(std::vector<int*>& d_send, std::vector<int*>& d_recv,
                             int total_send_count, int nranks,
                             std::vector<ncclComm_t>& comms,
                             std::vector<cudaStream_t>& streams,
                             int per_peer_chunk) {
    std::vector<int*> d_gather(nranks, nullptr);
    for (int rank = 0; rank < nranks; ++rank) {
        CHECK_CUDA(cudaSetDevice(rank));
        CHECK_CUDA(cudaMalloc(&d_gather[rank], total_send_count * nranks * sizeof(int)));
    }

    CHECK_NCCL(ncclGroupStart());
    for (int rank = 0; rank < nranks; ++rank) {
        CHECK_NCCL(ncclAllGather(d_send[rank], d_gather[rank], total_send_count, ncclInt, comms[rank], streams[rank]));
    }
    CHECK_NCCL(ncclGroupEnd());
    for (int rank = 0; rank < nranks; ++rank) {
        CHECK_CUDA(cudaSetDevice(rank));
        sync_and_check(streams[rank], "combine final");
    }

    for (int recv_rank = 0; recv_rank < nranks; ++recv_rank) {
        CHECK_CUDA(cudaSetDevice(recv_rank));
        std::vector<int> h_gather(total_send_count * nranks);
        std::vector<int> h_recv(total_send_count);
        CHECK_CUDA(cudaMemcpy(h_gather.data(), d_gather[recv_rank], h_gather.size() * sizeof(int), cudaMemcpyDeviceToHost));
        for (int src_rank = 0; src_rank < nranks; ++src_rank) {
            const int* src_base = &h_gather[src_rank * total_send_count + recv_rank * per_peer_chunk];
            int* dst_base = &h_recv[src_rank * per_peer_chunk];
            memcpy(dst_base, src_base, per_peer_chunk * sizeof(int));
        }
        CHECK_CUDA(cudaMemcpy(d_recv[recv_rank], h_recv.data(), h_recv.size() * sizeof(int), cudaMemcpyHostToDevice));
    }

    for (int rank = 0; rank < nranks; ++rank) {
        CHECK_CUDA(cudaSetDevice(rank));
        CHECK_CUDA(cudaFree(d_gather[rank]));
    }
}

void allocate_rank_buffers(RankBuffers& rb, const Config& cfg, int total_slots, int rank) {
    CHECK_CUDA(cudaSetDevice(rank));
    CHECK_CUDA(cudaMalloc(&rb.d_shared_out, cfg.T_local * cfg.H * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&rb.d_logits, cfg.T_local * cfg.num_experts * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&rb.d_routed_out, cfg.T_local * cfg.H * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&rb.d_final_out, cfg.T_local * cfg.H * sizeof(float)));

    CHECK_CUDA(cudaMalloc(&rb.d_send_hidden, total_slots * cfg.H * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&rb.d_recv_hidden, total_slots * cfg.H * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&rb.d_send_tok, total_slots * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&rb.d_recv_tok, total_slots * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&rb.d_send_eid, total_slots * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&rb.d_recv_eid, total_slots * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&rb.d_send_src, total_slots * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&rb.d_recv_src, total_slots * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&rb.d_send_w, total_slots * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&rb.d_recv_w, total_slots * sizeof(float)));

    CHECK_CUDA(cudaMalloc(&rb.d_local_out, total_slots * cfg.H * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&rb.d_back_send_hidden, total_slots * cfg.H * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&rb.d_back_recv_hidden, total_slots * cfg.H * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&rb.d_back_send_tok, total_slots * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&rb.d_back_recv_tok, total_slots * sizeof(int)));
}

void free_rank_buffers(RankBuffers& rb) {
    cudaFree(rb.d_x);
    cudaFree(rb.d_shared_out);
    cudaFree(rb.d_logits);
    cudaFree(rb.d_routed_out);
    cudaFree(rb.d_final_out);
    cudaFree(rb.d_send_hidden);
    cudaFree(rb.d_recv_hidden);
    cudaFree(rb.d_send_tok);
    cudaFree(rb.d_recv_tok);
    cudaFree(rb.d_send_eid);
    cudaFree(rb.d_recv_eid);
    cudaFree(rb.d_send_src);
    cudaFree(rb.d_recv_src);
    cudaFree(rb.d_send_w);
    cudaFree(rb.d_recv_w);
    cudaFree(rb.d_local_out);
    cudaFree(rb.d_back_send_hidden);
    cudaFree(rb.d_back_recv_hidden);
    cudaFree(rb.d_back_send_tok);
    cudaFree(rb.d_back_recv_tok);
}

void run_forward(const Config& cfg,
                 std::vector<RankBuffers>& rb,
                 std::vector<std::vector<ExpertWeights>>& shared_experts,
                 std::vector<std::vector<ExpertWeights>>& local_experts,
                 std::vector<float*>& d_gate_w,
                 std::vector<ncclComm_t>& comms,
                 std::vector<cudaStream_t>& streams,
                 int nranks) {
    const int experts_per_rank = cfg.num_experts / nranks;

    // A. shared branch + B. router per rank
    std::vector<std::vector<int>> all_topk_idx(nranks);
    std::vector<std::vector<float>> all_topk_w(nranks);
    std::vector<std::vector<float>> all_x_host(nranks);
    std::vector<std::vector<int>> all_send_counts(nranks, std::vector<int>(nranks, 0));

    for (int rank = 0; rank < nranks; ++rank) {
        CHECK_CUDA(cudaSetDevice(rank));
        zero_kernel<<<(cfg.T_local * cfg.H + 255) / 256, 256, 0, streams[rank]>>>(rb[rank].d_shared_out, cfg.T_local * cfg.H);
        for (int se = 0; se < cfg.num_shared_experts; ++se) {
            float* d_tmp = nullptr;
            CHECK_CUDA(cudaMalloc(&d_tmp, cfg.T_local * cfg.H * sizeof(float)));
            expert_forward_gpu(rb[rank].d_x, shared_experts[rank][se], d_tmp, cfg.T_local, cfg.H, cfg.I, streams[rank]);
            add_inplace_kernel<<<(cfg.T_local * cfg.H + 255) / 256, 256, 0, streams[rank]>>>(rb[rank].d_shared_out, d_tmp, cfg.T_local * cfg.H);
            CHECK_CUDA(cudaFree(d_tmp));
        }
        linear_forward(rb[rank].d_x, d_gate_w[rank], rb[rank].d_logits, cfg.T_local, cfg.H, cfg.num_experts, streams[rank]);
    }
    for (int rank = 0; rank < nranks; ++rank) {
        CHECK_CUDA(cudaSetDevice(rank));
        CHECK_CUDA(cudaStreamSynchronize(streams[rank]));

        std::vector<float> h_logits(cfg.T_local * cfg.num_experts);
        CHECK_CUDA(cudaMemcpy(h_logits.data(), rb[rank].d_logits, h_logits.size() * sizeof(float), cudaMemcpyDeviceToHost));
        router_topk_host(h_logits, cfg.T_local, cfg.num_experts, cfg.topk, all_topk_idx[rank], all_topk_w[rank]);

        all_x_host[rank].resize(cfg.T_local * cfg.H);
        CHECK_CUDA(cudaMemcpy(all_x_host[rank].data(), rb[rank].d_x, all_x_host[rank].size() * sizeof(float), cudaMemcpyDeviceToHost));

        for (int t = 0; t < cfg.T_local; ++t) {
            for (int k = 0; k < cfg.topk; ++k) {
                int eid = all_topk_idx[rank][t * cfg.topk + k];
                int dst = expert_owner(eid, experts_per_rank);
                all_send_counts[rank][dst]++;
            }
        }
    }

    const int capacity_slots = cfg.T_local * cfg.topk;
    const int total_slots = capacity_slots * nranks;

    // C/D. pack first send buffers
    for (int rank = 0; rank < nranks; ++rank) {
        std::vector<float> h_send_hidden(total_slots * cfg.H, 0.0f);
        std::vector<int>   h_send_tok(total_slots, -1);
        std::vector<int>   h_send_eid(total_slots, -1);
        std::vector<int>   h_send_src(total_slots, -1);
        std::vector<float> h_send_w(total_slots, 0.0f);
        std::vector<int> fill(nranks, 0);

        for (int t = 0; t < cfg.T_local; ++t) {
            for (int k = 0; k < cfg.topk; ++k) {
                int eid = all_topk_idx[rank][t * cfg.topk + k];
                float w = all_topk_w[rank][t * cfg.topk + k];
                int dst = expert_owner(eid, experts_per_rank);
                int pos = fill[dst]++;
                int slot = dst * capacity_slots + pos;
                for (int j = 0; j < cfg.H; ++j) h_send_hidden[slot * cfg.H + j] = all_x_host[rank][t * cfg.H + j];
                h_send_tok[slot] = t;
                h_send_eid[slot] = eid;
                h_send_src[slot] = rank;
                h_send_w[slot] = w;
            }
        }

        CHECK_CUDA(cudaSetDevice(rank));
        CHECK_CUDA(cudaMemcpy(rb[rank].d_send_hidden, h_send_hidden.data(), h_send_hidden.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(rb[rank].d_send_tok, h_send_tok.data(), h_send_tok.size() * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(rb[rank].d_send_eid, h_send_eid.data(), h_send_eid.size() * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(rb[rank].d_send_src, h_send_src.data(), h_send_src.size() * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(rb[rank].d_send_w, h_send_w.data(), h_send_w.size() * sizeof(float), cudaMemcpyHostToDevice));
    }

    // E. first exchange (emulated all-to-all)
    {
        std::vector<float*> send_hidden(nranks), recv_hidden(nranks), send_w(nranks), recv_w(nranks);
        std::vector<int*> send_tok(nranks), recv_tok(nranks), send_eid(nranks), recv_eid(nranks), send_src(nranks), recv_src(nranks);
        for (int rank = 0; rank < nranks; ++rank) {
            send_hidden[rank] = rb[rank].d_send_hidden;
            recv_hidden[rank] = rb[rank].d_recv_hidden;
            send_tok[rank] = rb[rank].d_send_tok;
            recv_tok[rank] = rb[rank].d_recv_tok;
            send_eid[rank] = rb[rank].d_send_eid;
            recv_eid[rank] = rb[rank].d_recv_eid;
            send_src[rank] = rb[rank].d_send_src;
            recv_src[rank] = rb[rank].d_recv_src;
            send_w[rank] = rb[rank].d_send_w;
            recv_w[rank] = rb[rank].d_recv_w;
        }
        all_to_all_emulated_float(send_hidden, recv_hidden, total_slots * cfg.H, nranks, comms, streams, capacity_slots * cfg.H);
        all_to_all_emulated_int(send_tok, recv_tok, total_slots, nranks, comms, streams, capacity_slots);
        all_to_all_emulated_int(send_eid, recv_eid, total_slots, nranks, comms, streams, capacity_slots);
        all_to_all_emulated_int(send_src, recv_src, total_slots, nranks, comms, streams, capacity_slots);
        all_to_all_emulated_float(send_w, recv_w, total_slots, nranks, comms, streams, capacity_slots);
    }

    // F. local expert compute
    for (int rank = 0; rank < nranks; ++rank) {
        CHECK_CUDA(cudaSetDevice(rank));
        zero_kernel<<<(total_slots * cfg.H + 255) / 256, 256, 0, streams[rank]>>>(rb[rank].d_local_out, total_slots * cfg.H);

        std::vector<int> h_recv_tok(total_slots), h_recv_eid(total_slots), h_recv_src(total_slots);
        std::vector<float> h_recv_w(total_slots);
        CHECK_CUDA(cudaMemcpy(h_recv_tok.data(), rb[rank].d_recv_tok, total_slots * sizeof(int), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_recv_eid.data(), rb[rank].d_recv_eid, total_slots * sizeof(int), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_recv_src.data(), rb[rank].d_recv_src, total_slots * sizeof(int), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_recv_w.data(), rb[rank].d_recv_w, total_slots * sizeof(float), cudaMemcpyDeviceToHost));

        for (int lid = 0; lid < experts_per_rank; ++lid) {
            std::vector<int> rows;
            std::vector<float> scales;
            for (int row = 0; row < total_slots; ++row) {
                int eid = h_recv_eid[row];
                if (eid < 0) continue;
                if (local_expert_id(eid, experts_per_rank) == lid) {
                    rows.push_back(row);
                    scales.push_back(h_recv_w[row]);
                }
            }
            int nrows = (int)rows.size();
            if (nrows == 0) continue;

            int* d_rows = nullptr;
            float* d_scales = nullptr;
            float* d_in = nullptr;
            float* d_out = nullptr;
            CHECK_CUDA(cudaMalloc(&d_rows, nrows * sizeof(int)));
            CHECK_CUDA(cudaMalloc(&d_scales, nrows * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&d_in, nrows * cfg.H * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&d_out, nrows * cfg.H * sizeof(float)));
            CHECK_CUDA(cudaMemcpy(d_rows, rows.data(), nrows * sizeof(int), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(d_scales, scales.data(), nrows * sizeof(float), cudaMemcpyHostToDevice));

            dim3 block(256);
            dim3 grid((cfg.H + block.x - 1) / block.x, nrows);
            gather_rows_kernel<<<grid, block, 0, streams[rank]>>>(rb[rank].d_recv_hidden, d_rows, d_in, nrows, cfg.H, total_slots);
            expert_forward_gpu(d_in, local_experts[rank][lid], d_out, nrows, cfg.H, cfg.I, streams[rank]);
            scale_rows_kernel<<<grid, block, 0, streams[rank]>>>(d_out, d_scales, nrows, cfg.H);
            scatter_rows_kernel<<<grid, block, 0, streams[rank]>>>(d_out, d_rows, rb[rank].d_local_out, nrows, cfg.H, total_slots);
            sync_and_check(streams[rank], "local expert compute");

            cudaFree(d_rows);
            cudaFree(d_scales);
            cudaFree(d_in);
            cudaFree(d_out);
        }
        CHECK_CUDA(cudaStreamSynchronize(streams[rank]));
    }

    // G. pack send-back buffers by original source rank
    for (int rank = 0; rank < nranks; ++rank) {
        std::vector<int> h_recv_tok(total_slots), h_recv_src(total_slots);
        std::vector<float> h_local_out(total_slots * cfg.H);
        CHECK_CUDA(cudaSetDevice(rank));
        CHECK_CUDA(cudaMemcpy(h_recv_tok.data(), rb[rank].d_recv_tok, total_slots * sizeof(int), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_recv_src.data(), rb[rank].d_recv_src, total_slots * sizeof(int), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_local_out.data(), rb[rank].d_local_out, h_local_out.size() * sizeof(float), cudaMemcpyDeviceToHost));

        std::vector<float> h_back_hidden(total_slots * cfg.H, 0.0f);
        std::vector<int>   h_back_tok(total_slots, -1);
        std::vector<int> fill(nranks, 0);

        for (int row = 0; row < total_slots; ++row) {
            int src_rank = h_recv_src[row];
            int tok = h_recv_tok[row];
            if (src_rank < 0 || tok < 0) continue;
            int pos = fill[src_rank]++;
            int slot = src_rank * capacity_slots + pos;
            for (int j = 0; j < cfg.H; ++j) h_back_hidden[slot * cfg.H + j] = h_local_out[row * cfg.H + j];
            h_back_tok[slot] = tok;
        }

        CHECK_CUDA(cudaMemcpy(rb[rank].d_back_send_hidden, h_back_hidden.data(), h_back_hidden.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(rb[rank].d_back_send_tok, h_back_tok.data(), h_back_tok.size() * sizeof(int), cudaMemcpyHostToDevice));
    }

    // H. second exchange (return expert outputs)
    {
        std::vector<float*> send_hidden(nranks), recv_hidden(nranks);
        std::vector<int*> send_tok(nranks), recv_tok(nranks);
        for (int rank = 0; rank < nranks; ++rank) {
            send_hidden[rank] = rb[rank].d_back_send_hidden;
            recv_hidden[rank] = rb[rank].d_back_recv_hidden;
            send_tok[rank] = rb[rank].d_back_send_tok;
            recv_tok[rank] = rb[rank].d_back_recv_tok;
        }
        all_to_all_emulated_float(send_hidden, recv_hidden, total_slots * cfg.H, nranks, comms, streams, capacity_slots * cfg.H);
        all_to_all_emulated_int(send_tok, recv_tok, total_slots, nranks, comms, streams, capacity_slots);
    }

    // I. combine routed + shared
    for (int rank = 0; rank < nranks; ++rank) {
        CHECK_CUDA(cudaSetDevice(rank));
        zero_kernel<<<(cfg.T_local * cfg.H + 255) / 256, 256, 0, streams[rank]>>>(rb[rank].d_routed_out, cfg.T_local * cfg.H);
        dim3 block(256);
        dim3 grid((cfg.H + block.x - 1) / block.x, total_slots);
        combine_routed_kernel<<<grid, block, 0, streams[rank]>>>(rb[rank].d_back_recv_hidden, rb[rank].d_back_recv_tok, rb[rank].d_routed_out, total_slots, cfg.H, cfg.T_local);
        CHECK_CUDA(cudaMemcpyAsync(rb[rank].d_final_out, rb[rank].d_routed_out, cfg.T_local * cfg.H * sizeof(float), cudaMemcpyDeviceToDevice, streams[rank]));
        add_inplace_kernel<<<(cfg.T_local * cfg.H + 255) / 256, 256, 0, streams[rank]>>>(rb[rank].d_final_out, rb[rank].d_shared_out, cfg.T_local * cfg.H);
    }
    for (int rank = 0; rank < nranks; ++rank) {
        CHECK_CUDA(cudaSetDevice(rank));
        CHECK_CUDA(cudaStreamSynchronize(streams[rank]));
    }

    // show routing once for debugging
    for (int rank = 0; rank < nranks; ++rank) {
        printf("Rank %d routing:\n", rank);
        for (int t = 0; t < cfg.T_local; ++t) {
            printf("  token %d -> ", t);
            for (int k = 0; k < cfg.topk; ++k) {
                printf("[eid=%d, w=%.4f] ", all_topk_idx[rank][t * cfg.topk + k], all_topk_w[rank][t * cfg.topk + k]);
            }
            printf("\n");
        }
    }
}

int main(int argc, char** argv) {
    Config cfg;
    cfg.H = get_int_arg(argc, argv, "--hidden", cfg.H);
    cfg.I = get_int_arg(argc, argv, "--intermediate", cfg.I);
    cfg.T_local = get_int_arg(argc, argv, "--tokens", cfg.T_local);
    cfg.num_experts = get_int_arg(argc, argv, "--experts", cfg.num_experts);
    cfg.topk = get_int_arg(argc, argv, "--topk", cfg.topk);
    cfg.num_shared_experts = get_int_arg(argc, argv, "--shared", cfg.num_shared_experts);
    cfg.warmup = get_int_arg(argc, argv, "--warmup", cfg.warmup);
    cfg.iters = get_int_arg(argc, argv, "--iters", cfg.iters);
    int ngpu = 0;
    CHECK_CUDA(cudaGetDeviceCount(&ngpu));
    if (ngpu < 2) {
        fprintf(stderr, "Need at least 2 visible GPUs, got %d\n", ngpu);
        return 1;
    }
    if (cfg.num_experts % ngpu != 0) {
        fprintf(stderr, "num_experts (%d) must be divisible by visible GPU count (%d)\n", cfg.num_experts, ngpu);
        return 1;
    }

    printf("Visible GPUs: %d\n", ngpu);
    printf("Config: H=%d I=%d T_local=%d E=%d topk=%d shared=%d warmup=%d iters=%d\n",
           cfg.H, cfg.I, cfg.T_local, cfg.num_experts, cfg.topk, cfg.num_shared_experts, cfg.warmup, cfg.iters);

    std::vector<int> devs(ngpu);
    std::vector<ncclComm_t> comms(ngpu);
    std::vector<cudaStream_t> streams(ngpu);
    for (int rank = 0; rank < ngpu; ++rank) devs[rank] = rank;
    CHECK_NCCL(ncclCommInitAll(comms.data(), ngpu, devs.data()));

    std::vector<RankBuffers> rb(ngpu);
    const int max_capacity_slots = cfg.T_local * cfg.topk;
    const int max_total_slots = max_capacity_slots * ngpu;
    for (int rank = 0; rank < ngpu; ++rank) {
        init_input_device(rb[rank].d_x, cfg.T_local, cfg.H, rank);
        CHECK_CUDA(cudaSetDevice(rank));
        CHECK_CUDA(cudaStreamCreate(&streams[rank]));
        allocate_rank_buffers(rb[rank], cfg, max_total_slots, rank);
    }

    std::vector<float*> d_gate_w(ngpu, nullptr);
    for (int rank = 0; rank < ngpu; ++rank) init_matrix_device(d_gate_w[rank], cfg.H, cfg.num_experts, 2025, rank);

    std::vector<std::vector<ExpertWeights>> shared_experts(ngpu, std::vector<ExpertWeights>(cfg.num_shared_experts));
    for (int rank = 0; rank < ngpu; ++rank) {
        for (int se = 0; se < cfg.num_shared_experts; ++se) {
            init_matrix_device(shared_experts[rank][se].d_w1, cfg.H, cfg.I, 3000 + se * 10 + 1, rank);
            init_matrix_device(shared_experts[rank][se].d_w2, cfg.I, cfg.H, 3000 + se * 10 + 2, rank);
        }
    }

    const int experts_per_rank = cfg.num_experts / ngpu;
    std::vector<std::vector<ExpertWeights>> local_experts(ngpu, std::vector<ExpertWeights>(experts_per_rank));
    for (int rank = 0; rank < ngpu; ++rank) {
        for (int lid = 0; lid < experts_per_rank; ++lid) {
            int gid = rank * experts_per_rank + lid;
            init_matrix_device(local_experts[rank][lid].d_w1, cfg.H, cfg.I, 5000 + gid * 10 + 1, rank);
            init_matrix_device(local_experts[rank][lid].d_w2, cfg.I, cfg.H, 5000 + gid * 10 + 2, rank);
        }
    }

    // warmup + benchmark
    for (int i = 0; i < cfg.warmup; ++i) run_forward(cfg, rb, shared_experts, local_experts, d_gate_w, comms, streams, ngpu);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < cfg.iters; ++i) run_forward(cfg, rb, shared_experts, local_experts, d_gate_w, comms, streams, ngpu);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / cfg.iters;

    printf("Average forward time: %.3f ms\n", ms);

    for (int rank = 0; rank < ngpu; ++rank) {
        CHECK_CUDA(cudaSetDevice(rank));
        std::vector<float> h_final(cfg.T_local * cfg.H);
        CHECK_CUDA(cudaMemcpy(h_final.data(), rb[rank].d_final_out, h_final.size() * sizeof(float), cudaMemcpyDeviceToHost));
        printf("Final output rank %d:\n", rank);
        for (int t = 0; t < cfg.T_local; ++t) {
            printf("  token %d :", t);
            for (int j = 0; j < cfg.H; ++j) printf(" %.5f", h_final[t * cfg.H + j]);
            printf("\n");
        }
    }

    for (int rank = 0; rank < ngpu; ++rank) {
        CHECK_CUDA(cudaSetDevice(rank));
        free_rank_buffers(rb[rank]);
        cudaFree(d_gate_w[rank]);
        cudaStreamDestroy(streams[rank]);
    }
    for (int rank = 0; rank < ngpu; ++rank) {
        CHECK_CUDA(cudaSetDevice(rank));
        for (auto& e : shared_experts[rank]) {
            cudaFree(e.d_w1);
            cudaFree(e.d_w2);
        }
    }
    for (int rank = 0; rank < ngpu; ++rank) {
        CHECK_CUDA(cudaSetDevice(rank));
        for (auto& e : local_experts[rank]) {
            cudaFree(e.d_w1);
            cudaFree(e.d_w2);
        }
        ncclCommDestroy(comms[rank]);
    }

    return 0;
}