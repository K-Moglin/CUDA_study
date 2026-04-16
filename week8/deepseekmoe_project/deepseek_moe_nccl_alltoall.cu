#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <nccl.h>

#define CHECK_CUDA(cmd)                                                                          \
    do {                                                                                         \
        cudaError_t e = (cmd);                                                                   \
        if (e != cudaSuccess) {                                                                  \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
            std::exit(1);                                                                        \
        }                                                                                        \
    } while (0)

#define CHECK_NCCL(cmd)                                                                          \
    do {                                                                                         \
        ncclResult_t r = (cmd);                                                                  \
        if (r != ncclSuccess) {                                                                  \
            fprintf(stderr, "NCCL error %s:%d: %s\n", __FILE__, __LINE__, ncclGetErrorString(r)); \
            std::exit(1);                                                                        \
        }                                                                                        \
    } while (0)

struct Config {
    int H = 8;
    int I = 16;
    int T_local = 4;
    int E = 4;
    int shared = 1;
    int topk = 2;
    int warmup = 1;
    int iters = 5;
    int check = 0;
};

struct ExpertWeights {
    float* d_w1 = nullptr;  // [H, I]
    float* d_w2 = nullptr;  // [I, H]
};

struct RankBuffers {
    float* d_x = nullptr;           // [T_local, H]
    float* d_logits = nullptr;      // [T_local, E]
    float* d_shared_out = nullptr;  // [T_local, H]
    float* d_routed_out = nullptr;  // [T_local, H]
    float* d_final_out = nullptr;   // [T_local, H]

    float* d_send_hidden = nullptr; // [total_slots, H]
    int* d_send_tok = nullptr;      // [total_slots]
    int* d_send_eid = nullptr;      // [total_slots]
    int* d_send_src = nullptr;      // [total_slots]
    float* d_send_w = nullptr;      // [total_slots]

    float* d_recv_hidden = nullptr; // [total_slots, H]
    int* d_recv_tok = nullptr;      // [total_slots]
    int* d_recv_eid = nullptr;      // [total_slots]
    int* d_recv_src = nullptr;      // [total_slots]
    float* d_recv_w = nullptr;      // [total_slots]

    float* d_local_out = nullptr;   // [total_slots, H]

    float* d_back_send_hidden = nullptr; // [total_slots, H]
    int* d_back_send_tok = nullptr;      // [total_slots]
    float* d_back_recv_hidden = nullptr; // [total_slots, H]
    int* d_back_recv_tok = nullptr;      // [total_slots]
};

static inline float rand_uniform_from_seed(int seed, int idx) {
    unsigned int x = (unsigned int)(seed * 1315423911u + idx * 2654435761u);
    x ^= (x >> 16);
    x *= 2246822519u;
    x ^= (x >> 13);
    x *= 3266489917u;
    x ^= (x >> 16);
    return ((x % 20001) - 10000) / 10000.0f * 0.05f;
}

int get_int_arg(int argc, char** argv, const char* key, int default_val) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], key) == 0) return std::atoi(argv[i + 1]);
    }
    return default_val;
}

bool has_flag(int argc, char** argv, const char* key) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], key) == 0) return true;
    }
    return false;
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

void sync_stream(cudaStream_t stream, const char* tag) {
    CHECK_CUDA(cudaStreamSynchronize(stream));
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel error after %s: %s\n", tag, cudaGetErrorString(err));
        std::exit(1);
    }
}

void linear_forward(const float* d_x, const float* d_w, float* d_y, int N, int in_dim, int out_dim, cudaStream_t stream) {
    dim3 block(256);
    dim3 grid((out_dim + block.x - 1) / block.x, N);
    linear_kernel<<<grid, block, 0, stream>>>(d_x, d_w, d_y, N, in_dim, out_dim);
}

void expert_forward_gpu(const float* d_x, const ExpertWeights& ew, float* d_out, int N, int H, int I, cudaStream_t stream) {
    if (N == 0) return;
    float* d_hidden = nullptr;
    CHECK_CUDA(cudaMalloc(&d_hidden, (size_t)N * I * sizeof(float)));
    linear_forward(d_x, ew.d_w1, d_hidden, N, H, I, stream);
    gelu_kernel<<<((size_t)N * I + 255) / 256, 256, 0, stream>>>(d_hidden, N * I);
    linear_forward(d_hidden, ew.d_w2, d_out, N, I, H, stream);
    CHECK_CUDA(cudaFree(d_hidden));
}

int expert_owner(int expert_id, int experts_per_rank) { return expert_id / experts_per_rank; }
int local_expert_id(int expert_id, int experts_per_rank) { return expert_id % experts_per_rank; }

template <typename T> ncclDataType_t nccl_dtype();
template <> ncclDataType_t nccl_dtype<float>() { return ncclFloat; }
template <> ncclDataType_t nccl_dtype<int>() { return ncclInt; }

template <typename T>
void nccl_all_to_all_fixed(
    std::vector<T*>& d_send,
    std::vector<T*>& d_recv,
    int per_peer_count,
    int nranks,
    std::vector<ncclComm_t>& comms,
    std::vector<cudaStream_t>& streams) {
    CHECK_NCCL(ncclGroupStart());
    for (int rank = 0; rank < nranks; ++rank) {
        for (int peer = 0; peer < nranks; ++peer) {
            const T* send_ptr = d_send[rank] + (size_t)peer * per_peer_count;
            T* recv_ptr = d_recv[rank] + (size_t)peer * per_peer_count;
            CHECK_NCCL(ncclSend(send_ptr, per_peer_count, nccl_dtype<T>(), peer, comms[rank], streams[rank]));
            CHECK_NCCL(ncclRecv(recv_ptr, per_peer_count, nccl_dtype<T>(), peer, comms[rank], streams[rank]));
        }
    }
    CHECK_NCCL(ncclGroupEnd());
    for (int rank = 0; rank < nranks; ++rank) {
        CHECK_CUDA(cudaSetDevice(rank));
        sync_stream(streams[rank], "nccl all-to-all");
    }
}

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
            probs[e] = std::exp(row[e] - maxv);
            sum += probs[e];
        }
        for (int e = 0; e < E; ++e) probs[e] /= sum;
        std::vector<int> idx(E);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](int a, int b) { return probs[a] < probs[b]; });
        float wsum = 0.0f;
        for (int k = 0; k < topk; ++k) {
            int eid = idx[E - topk + k];
            topk_idx[t * topk + k] = eid;
            topk_w[t * topk + k] = probs[eid];
            wsum += probs[eid];
        }
        for (int k = 0; k < topk; ++k) topk_w[t * topk + k] /= wsum;
    }
}

void init_matrix_device(float*& d_ptr, int rows, int cols, int seed, int device) {
    std::vector<float> h((size_t)rows * cols);
    for (int i = 0; i < rows * cols; ++i) h[i] = rand_uniform_from_seed(seed, i);
    CHECK_CUDA(cudaSetDevice(device));
    CHECK_CUDA(cudaMalloc(&d_ptr, (size_t)rows * cols * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_ptr, h.data(), (size_t)rows * cols * sizeof(float), cudaMemcpyHostToDevice));
}

void init_input_device(float*& d_ptr, int rows, int cols, int rank) {
    std::vector<float> h((size_t)rows * cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) h[i * cols + j] = 0.01f * (rank * 100 + i * cols + j);
    }
    CHECK_CUDA(cudaSetDevice(rank));
    CHECK_CUDA(cudaMalloc(&d_ptr, (size_t)rows * cols * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_ptr, h.data(), (size_t)rows * cols * sizeof(float), cudaMemcpyHostToDevice));
}

void allocate_rank_buffers(RankBuffers& rb, const Config& cfg, int total_slots, int rank) {
    CHECK_CUDA(cudaSetDevice(rank));
    CHECK_CUDA(cudaMalloc(&rb.d_logits, (size_t)cfg.T_local * cfg.E * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&rb.d_shared_out, (size_t)cfg.T_local * cfg.H * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&rb.d_routed_out, (size_t)cfg.T_local * cfg.H * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&rb.d_final_out, (size_t)cfg.T_local * cfg.H * sizeof(float)));

    CHECK_CUDA(cudaMalloc(&rb.d_send_hidden, (size_t)total_slots * cfg.H * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&rb.d_send_tok, (size_t)total_slots * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&rb.d_send_eid, (size_t)total_slots * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&rb.d_send_src, (size_t)total_slots * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&rb.d_send_w, (size_t)total_slots * sizeof(float)));

    CHECK_CUDA(cudaMalloc(&rb.d_recv_hidden, (size_t)total_slots * cfg.H * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&rb.d_recv_tok, (size_t)total_slots * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&rb.d_recv_eid, (size_t)total_slots * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&rb.d_recv_src, (size_t)total_slots * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&rb.d_recv_w, (size_t)total_slots * sizeof(float)));

    CHECK_CUDA(cudaMalloc(&rb.d_local_out, (size_t)total_slots * cfg.H * sizeof(float)));

    CHECK_CUDA(cudaMalloc(&rb.d_back_send_hidden, (size_t)total_slots * cfg.H * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&rb.d_back_send_tok, (size_t)total_slots * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&rb.d_back_recv_hidden, (size_t)total_slots * cfg.H * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&rb.d_back_recv_tok, (size_t)total_slots * sizeof(int)));
}

void free_rank_buffers(RankBuffers& rb) {
    cudaFree(rb.d_x);
    cudaFree(rb.d_logits);
    cudaFree(rb.d_shared_out);
    cudaFree(rb.d_routed_out);
    cudaFree(rb.d_final_out);
    cudaFree(rb.d_send_hidden);
    cudaFree(rb.d_send_tok);
    cudaFree(rb.d_send_eid);
    cudaFree(rb.d_send_src);
    cudaFree(rb.d_send_w);
    cudaFree(rb.d_recv_hidden);
    cudaFree(rb.d_recv_tok);
    cudaFree(rb.d_recv_eid);
    cudaFree(rb.d_recv_src);
    cudaFree(rb.d_recv_w);
    cudaFree(rb.d_local_out);
    cudaFree(rb.d_back_send_hidden);
    cudaFree(rb.d_back_send_tok);
    cudaFree(rb.d_back_recv_hidden);
    cudaFree(rb.d_back_recv_tok);
}

struct RunOutputs {
    std::vector<std::vector<float>> final_per_rank;
    double avg_ms = 0.0;
};

RunOutputs run_forward(const Config& cfg,
                      std::vector<RankBuffers>& rb,
                      std::vector<std::vector<ExpertWeights>>& shared_experts,
                      std::vector<std::vector<ExpertWeights>>& local_experts,
                      std::vector<float*>& d_gate_w,
                      std::vector<ncclComm_t>& comms,
                      std::vector<cudaStream_t>& streams,
                      int nranks) {
    const int experts_per_rank = cfg.E / nranks;
    const int capacity_slots = cfg.T_local * cfg.topk;
    const int total_slots = capacity_slots * nranks;

    std::vector<std::vector<float>> last_final(nranks);
    double elapsed_ms = 0.0;

    for (int iter = 0; iter < cfg.warmup + cfg.iters; ++iter) {
        auto t0 = std::chrono::high_resolution_clock::now();

        std::vector<std::vector<int>> all_topk_idx(nranks);
        std::vector<std::vector<float>> all_topk_w(nranks);
        std::vector<std::vector<float>> all_x_host(nranks);

        // A. shared branch + router logits.
        for (int rank = 0; rank < nranks; ++rank) {
            CHECK_CUDA(cudaSetDevice(rank));
            zero_kernel<<<((size_t)cfg.T_local * cfg.H + 255) / 256, 256, 0, streams[rank]>>>(rb[rank].d_shared_out, cfg.T_local * cfg.H);
            for (int se = 0; se < cfg.shared; ++se) {
                float* d_tmp = nullptr;
                CHECK_CUDA(cudaMalloc(&d_tmp, (size_t)cfg.T_local * cfg.H * sizeof(float)));
                expert_forward_gpu(rb[rank].d_x, shared_experts[rank][se], d_tmp, cfg.T_local, cfg.H, cfg.I, streams[rank]);
                add_inplace_kernel<<<((size_t)cfg.T_local * cfg.H + 255) / 256, 256, 0, streams[rank]>>>(rb[rank].d_shared_out, d_tmp, cfg.T_local * cfg.H);
                CHECK_CUDA(cudaFree(d_tmp));
            }
            linear_forward(rb[rank].d_x, d_gate_w[rank], rb[rank].d_logits, cfg.T_local, cfg.H, cfg.E, streams[rank]);
        }
        for (int rank = 0; rank < nranks; ++rank) {
            CHECK_CUDA(cudaSetDevice(rank));
            sync_stream(streams[rank], "shared+router");

            std::vector<float> h_logits((size_t)cfg.T_local * cfg.E);
            CHECK_CUDA(cudaMemcpy(h_logits.data(), rb[rank].d_logits, h_logits.size() * sizeof(float), cudaMemcpyDeviceToHost));
            router_topk_host(h_logits, cfg.T_local, cfg.E, cfg.topk, all_topk_idx[rank], all_topk_w[rank]);

            all_x_host[rank].resize((size_t)cfg.T_local * cfg.H);
            CHECK_CUDA(cudaMemcpy(all_x_host[rank].data(), rb[rank].d_x, all_x_host[rank].size() * sizeof(float), cudaMemcpyDeviceToHost));
        }

        // B. pack dispatch buffers on host.
        for (int rank = 0; rank < nranks; ++rank) {
            std::vector<float> h_send_hidden((size_t)total_slots * cfg.H, 0.0f);
            std::vector<int> h_send_tok(total_slots, -1);
            std::vector<int> h_send_eid(total_slots, -1);
            std::vector<int> h_send_src(total_slots, -1);
            std::vector<float> h_send_w(total_slots, 0.0f);
            std::vector<int> fill(nranks, 0);

            for (int t = 0; t < cfg.T_local; ++t) {
                for (int k = 0; k < cfg.topk; ++k) {
                    int eid = all_topk_idx[rank][t * cfg.topk + k];
                    float w = all_topk_w[rank][t * cfg.topk + k];
                    int dst = expert_owner(eid, experts_per_rank);
                    int pos = fill[dst]++;
                    int slot = dst * capacity_slots + pos;
                    for (int j = 0; j < cfg.H; ++j) h_send_hidden[(size_t)slot * cfg.H + j] = all_x_host[rank][(size_t)t * cfg.H + j];
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

        // C. first true all-to-all.
        {
            std::vector<float*> send_hidden(nranks), recv_hidden(nranks), send_w(nranks), recv_w(nranks);
            std::vector<int*> send_tok(nranks), recv_tok(nranks), send_eid(nranks), recv_eid(nranks), send_src(nranks), recv_src(nranks);
            for (int rank = 0; rank < nranks; ++rank) {
                send_hidden[rank] = rb[rank].d_send_hidden; recv_hidden[rank] = rb[rank].d_recv_hidden;
                send_tok[rank] = rb[rank].d_send_tok; recv_tok[rank] = rb[rank].d_recv_tok;
                send_eid[rank] = rb[rank].d_send_eid; recv_eid[rank] = rb[rank].d_recv_eid;
                send_src[rank] = rb[rank].d_send_src; recv_src[rank] = rb[rank].d_recv_src;
                send_w[rank] = rb[rank].d_send_w; recv_w[rank] = rb[rank].d_recv_w;
            }
            nccl_all_to_all_fixed<float>(send_hidden, recv_hidden, capacity_slots * cfg.H, nranks, comms, streams);
            nccl_all_to_all_fixed<int>(send_tok, recv_tok, capacity_slots, nranks, comms, streams);
            nccl_all_to_all_fixed<int>(send_eid, recv_eid, capacity_slots, nranks, comms, streams);
            nccl_all_to_all_fixed<int>(send_src, recv_src, capacity_slots, nranks, comms, streams);
            nccl_all_to_all_fixed<float>(send_w, recv_w, capacity_slots, nranks, comms, streams);
        }

        // D. local expert compute.
        for (int rank = 0; rank < nranks; ++rank) {
            CHECK_CUDA(cudaSetDevice(rank));
            zero_kernel<<<((size_t)total_slots * cfg.H + 255) / 256, 256, 0, streams[rank]>>>(rb[rank].d_local_out, total_slots * cfg.H);
            sync_stream(streams[rank], "zero local out");

            std::vector<int> h_recv_eid(total_slots), h_recv_tok(total_slots), h_recv_src(total_slots);
            std::vector<float> h_recv_w(total_slots);
            CHECK_CUDA(cudaMemcpy(h_recv_eid.data(), rb[rank].d_recv_eid, (size_t)total_slots * sizeof(int), cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(h_recv_tok.data(), rb[rank].d_recv_tok, (size_t)total_slots * sizeof(int), cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(h_recv_src.data(), rb[rank].d_recv_src, (size_t)total_slots * sizeof(int), cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(h_recv_w.data(), rb[rank].d_recv_w, (size_t)total_slots * sizeof(float), cudaMemcpyDeviceToHost));

            for (int lid = 0; lid < experts_per_rank; ++lid) {
                std::vector<int> rows;
                std::vector<float> scales;
                for (int row = 0; row < total_slots; ++row) {
                    int eid = h_recv_eid[row];
                    if (eid >= 0 && local_expert_id(eid, experts_per_rank) == lid) {
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
                CHECK_CUDA(cudaMalloc(&d_rows, (size_t)nrows * sizeof(int)));
                CHECK_CUDA(cudaMalloc(&d_scales, (size_t)nrows * sizeof(float)));
                CHECK_CUDA(cudaMalloc(&d_in, (size_t)nrows * cfg.H * sizeof(float)));
                CHECK_CUDA(cudaMalloc(&d_out, (size_t)nrows * cfg.H * sizeof(float)));
                CHECK_CUDA(cudaMemcpy(d_rows, rows.data(), (size_t)nrows * sizeof(int), cudaMemcpyHostToDevice));
                CHECK_CUDA(cudaMemcpy(d_scales, scales.data(), (size_t)nrows * sizeof(float), cudaMemcpyHostToDevice));

                dim3 block(256);
                dim3 grid((cfg.H + block.x - 1) / block.x, nrows);
                gather_rows_kernel<<<grid, block, 0, streams[rank]>>>(rb[rank].d_recv_hidden, d_rows, d_in, nrows, cfg.H, total_slots);
                expert_forward_gpu(d_in, local_experts[rank][lid], d_out, nrows, cfg.H, cfg.I, streams[rank]);
                scale_rows_kernel<<<grid, block, 0, streams[rank]>>>(d_out, d_scales, nrows, cfg.H);
                scatter_rows_kernel<<<grid, block, 0, streams[rank]>>>(d_out, d_rows, rb[rank].d_local_out, nrows, cfg.H, total_slots);
                sync_stream(streams[rank], "local expert compute");

                cudaFree(d_rows); cudaFree(d_scales); cudaFree(d_in); cudaFree(d_out);
            }
        }

        // E. pack return buffers.
        for (int rank = 0; rank < nranks; ++rank) {
            std::vector<float> h_local_out((size_t)total_slots * cfg.H);
            std::vector<int> h_recv_src(total_slots), h_recv_tok(total_slots);
            CHECK_CUDA(cudaSetDevice(rank));
            CHECK_CUDA(cudaMemcpy(h_local_out.data(), rb[rank].d_local_out, h_local_out.size() * sizeof(float), cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(h_recv_src.data(), rb[rank].d_recv_src, (size_t)total_slots * sizeof(int), cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(h_recv_tok.data(), rb[rank].d_recv_tok, (size_t)total_slots * sizeof(int), cudaMemcpyDeviceToHost));

            std::vector<float> h_back_send((size_t)total_slots * cfg.H, 0.0f);
            std::vector<int> h_back_tok(total_slots, -1);
            std::vector<int> fill(nranks, 0);
            for (int row = 0; row < total_slots; ++row) {
                int dst = h_recv_src[row];
                if (dst < 0) continue;
                int pos = fill[dst]++;
                int slot = dst * capacity_slots + pos;
                std::memcpy(&h_back_send[(size_t)slot * cfg.H], &h_local_out[(size_t)row * cfg.H], (size_t)cfg.H * sizeof(float));
                h_back_tok[slot] = h_recv_tok[row];
            }

            CHECK_CUDA(cudaMemcpy(rb[rank].d_back_send_hidden, h_back_send.data(), h_back_send.size() * sizeof(float), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(rb[rank].d_back_send_tok, h_back_tok.data(), h_back_tok.size() * sizeof(int), cudaMemcpyHostToDevice));
        }

        // F. second true all-to-all.
        {
            std::vector<float*> send_hidden(nranks), recv_hidden(nranks);
            std::vector<int*> send_tok(nranks), recv_tok(nranks);
            for (int rank = 0; rank < nranks; ++rank) {
                send_hidden[rank] = rb[rank].d_back_send_hidden; recv_hidden[rank] = rb[rank].d_back_recv_hidden;
                send_tok[rank] = rb[rank].d_back_send_tok; recv_tok[rank] = rb[rank].d_back_recv_tok;
            }
            nccl_all_to_all_fixed<float>(send_hidden, recv_hidden, capacity_slots * cfg.H, nranks, comms, streams);
            nccl_all_to_all_fixed<int>(send_tok, recv_tok, capacity_slots, nranks, comms, streams);
        }

        // G. combine + shared.
        for (int rank = 0; rank < nranks; ++rank) {
            CHECK_CUDA(cudaSetDevice(rank));
            zero_kernel<<<((size_t)cfg.T_local * cfg.H + 255) / 256, 256, 0, streams[rank]>>>(rb[rank].d_routed_out, cfg.T_local * cfg.H);
            sync_stream(streams[rank], "zero routed out");
            dim3 block(256);
            dim3 grid((cfg.H + block.x - 1) / block.x, total_slots);
            combine_routed_kernel<<<grid, block, 0, streams[rank]>>>(rb[rank].d_back_recv_hidden, rb[rank].d_back_recv_tok, rb[rank].d_routed_out, total_slots, cfg.H, cfg.T_local);
            sync_stream(streams[rank], "combine routed");

            CHECK_CUDA(cudaMemcpyAsync(rb[rank].d_final_out, rb[rank].d_routed_out, (size_t)cfg.T_local * cfg.H * sizeof(float), cudaMemcpyDeviceToDevice, streams[rank]));
            add_inplace_kernel<<<((size_t)cfg.T_local * cfg.H + 255) / 256, 256, 0, streams[rank]>>>(rb[rank].d_final_out, rb[rank].d_shared_out, cfg.T_local * cfg.H);
            sync_stream(streams[rank], "add shared");
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        if (iter >= cfg.warmup) {
            elapsed_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
    }

    for (int rank = 0; rank < nranks; ++rank) {
        last_final[rank].resize((size_t)cfg.T_local * cfg.H);
        CHECK_CUDA(cudaSetDevice(rank));
        CHECK_CUDA(cudaMemcpy(last_final[rank].data(), rb[rank].d_final_out, last_final[rank].size() * sizeof(float), cudaMemcpyDeviceToHost));
    }

    RunOutputs out;
    out.final_per_rank = std::move(last_final);
    out.avg_ms = elapsed_ms / std::max(1, cfg.iters);
    return out;
}

void expert_forward_cpu(const std::vector<float>& x, const std::vector<float>& w1, const std::vector<float>& w2, int H, int I, std::vector<float>& y) {
    std::vector<float> h1(I, 0.0f);
    for (int j = 0; j < I; ++j) {
        for (int k = 0; k < H; ++k) h1[j] += x[k] * w1[k * I + j];
        float v = h1[j];
        h1[j] = 0.5f * v * (1.0f + std::tanh(0.7978845608f * (v + 0.044715f * v * v * v)));
    }
    y.assign(H, 0.0f);
    for (int j = 0; j < H; ++j) {
        for (int k = 0; k < I; ++k) y[j] += h1[k] * w2[k * H + j];
    }
}

std::vector<std::vector<float>> cpu_reference(const Config& cfg, int nranks) {
    const int T_total = cfg.T_local * nranks;

    std::vector<std::vector<float>> gate_w(nranks, std::vector<float>((size_t)cfg.H * cfg.E));
    std::vector<std::vector<std::vector<float>>> shared_w1(nranks, std::vector<std::vector<float>>(cfg.shared));
    std::vector<std::vector<std::vector<float>>> shared_w2(nranks, std::vector<std::vector<float>>(cfg.shared));

    const int experts_per_rank = cfg.E / nranks;
    std::vector<std::vector<std::vector<float>>> local_w1(nranks, std::vector<std::vector<float>>(experts_per_rank));
    std::vector<std::vector<std::vector<float>>> local_w2(nranks, std::vector<std::vector<float>>(experts_per_rank));

    for (int rank = 0; rank < nranks; ++rank) {
        for (int i = 0; i < cfg.H * cfg.E; ++i) gate_w[rank][i] = rand_uniform_from_seed(17 + rank, i);
        for (int se = 0; se < cfg.shared; ++se) {
            shared_w1[rank][se].resize((size_t)cfg.H * cfg.I);
            shared_w2[rank][se].resize((size_t)cfg.I * cfg.H);
            for (int i = 0; i < cfg.H * cfg.I; ++i) shared_w1[rank][se][i] = rand_uniform_from_seed(1000 + rank * 97 + se * 3, i);
            for (int i = 0; i < cfg.I * cfg.H; ++i) shared_w2[rank][se][i] = rand_uniform_from_seed(2000 + rank * 97 + se * 3, i);
        }
        for (int lid = 0; lid < experts_per_rank; ++lid) {
            local_w1[rank][lid].resize((size_t)cfg.H * cfg.I);
            local_w2[rank][lid].resize((size_t)cfg.I * cfg.H);
            int eid = rank * experts_per_rank + lid;
            for (int i = 0; i < cfg.H * cfg.I; ++i) local_w1[rank][lid][i] = rand_uniform_from_seed(3000 + eid * 11, i);
            for (int i = 0; i < cfg.I * cfg.H; ++i) local_w2[rank][lid][i] = rand_uniform_from_seed(4000 + eid * 11, i);
        }
    }

    std::vector<std::vector<float>> final_per_rank(nranks, std::vector<float>((size_t)cfg.T_local * cfg.H, 0.0f));
    for (int rank = 0; rank < nranks; ++rank) {
        for (int t = 0; t < cfg.T_local; ++t) {
            std::vector<float> x(cfg.H);
            for (int j = 0; j < cfg.H; ++j) x[j] = 0.01f * (rank * 100 + t * cfg.H + j);

            std::vector<float> shared_out(cfg.H, 0.0f);
            for (int se = 0; se < cfg.shared; ++se) {
                std::vector<float> tmp;
                expert_forward_cpu(x, shared_w1[rank][se], shared_w2[rank][se], cfg.H, cfg.I, tmp);
                for (int j = 0; j < cfg.H; ++j) shared_out[j] += tmp[j];
            }

            std::vector<float> logits(cfg.E, 0.0f);
            for (int e = 0; e < cfg.E; ++e) {
                for (int j = 0; j < cfg.H; ++j) logits[e] += x[j] * gate_w[rank][j * cfg.E + e];
            }
            float maxv = *std::max_element(logits.begin(), logits.end());
            std::vector<float> probs(cfg.E);
            float s = 0.0f;
            for (int e = 0; e < cfg.E; ++e) { probs[e] = std::exp(logits[e] - maxv); s += probs[e]; }
            for (int e = 0; e < cfg.E; ++e) probs[e] /= s;
            std::vector<int> idx(cfg.E);
            std::iota(idx.begin(), idx.end(), 0);
            std::sort(idx.begin(), idx.end(), [&](int a, int b) { return probs[a] < probs[b]; });

            std::vector<float> routed(cfg.H, 0.0f);
            float wsum = 0.0f;
            for (int k = 0; k < cfg.topk; ++k) wsum += probs[idx[cfg.E - cfg.topk + k]];
            for (int k = 0; k < cfg.topk; ++k) {
                int eid = idx[cfg.E - cfg.topk + k];
                float w = probs[eid] / wsum;
                int owner = expert_owner(eid, experts_per_rank);
                int lid = local_expert_id(eid, experts_per_rank);
                std::vector<float> tmp;
                expert_forward_cpu(x, local_w1[owner][lid], local_w2[owner][lid], cfg.H, cfg.I, tmp);
                for (int j = 0; j < cfg.H; ++j) routed[j] += w * tmp[j];
            }
            for (int j = 0; j < cfg.H; ++j) final_per_rank[rank][(size_t)t * cfg.H + j] = routed[j] + shared_out[j];
        }
    }
    return final_per_rank;
}

int main(int argc, char** argv) {
    Config cfg;
    cfg.H = get_int_arg(argc, argv, "--H", cfg.H);
    cfg.I = get_int_arg(argc, argv, "--I", cfg.I);
    cfg.T_local = get_int_arg(argc, argv, "--T_local", cfg.T_local);
    cfg.E = get_int_arg(argc, argv, "--E", cfg.E);
    cfg.shared = get_int_arg(argc, argv, "--shared", cfg.shared);
    cfg.topk = get_int_arg(argc, argv, "--topk", cfg.topk);
    cfg.warmup = get_int_arg(argc, argv, "--warmup", cfg.warmup);
    cfg.iters = get_int_arg(argc, argv, "--iters", cfg.iters);
    cfg.check = has_flag(argc, argv, "--check") ? 1 : 0;

    int nranks = 0;
    CHECK_CUDA(cudaGetDeviceCount(&nranks));
    if (nranks < 2) {
        std::cerr << "Need at least 2 GPUs for this project.\n";
        return 1;
    }
    nranks = 2;
    if (cfg.E % nranks != 0) {
        std::cerr << "E must be divisible by number of ranks.\n";
        return 1;
    }

    const int experts_per_rank = cfg.E / nranks;
    const int capacity_slots = cfg.T_local * cfg.topk;
    const int total_slots = capacity_slots * nranks;

    std::cout << "Config: nranks=" << nranks
              << " T_local=" << cfg.T_local
              << " H=" << cfg.H
              << " I=" << cfg.I
              << " E=" << cfg.E
              << " shared=" << cfg.shared
              << " topk=" << cfg.topk << "\n";

    std::vector<ncclComm_t> comms(nranks);
    std::vector<cudaStream_t> streams(nranks);
    std::vector<RankBuffers> rb(nranks);

    ncclUniqueId id;
    CHECK_NCCL(ncclGetUniqueId(&id));
    for (int rank = 0; rank < nranks; ++rank) {
        CHECK_CUDA(cudaSetDevice(rank));
        CHECK_CUDA(cudaStreamCreate(&streams[rank]));
    }
    CHECK_NCCL(ncclGroupStart());
    for (int rank = 0; rank < nranks; ++rank) {
        CHECK_CUDA(cudaSetDevice(rank));
        CHECK_NCCL(ncclCommInitRank(&comms[rank], nranks, id, rank));
    }
    CHECK_NCCL(ncclGroupEnd());

    std::vector<float*> d_gate_w(nranks, nullptr);
    std::vector<std::vector<ExpertWeights>> shared_experts(nranks, std::vector<ExpertWeights>(cfg.shared));
    std::vector<std::vector<ExpertWeights>> local_experts(nranks, std::vector<ExpertWeights>(experts_per_rank));

    for (int rank = 0; rank < nranks; ++rank) {
        init_input_device(rb[rank].d_x, cfg.T_local, cfg.H, rank);
        init_matrix_device(d_gate_w[rank], cfg.H, cfg.E, 17 + rank, rank);
        for (int se = 0; se < cfg.shared; ++se) {
            init_matrix_device(shared_experts[rank][se].d_w1, cfg.H, cfg.I, 1000 + rank * 97 + se * 3, rank);
            init_matrix_device(shared_experts[rank][se].d_w2, cfg.I, cfg.H, 2000 + rank * 97 + se * 3, rank);
        }
        for (int lid = 0; lid < experts_per_rank; ++lid) {
            int eid = rank * experts_per_rank + lid;
            init_matrix_device(local_experts[rank][lid].d_w1, cfg.H, cfg.I, 3000 + eid * 11, rank);
            init_matrix_device(local_experts[rank][lid].d_w2, cfg.I, cfg.H, 4000 + eid * 11, rank);
        }
        allocate_rank_buffers(rb[rank], cfg, total_slots, rank);
    }

    RunOutputs gpu_out = run_forward(cfg, rb, shared_experts, local_experts, d_gate_w, comms, streams, nranks);

    std::cout << "Average forward time: " << gpu_out.avg_ms << " ms\n";
    for (int rank = 0; rank < nranks; ++rank) {
        std::cout << "Output rank" << rank << ": ";
        for (int i = 0; i < std::min(8, cfg.T_local * cfg.H); ++i) std::cout << gpu_out.final_per_rank[rank][i] << ' ';
        std::cout << "\n";
    }

    if (cfg.check) {
        auto ref = cpu_reference(cfg, nranks);
        float max_abs = 0.0f;
        float max_rel = 0.0f;
        for (int rank = 0; rank < nranks; ++rank) {
            for (size_t i = 0; i < ref[rank].size(); ++i) {
                float a = gpu_out.final_per_rank[rank][i];
                float b = ref[rank][i];
                float abs_err = std::fabs(a - b);
                float rel_err = abs_err / std::max(1e-6f, std::fabs(b));
                max_abs = std::max(max_abs, abs_err);
                max_rel = std::max(max_rel, rel_err);
            }
        }
        std::cout << "max |gpu-cpu| = " << max_abs << "\n";
        std::cout << "max rel error = " << max_rel << "\n";

        // For large-magnitude outputs, relative error is the meaningful metric.
        // Use a mixed tolerance similar to allclose: abs_err <= atol + rtol * |ref|
        const float atol = 1e-3f;
        const float rtol = 1e-4f;

        bool pass = true;
        for (int rank = 0; rank < nranks; ++rank) {
            for (size_t i = 0; i < ref[rank].size(); ++i) {
                float a = gpu_out.final_per_rank[rank][i];
                float b = ref[rank][i];
                float abs_err = std::fabs(a - b);
                if (abs_err > atol + rtol * std::fabs(b)) {
                    pass = false;
                    break;
                }
            }
            if (!pass) break;
        }

        if (pass) {
            std::cout << "TEST PASS\n";
        } else {
            std::cout << "TEST FAIL\n";
            return 2;
        }
    }

    for (int rank = 0; rank < nranks; ++rank) {
        CHECK_CUDA(cudaSetDevice(rank));
        free_rank_buffers(rb[rank]);
        cudaFree(d_gate_w[rank]);
        for (int se = 0; se < cfg.shared; ++se) {
            cudaFree(shared_experts[rank][se].d_w1);
            cudaFree(shared_experts[rank][se].d_w2);
        }
        for (int lid = 0; lid < experts_per_rank; ++lid) {
            cudaFree(local_experts[rank][lid].d_w1);
            cudaFree(local_experts[rank][lid].d_w2);
        }
        ncclCommDestroy(comms[rank]);
        cudaStreamDestroy(streams[rank]);
    }
    return 0;
}
