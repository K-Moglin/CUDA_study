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
#include <cuda_bf16.h>
#include <nccl.h>

// TK integration hook. This file keeps a compileable baseline path and centralizes
// expert GEMM dispatch so the ThunderKittens implementation can be swapped in one place.
#ifndef USE_TK_GEMM
#define USE_TK_GEMM 0
#endif

#if USE_TK_GEMM
#include "kittens.cuh"
#include "prototype.cuh"
#endif

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
    __nv_bfloat16* d_w1_bf16 = nullptr; // [H, I]
    __nv_bfloat16* d_w2_bf16 = nullptr; // [I, H]
};

struct RankBuffers {
    float* d_x = nullptr;           // [T_local, H]
    float* d_logits = nullptr;      // [T_local, E]
    float* d_shared_out = nullptr;  // [T_local, H]
    float* d_routed_out = nullptr;  // [T_local, H]
    float* d_final_out = nullptr;   // [T_local, H]

    // Reusable scratch buffers. These remove repeated cudaMalloc/cudaFree from the hot path
    // and also define the exact GEMM handoff points for future ThunderKittens kernels.
    float* d_shared_tmp = nullptr;      // [T_local, H]
    float* d_shared_hidden = nullptr;   // [T_local, I]

    // Device-side routed top-k metadata and small counters
    int* d_topk_idx = nullptr;          // [T_local, topk]
    float* d_topk_w = nullptr;          // [T_local, topk]
    int* d_count = nullptr;             // scalar counter used by row/scales builders

    int* d_rows_buf = nullptr;          // [total_slots]
    float* d_scales_buf = nullptr;      // [total_slots]
    float* d_expert_in = nullptr;       // [total_slots, H]
    float* d_expert_hidden = nullptr;   // [total_slots, I]
    float* d_expert_out = nullptr;      // [total_slots, H]

    // ThunderKittens path scratch (bf16 staging / output conversion)
    __nv_bfloat16* d_tk_a = nullptr;    // [total_slots, max(H, I)]
    __nv_bfloat16* d_tk_c = nullptr;    // [total_slots, max(H, I)]

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


__global__ void pack_dispatch_kernel(const float* x, const int* topk_idx, const float* topk_w,
                                     float* send_hidden, int* send_tok, int* send_eid, int* send_src, float* send_w,
                                     int T_local, int H, int topk, int experts_per_rank, int capacity_slots, int rank, int nranks) {
    int slot = blockIdx.x * blockDim.x + threadIdx.x;
    int total_slots = capacity_slots * nranks;
    if (slot >= total_slots) return;
    int peer = slot / capacity_slots;
    int local = slot % capacity_slots;
    int t = local / topk;
    int k = local % topk;
    int eid = -1;
    float w = 0.0f;
    if (t < T_local) {
        eid = topk_idx[t * topk + k];
        w = topk_w[t * topk + k];
    }
    bool active = (t < T_local) && (eid >= 0) && ((eid / experts_per_rank) == peer);
    send_tok[slot] = active ? t : -1;
    send_eid[slot] = active ? eid : -1;
    send_src[slot] = active ? rank : -1;
    send_w[slot] = active ? w : 0.0f;
    for (int j = 0; j < H; ++j) {
        send_hidden[(size_t)slot * H + j] = active ? x[(size_t)t * H + j] : 0.0f;
    }
}

__global__ void build_rows_scales_kernel(const int* recv_eid, const float* recv_w, int total_slots, int experts_per_rank, int lid,
                                         int* counter, int* rows_out, float* scales_out) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= total_slots) return;
    int eid = recv_eid[row];
    if (eid >= 0 && (eid % experts_per_rank) == lid) {
        int pos = atomicAdd(counter, 1);
        rows_out[pos] = row;
        scales_out[pos] = recv_w[row];
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


__global__ void fp32_to_bf16_kernel(const float* src, __nv_bfloat16* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2bfloat16(src[i]);
}

__global__ void bf16_to_fp32_kernel(const __nv_bfloat16* src, float* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __bfloat162float(src[i]);
}

__global__ void zero_bf16_kernel(__nv_bfloat16* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = __float2bfloat16(0.0f);
}

#if USE_TK_GEMM
using namespace kittens;

namespace tkrect {
static constexpr int TILE_M = 128;
static constexpr int TILE_N = 256;
static constexpr int TILE_K = 64;
static constexpr int PIPE_STAGES = 3;
static constexpr int EPI_PIPE_DEPTH = 4;
static constexpr int NUM_WARPS = 4;
static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
using a_tile = st_bf<TILE_M, TILE_K>;
using b_tile = st_bf<TILE_K, TILE_N>;
using d_tile = st_bf<TILE_M, TILE_N / EPI_PIPE_DEPTH>;
using a_gl = gl<bf16, 1, 1, -1, -1, a_tile>;
using b_gl = gl<bf16, 1, 1, -1, -1, b_tile>;
using d_gl = gl<bf16, 1, 1, -1, -1, d_tile>;
using d_tt_t = tt<float, TILE_M, TILE_N>;
using d_tt_sub = tt<float, TILE_M, TILE_N / EPI_PIPE_DEPTH>;

__global__ __launch_bounds__(NUM_THREADS, 1) void matmul_kernel(
    const __grid_constant__ a_gl A_layout,
    const __grid_constant__ b_gl B_layout,
    const __grid_constant__ d_gl D_layout,
    int M, int N, int K) {
    const int warpid = threadIdx.x / WARP_THREADS;
    const int laneid = threadIdx.x % WARP_THREADS;
    const int wg_laneid = warpgroup::laneid();

    const int grid_n = N / TILE_N;
    const int bid_m = blockIdx.x / grid_n;
    const int bid_n = blockIdx.x % grid_n;

    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int *)&__shm[0]);
    a_tile(&a_smem)[PIPE_STAGES] = al.allocate<a_tile, PIPE_STAGES>();
    b_tile(&b_smem)[PIPE_STAGES] = al.allocate<b_tile, PIPE_STAGES>();
    d_tile(&d_smem)[EPI_PIPE_DEPTH] = al.allocate<d_tile, EPI_PIPE_DEPTH>();

    __shared__ semaphore inputs_arrived[PIPE_STAGES];
    __shared__ semaphore inputs_finished[PIPE_STAGES];
    __shared__ semaphore compute_done;

    if (threadIdx.x == 0) {
        for (int i = 0; i < PIPE_STAGES; i++) {
            init_semaphore(inputs_arrived[i], 0, 1);
            init_semaphore(inputs_finished[i], 1, 0);
        }
        init_semaphore(compute_done, 0, 1);
        A_layout.template prefetch_tma<a_tile>();
        B_layout.template prefetch_tma<b_tile>();
        D_layout.template prefetch_tma<d_tile>();
    }
    __syncthreads();

    tensor_allocator<1, 1> tm_alloc{};
    d_tt_t accum;
    if (wg_laneid == 0) accum = tm_alloc.allocate<d_tt_t>(0);
    __syncthreads();

    const int num_k_iters = K / TILE_K;
    int phase = 0;

    if (warpid == 0 && laneid == 0) {
        for (int iter_k = 0; iter_k < num_k_iters; iter_k++) {
            const int stage = iter_k % PIPE_STAGES;
            wait(inputs_finished[stage], phase ^ 1);
            if (stage == PIPE_STAGES - 1) phase ^= 1;
            tma::expect_bytes(inputs_arrived[stage], sizeof(a_tile) + sizeof(b_tile));
            tma::load_async(a_smem[stage], A_layout, {bid_m, iter_k}, inputs_arrived[stage]);
            tma::load_async(b_smem[stage], B_layout, {iter_k, bid_n}, inputs_arrived[stage]);
        }
    } else if (warpid == 1 && laneid == 0) {
        for (int iter_k = 0; iter_k < num_k_iters; iter_k++) {
            const int stage = iter_k % PIPE_STAGES;
            wait(inputs_arrived[stage], phase);
            if (stage == PIPE_STAGES - 1) phase ^= 1;
            if (iter_k == 0) mm_AB(accum, a_smem[stage], b_smem[stage], inputs_finished[stage]);
            else mma_AB(accum, a_smem[stage], b_smem[stage], inputs_finished[stage]);
        }
        detail::tcgen05::commit<1>(compute_done);
    }

    wait(compute_done, 0);
    rt_bf<TILE_M / 4, TILE_N / EPI_PIPE_DEPTH> d_regs[EPI_PIPE_DEPTH];
#pragma unroll
    for (int i = 0; i < EPI_PIPE_DEPTH; i++) {
        warpgroup::load_async(d_regs[i], accum.subtile<d_tt_sub>(0, i * (TILE_N / EPI_PIPE_DEPTH)));
    }
    tensor_load_wait();
#pragma unroll
    for (int i = 0; i < EPI_PIPE_DEPTH; i++) {
        warpgroup::sync(1);
        warpgroup::store(d_smem[i], d_regs[i]);
        warpgroup::sync(1);
        if (wg_laneid == 0) {
            tma::store_async(D_layout, d_smem[i], {bid_m, bid_n * EPI_PIPE_DEPTH + i});
        }
    }
    tma::store_async_read_wait();
}

inline void matmul_rect(__nv_bfloat16* A, __nv_bfloat16* B, __nv_bfloat16* C, int M, int N, int K, cudaStream_t stream) {
    a_gl A_layout{reinterpret_cast<bf16 *>(A), nullptr, nullptr, (unsigned long)M, (unsigned long)K};
    b_gl B_layout{reinterpret_cast<bf16 *>(B), nullptr, nullptr, (unsigned long)K, (unsigned long)N};
    d_gl D_layout{reinterpret_cast<bf16 *>(C), nullptr, nullptr, (unsigned long)M, (unsigned long)N};
    int grid = (M / TILE_M) * (N / TILE_N);
    int smem_size = MAX_SHARED_MEMORY - 1024;
    CHECK_CUDA(cudaFuncSetAttribute(matmul_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    matmul_kernel<<<grid, NUM_THREADS, smem_size, stream>>>(A_layout, B_layout, D_layout, M, N, K);
}
} // namespace tkrect
#endif

void linear_forward_baseline(const float* d_x, const float* d_w, float* d_y, int N, int in_dim, int out_dim, cudaStream_t stream) {

    dim3 block(256);
    dim3 grid((out_dim + block.x - 1) / block.x, N);
    linear_kernel<<<grid, block, 0, stream>>>(d_x, d_w, d_y, N, in_dim, out_dim);
}

#if USE_TK_GEMM
bool tk_gemm_supported(int N, int in_dim, int out_dim) {
    return N > 0 && (in_dim % tkrect::TILE_K == 0) && (out_dim % tkrect::TILE_N == 0);
}

void linear_forward_tk_ws(const float* d_x, const __nv_bfloat16* d_w_bf16, float* d_y,
                          __nv_bfloat16* dA, __nv_bfloat16* dC,
                          int N, int in_dim, int out_dim, cudaStream_t stream) {
    if (!d_w_bf16 || !tk_gemm_supported(N, in_dim, out_dim)) {
        return;
    }
    const int Mpad = ((N + tkrect::TILE_M - 1) / tkrect::TILE_M) * tkrect::TILE_M;
    const size_t a_elems = (size_t)Mpad * in_dim;
    const size_t c_elems = (size_t)Mpad * out_dim;

    // Only zero the padded tail rows for A; valid rows will be overwritten by conversion.
    if (Mpad > N) {
        const size_t tail = (size_t)(Mpad - N) * in_dim;
        zero_bf16_kernel<<<(tail + 255) / 256, 256, 0, stream>>>(dA + (size_t)N * in_dim, (int)tail);
    }
    fp32_to_bf16_kernel<<<((size_t)N * in_dim + 255) / 256, 256, 0, stream>>>(d_x, dA, N * in_dim);
    tkrect::matmul_rect(dA, const_cast<__nv_bfloat16*>(d_w_bf16), dC, Mpad, out_dim, in_dim, stream);
    bf16_to_fp32_kernel<<<((size_t)N * out_dim + 255) / 256, 256, 0, stream>>>(dC, d_y, N * out_dim);
}
#endif

void linear_forward(const float* d_x, const float* d_w, float* d_y, int N, int in_dim, int out_dim, cudaStream_t stream) {
    linear_forward_baseline(d_x, d_w, d_y, N, in_dim, out_dim, stream);
}

void expert_forward_gpu_ws(const float* d_x, const ExpertWeights& ew, float* d_hidden, float* d_out,
                           __nv_bfloat16* d_tk_a, __nv_bfloat16* d_tk_c,
                           int N, int H, int I, cudaStream_t stream) {
    if (N == 0) return;
#if USE_TK_GEMM
    if (d_tk_a && d_tk_c && ew.d_w1_bf16 && ew.d_w2_bf16 && tk_gemm_supported(N, H, I)) {
        linear_forward_tk_ws(d_x, ew.d_w1_bf16, d_hidden, d_tk_a, d_tk_c, N, H, I, stream);
        gelu_kernel<<<((size_t)N * I + 255) / 256, 256, 0, stream>>>(d_hidden, N * I);
        linear_forward_tk_ws(d_hidden, ew.d_w2_bf16, d_out, d_tk_a, d_tk_c, N, I, H, stream);
        return;
    }
#endif
    linear_forward_baseline(d_x, ew.d_w1, d_hidden, N, H, I, stream);
    gelu_kernel<<<((size_t)N * I + 255) / 256, 256, 0, stream>>>(d_hidden, N * I);
    linear_forward_baseline(d_hidden, ew.d_w2, d_out, N, I, H, stream);
}

void expert_forward_gpu(const float* d_x, const ExpertWeights& ew, float* d_out, int N, int H, int I, cudaStream_t stream) {
    if (N == 0) return;
    float* d_hidden = nullptr;
    CHECK_CUDA(cudaMalloc(&d_hidden, (size_t)N * I * sizeof(float)));
    expert_forward_gpu_ws(d_x, ew, d_hidden, d_out, nullptr, nullptr, N, H, I, stream);
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

void init_matrix_device(float*& d_ptr, __nv_bfloat16*& d_ptr_bf16, int rows, int cols, int seed, int device) {
    std::vector<float> h((size_t)rows * cols);
    std::vector<__nv_bfloat16> h_bf16((size_t)rows * cols);
    for (int i = 0; i < rows * cols; ++i) {
        h[i] = rand_uniform_from_seed(seed, i);
        h_bf16[i] = __float2bfloat16(h[i]);
    }
    CHECK_CUDA(cudaSetDevice(device));
    CHECK_CUDA(cudaMalloc(&d_ptr, (size_t)rows * cols * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_ptr, h.data(), (size_t)rows * cols * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMalloc(&d_ptr_bf16, (size_t)rows * cols * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMemcpy(d_ptr_bf16, h_bf16.data(), (size_t)rows * cols * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
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
    CHECK_CUDA(cudaMalloc(&rb.d_shared_tmp, (size_t)cfg.T_local * cfg.H * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&rb.d_shared_hidden, (size_t)cfg.T_local * cfg.I * sizeof(float)));
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
    CHECK_CUDA(cudaMalloc(&rb.d_rows_buf, (size_t)total_slots * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&rb.d_scales_buf, (size_t)total_slots * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&rb.d_expert_in, (size_t)total_slots * cfg.H * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&rb.d_expert_hidden, (size_t)total_slots * cfg.I * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&rb.d_expert_out, (size_t)total_slots * cfg.H * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&rb.d_tk_a, (size_t)total_slots * std::max(cfg.H, cfg.I) * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&rb.d_tk_c, (size_t)total_slots * std::max(cfg.H, cfg.I) * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&rb.d_topk_idx, (size_t)cfg.T_local * cfg.topk * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&rb.d_topk_w, (size_t)cfg.T_local * cfg.topk * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&rb.d_count, sizeof(int)));

    CHECK_CUDA(cudaMalloc(&rb.d_back_send_hidden, (size_t)total_slots * cfg.H * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&rb.d_back_send_tok, (size_t)total_slots * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&rb.d_back_recv_hidden, (size_t)total_slots * cfg.H * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&rb.d_back_recv_tok, (size_t)total_slots * sizeof(int)));
}

void free_rank_buffers(RankBuffers& rb) {
    cudaFree(rb.d_x);
    cudaFree(rb.d_logits);
    cudaFree(rb.d_shared_out);
    cudaFree(rb.d_shared_tmp);
    cudaFree(rb.d_shared_hidden);
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
    cudaFree(rb.d_rows_buf);
    cudaFree(rb.d_scales_buf);
    cudaFree(rb.d_expert_in);
    cudaFree(rb.d_expert_hidden);
    cudaFree(rb.d_expert_out);
    cudaFree(rb.d_tk_a);
    cudaFree(rb.d_tk_c);
    cudaFree(rb.d_topk_idx);
    cudaFree(rb.d_topk_w);
    cudaFree(rb.d_count);
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

        // A. shared branch + router logits.
        for (int rank = 0; rank < nranks; ++rank) {
            CHECK_CUDA(cudaSetDevice(rank));
            zero_kernel<<<((size_t)cfg.T_local * cfg.H + 255) / 256, 256, 0, streams[rank]>>>(rb[rank].d_shared_out, cfg.T_local * cfg.H);
            for (int se = 0; se < cfg.shared; ++se) {
                expert_forward_gpu_ws(rb[rank].d_x, shared_experts[rank][se], rb[rank].d_shared_hidden, rb[rank].d_shared_tmp, rb[rank].d_tk_a, rb[rank].d_tk_c, cfg.T_local, cfg.H, cfg.I, streams[rank]);
                add_inplace_kernel<<<((size_t)cfg.T_local * cfg.H + 255) / 256, 256, 0, streams[rank]>>>(rb[rank].d_shared_out, rb[rank].d_shared_tmp, cfg.T_local * cfg.H);
            }
            linear_forward(rb[rank].d_x, d_gate_w[rank], rb[rank].d_logits, cfg.T_local, cfg.H, cfg.E, streams[rank]);
        }
        for (int rank = 0; rank < nranks; ++rank) {
            CHECK_CUDA(cudaSetDevice(rank));
            sync_stream(streams[rank], "shared+router");

            std::vector<float> h_logits((size_t)cfg.T_local * cfg.E);
            CHECK_CUDA(cudaMemcpy(h_logits.data(), rb[rank].d_logits, h_logits.size() * sizeof(float), cudaMemcpyDeviceToHost));
            router_topk_host(h_logits, cfg.T_local, cfg.E, cfg.topk, all_topk_idx[rank], all_topk_w[rank]);

            CHECK_CUDA(cudaMemcpy(rb[rank].d_topk_idx, all_topk_idx[rank].data(), (size_t)cfg.T_local * cfg.topk * sizeof(int), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(rb[rank].d_topk_w, all_topk_w[rank].data(), (size_t)cfg.T_local * cfg.topk * sizeof(float), cudaMemcpyHostToDevice));
        }

        // B. pack dispatch buffers on device.
        for (int rank = 0; rank < nranks; ++rank) {
            CHECK_CUDA(cudaSetDevice(rank));
            int threads = 128;
            int total = total_slots;
            int blocks = (total + threads - 1) / threads;
            pack_dispatch_kernel<<<blocks, threads, 0, streams[rank]>>>(
                rb[rank].d_x, rb[rank].d_topk_idx, rb[rank].d_topk_w,
                rb[rank].d_send_hidden, rb[rank].d_send_tok, rb[rank].d_send_eid, rb[rank].d_send_src, rb[rank].d_send_w,
                cfg.T_local, cfg.H, cfg.topk, experts_per_rank, capacity_slots, rank, nranks);
            sync_stream(streams[rank], "pack dispatch");
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

            for (int lid = 0; lid < experts_per_rank; ++lid) {
                CHECK_CUDA(cudaMemsetAsync(rb[rank].d_count, 0, sizeof(int), streams[rank]));
                int threads = 256;
                int blocks = (total_slots + threads - 1) / threads;
                build_rows_scales_kernel<<<blocks, threads, 0, streams[rank]>>>(
                    rb[rank].d_recv_eid, rb[rank].d_recv_w, total_slots, experts_per_rank, lid,
                    rb[rank].d_count, rb[rank].d_rows_buf, rb[rank].d_scales_buf);
                int nrows = 0;
                CHECK_CUDA(cudaMemcpyAsync(&nrows, rb[rank].d_count, sizeof(int), cudaMemcpyDeviceToHost, streams[rank]));
                sync_stream(streams[rank], "build rows");
                if (nrows == 0) continue;

                dim3 block(256);
                dim3 grid((cfg.H + block.x - 1) / block.x, nrows);
                gather_rows_kernel<<<grid, block, 0, streams[rank]>>>(rb[rank].d_recv_hidden, rb[rank].d_rows_buf, rb[rank].d_expert_in, nrows, cfg.H, total_slots);
                expert_forward_gpu_ws(rb[rank].d_expert_in, local_experts[rank][lid], rb[rank].d_expert_hidden, rb[rank].d_expert_out, rb[rank].d_tk_a, rb[rank].d_tk_c, nrows, cfg.H, cfg.I, streams[rank]);
                scale_rows_kernel<<<grid, block, 0, streams[rank]>>>(rb[rank].d_expert_out, rb[rank].d_scales_buf, nrows, cfg.H);
                scatter_rows_kernel<<<grid, block, 0, streams[rank]>>>(rb[rank].d_expert_out, rb[rank].d_rows_buf, rb[rank].d_local_out, nrows, cfg.H, total_slots);
                sync_stream(streams[rank], "local expert compute");
            }
        }

        // E. pack return buffers (device-to-device, preserving peer-segment layout).
        for (int rank = 0; rank < nranks; ++rank) {
            CHECK_CUDA(cudaSetDevice(rank));
            CHECK_CUDA(cudaMemcpyAsync(rb[rank].d_back_send_hidden, rb[rank].d_local_out, (size_t)total_slots * cfg.H * sizeof(float), cudaMemcpyDeviceToDevice, streams[rank]));
            CHECK_CUDA(cudaMemcpyAsync(rb[rank].d_back_send_tok, rb[rank].d_recv_tok, (size_t)total_slots * sizeof(int), cudaMemcpyDeviceToDevice, streams[rank]));
            sync_stream(streams[rank], "pack return");
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
            init_matrix_device(shared_experts[rank][se].d_w1, shared_experts[rank][se].d_w1_bf16, cfg.H, cfg.I, 1000 + rank * 97 + se * 3, rank);
            init_matrix_device(shared_experts[rank][se].d_w2, shared_experts[rank][se].d_w2_bf16, cfg.I, cfg.H, 2000 + rank * 97 + se * 3, rank);
        }
        for (int lid = 0; lid < experts_per_rank; ++lid) {
            int eid = rank * experts_per_rank + lid;
            init_matrix_device(local_experts[rank][lid].d_w1, local_experts[rank][lid].d_w1_bf16, cfg.H, cfg.I, 3000 + eid * 11, rank);
            init_matrix_device(local_experts[rank][lid].d_w2, local_experts[rank][lid].d_w2_bf16, cfg.I, cfg.H, 4000 + eid * 11, rank);
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
            cudaFree(shared_experts[rank][se].d_w1_bf16);
            cudaFree(shared_experts[rank][se].d_w2_bf16);
        }
        for (int lid = 0; lid < experts_per_rank; ++lid) {
            cudaFree(local_experts[rank][lid].d_w1);
            cudaFree(local_experts[rank][lid].d_w2);
            cudaFree(local_experts[rank][lid].d_w1_bf16);
            cudaFree(local_experts[rank][lid].d_w2_bf16);
        }
        ncclCommDestroy(comms[rank]);
        cudaStreamDestroy(streams[rank]);
    }
    return 0;
}
