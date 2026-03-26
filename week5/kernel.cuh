#pragma once
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cmath>

// CuTe
#include <cute/tensor.hpp>
#include <cute/layout.hpp>

#ifndef CEIL_DIV
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
#endif

// -------------------------
// Simple warp/block reductions (max, sum)
// -------------------------
__inline__ __device__ float warpReduceMax(float v) {
  for (int off = 16; off > 0; off >>= 1) {
    v = fmaxf(v, __shfl_down_sync(0xffffffff, v, off));
  }
  return v;
}

__inline__ __device__ float warpReduceSum(float v) {
  for (int off = 16; off > 0; off >>= 1) {
    v += __shfl_down_sync(0xffffffff, v, off);
  }
  return v;
}

__inline__ __device__ float blockReduceMax(float v) {
  __shared__ float smem[32];       // up to 1024 threads -> 32 warps
  __shared__ float block_result;   // final result visible to all threads

  const int lane = threadIdx.x & 31;
  const int wid  = threadIdx.x >> 5;
  const int num_warps = (blockDim.x + 31) >> 5;  // ceil(blockDim.x / 32)

  v = warpReduceMax(v);

  if (lane == 0) smem[wid] = v;
  __syncthreads();

  if (wid == 0) {
    float out = (lane < num_warps) ? smem[lane] : -CUDART_INF_F;
    out = warpReduceMax(out);
    if (lane == 0) block_result = out;
  }
  __syncthreads();

  return block_result;
}

// Correct block-wide sum reduction.
__inline__ __device__ float blockReduceSum(float v) {
  __shared__ float smem[32];
  __shared__ float block_result;

  const int lane = threadIdx.x & 31;
  const int wid  = threadIdx.x >> 5;
  const int num_warps = (blockDim.x + 31) >> 5;  // ceil(blockDim.x / 32)

  v = warpReduceSum(v);

  if (lane == 0) smem[wid] = v;
  __syncthreads();

  if (wid == 0) {
    float out = (lane < num_warps) ? smem[lane] : 0.f;
    out = warpReduceSum(out);
    if (lane == 0) block_result = out;
  }
  __syncthreads();

  return block_result;
}

// -------------------------
// FlashAttention-2 Algorithm 1 (Forward) — CuTe tensor-view version
// - Single-head
// - FP32
// - One CUDA block computes one row i (for clarity).
//
// Uses CuTe to build tensors with explicit row-major layout:
//   Q: (N,d) stride (d,1)
//   K: (N,d) stride (d,1)
//   V: (N,d) stride (d,1)
//   O: (N,d) stride (d,1)
//   L: (N)
//
// Algorithm 1 core:
// for each KV tile:
//   tile_m = max(score)
//   m_new = max(m_old, tile_m)
//   rescale old: l *= exp(m_old-m_new), Otilde *= exp(m_old-m_new)
//   p_j = exp(score - m_new)
//   Otilde += sum p_j * V
//   l += sum p_j
// end
// O = Otilde / l ; L = m + log(l)
// -------------------------
template<int Bc, int BLOCK_THREADS>
__global__ void fa2_alg1_forward_cute_rowwise(
    const float* __restrict__ Qp,
    const float* __restrict__ Kp,
    const float* __restrict__ Vp,
    float* __restrict__ Op,
    float* __restrict__ Lp,
    int N, int d,
    int causal
) {
  using namespace cute;

  const int i = (int)blockIdx.x;
  if (i >= N) return;

  // Build CuTe tensors (dynamic shapes/strides), row-major
  auto layout2d = make_layout(make_shape(N, d), make_stride(d, 1));
  auto gQ = make_tensor(make_gmem_ptr(Qp), layout2d);
  auto gK = make_tensor(make_gmem_ptr(Kp), layout2d);
  auto gV = make_tensor(make_gmem_ptr(Vp), layout2d);
  auto gO = make_tensor(make_gmem_ptr(Op), layout2d);
  auto gL = make_tensor(make_gmem_ptr(Lp), make_layout(make_shape(N), make_stride(1)));

  const float inv_sqrt_d = rsqrtf((float)d);

  // Shared storage for this tile
  __shared__ float sh_scores[Bc];
  __shared__ float sh_probs[Bc];

  // Running softmax state for row i
  float m = -CUDART_INF_F;
  float l = 0.f;

  // Use gO(i, :) as Otilde buffer during accumulation
  for (int k = threadIdx.x; k < d; k += BLOCK_THREADS) {
    gO(i, k) = 0.f;
  }
  __syncthreads();

  const int tiles = CEIL_DIV(N, Bc);

  for (int t = 0; t < tiles; ++t) {
    const int j0 = t * Bc;

    // 1) Compute scores for current tile
    if (threadIdx.x < Bc) {
      const int j = j0 + threadIdx.x;
      float s = -CUDART_INF_F;

      if (j < N && !(causal && j > i)) {
        float dot = 0.f;
        for (int k = 0; k < d; ++k) {
          dot += gQ(i, k) * gK(j, k);
        }
        s = dot * inv_sqrt_d;
      }
      sh_scores[threadIdx.x] = s;
    }
    __syncthreads();

    // 2) Tile max
    float local_max = -CUDART_INF_F;
    if (threadIdx.x < Bc) local_max = sh_scores[threadIdx.x];
    float tile_m = blockReduceMax(local_max);

    // Fully masked tile: skip
    if (!isfinite(tile_m)) {
      __syncthreads();
      continue;
    }

    // 3) Update running max and rescale previous accumulators
    const float m_old = m;
    const float m_new = fmaxf(m_old, tile_m);
    const float scale_old = isfinite(m_old) ? expf(m_old - m_new) : 0.f;

    l *= scale_old;

    for (int k = threadIdx.x; k < d; k += BLOCK_THREADS) {
      gO(i, k) *= scale_old;
    }
    __syncthreads();

    // 4) Compute probabilities and tile_l
    if (threadIdx.x < Bc) {
      const float sj = sh_scores[threadIdx.x];
      const float p = isfinite(sj) ? expf(sj - m_new) : 0.f;
      sh_probs[threadIdx.x] = p;
    }
    __syncthreads();

    float local_sum = 0.f;
    if (threadIdx.x < Bc) local_sum = sh_probs[threadIdx.x];
    float tile_l = blockReduceSum(local_sum);

    // 5) Otilde += sum_j p_j * V_j
    for (int k = threadIdx.x; k < d; k += BLOCK_THREADS) {
      float acc = gO(i, k);
      #pragma unroll
      for (int cj = 0; cj < Bc; ++cj) {
        const int j = j0 + cj;
        if (j >= N) break;
        const float pj = sh_probs[cj];
        if (pj != 0.f) {
          acc += pj * gV(j, k);
        }
      }
      gO(i, k) = acc;
    }
    __syncthreads();

    // 6) Update running state
    l += tile_l;
    m = m_new;
    __syncthreads();
  }

  // Finalize
  if (!(l > 0.f) || !isfinite(l) || !isfinite(m)) {
    for (int k = threadIdx.x; k < d; k += BLOCK_THREADS) {
      gO(i, k) = 0.f;
    }
    if (threadIdx.x == 0) {
      gL(i) = -CUDART_INF_F;
    }
    return;
  }

  const float inv_l = 1.f / l;
  for (int k = threadIdx.x; k < d; k += BLOCK_THREADS) {
    gO(i, k) *= inv_l;
  }

  if (threadIdx.x == 0) {
    gL(i) = m + logf(l);
  }
}
