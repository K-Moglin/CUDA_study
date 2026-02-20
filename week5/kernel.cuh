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
// Simple block reductions (max, sum)
// -------------------------
__inline__ __device__ float warpReduceMax(float v) {
  for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffff, v, off));
  return v;
}
__inline__ __device__ float warpReduceSum(float v) {
  for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
  return v;
}
__inline__ __device__ float blockReduceMax(float v) {
  __shared__ float smem[32];
  int lane = threadIdx.x & 31;
  int wid  = threadIdx.x >> 5;

  v = warpReduceMax(v);
  if (lane == 0) smem[wid] = v;
  __syncthreads();

  float out = -CUDART_INF_F;
  if (wid == 0) {
    out = (threadIdx.x < (blockDim.x >> 5)) ? smem[lane] : -CUDART_INF_F;
    out = warpReduceMax(out);
  }
  return __shfl_sync(0xffffffff, out, 0);
}
__inline__ __device__ float blockReduceSum(float v) {
  __shared__ float smem[32];
  int lane = threadIdx.x & 31;
  int wid  = threadIdx.x >> 5;

  v = warpReduceSum(v);
  if (lane == 0) smem[wid] = v;
  __syncthreads();

  float out = 0.f;
  if (wid == 0) {
    out = (threadIdx.x < (blockDim.x >> 5)) ? smem[lane] : 0.f;
    out = warpReduceSum(out);
  }
  return __shfl_sync(0xffffffff, out, 0);
}

// -------------------------
// FlashAttention-2 Algorithm 1 (Forward) — CuTe tensor-view version
// - Single-head
// - FP32
// - One CUDA block computes one row i (for clarity).
//
// Uses CuTe to build tensors with explicit LayoutRight (row-major):
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

  // Build CuTe tensors (dynamic shapes/strides)
  auto layout2d = make_layout(make_shape(N, d), make_stride(d, 1)); // row-major
  auto gQ = make_tensor(make_gmem_ptr(Qp), layout2d);
  auto gK = make_tensor(make_gmem_ptr(Kp), layout2d);
  auto gV = make_tensor(make_gmem_ptr(Vp), layout2d);
  auto gO = make_tensor(make_gmem_ptr(Op), layout2d);

  auto gL = make_tensor(make_gmem_ptr(Lp), make_layout(make_shape(N), make_stride(1)));

  const float inv_sqrt_d = rsqrtf((float)d);

  __shared__ float sh_scores[Bc];
  __shared__ float sh_probs[Bc];

  float m = -CUDART_INF_F;
  float l = 0.f;

  // Use gO row i as Otilde buffer (register-strided update)
  for (int k = threadIdx.x; k < d; k += BLOCK_THREADS) {
    gO(i, k) = 0.f;
  }
  __syncthreads();

  const int tiles = CEIL_DIV(N, Bc);

  for (int t = 0; t < tiles; ++t) {
    const int j0 = t * Bc;

    // 1) scores for this tile
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

    // 2) tile max
    float local_max = -CUDART_INF_F;
    if (threadIdx.x < Bc) local_max = sh_scores[threadIdx.x];
    float tile_m = blockReduceMax(local_max);

    if (!isfinite(tile_m)) {
      __syncthreads();
      continue;
    }

    // 3) update m, rescale old accumulators
    float m_old = m;
    float m_new = fmaxf(m_old, tile_m);
    float scale_old = isfinite(m_old) ? expf(m_old - m_new) : 0.f;

    l *= scale_old;
    for (int k = threadIdx.x; k < d; k += BLOCK_THREADS) {
      gO(i, k) *= scale_old;
    }
    __syncthreads();

    // 4) probs + tile sum
    if (threadIdx.x < Bc) {
      float sj = sh_scores[threadIdx.x];
      float p = isfinite(sj) ? expf(sj - m_new) : 0.f;
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
        int j = j0 + cj;
        if (j >= N) break;
        float pj = sh_probs[cj];
        if (pj != 0.f) acc += pj * gV(j, k);
      }
      gO(i, k) = acc;
    }
    __syncthreads();

    l += tile_l;
    m = m_new;
    __syncthreads();
  }

  // finalize
  if (!(l > 0.f) || !isfinite(l) || !isfinite(m)) {
    for (int k = threadIdx.x; k < d; k += BLOCK_THREADS) gO(i, k) = 0.f;
    if (threadIdx.x == 0) gL(i) = -CUDART_INF_F;
    return;
  }

  float inv_l = 1.f / l;
  for (int k = threadIdx.x; k < d; k += BLOCK_THREADS) gO(i, k) *= inv_l;
  if (threadIdx.x == 0) gL(i) = m + logf(l);
}
