#pragma once
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cmath>

#ifndef CEIL_DIV
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
#endif

// -------------------------
// Warp / Block reductions
// -------------------------
__inline__ __device__ float warpReduceMax(float val) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
  }
  return val;
}

__inline__ __device__ float warpReduceSum(float val) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

__inline__ __device__ float blockReduceMax(float val) {
  __shared__ float shared[32]; // up to 1024 threads -> 32 warps
  int lane = threadIdx.x & 31;
  int wid  = threadIdx.x >> 5;

  val = warpReduceMax(val);
  if (lane == 0) shared[wid] = val;
  __syncthreads();

  // final reduce in warp 0
  float out = -CUDART_INF_F;
  if (wid == 0) {
    out = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : -CUDART_INF_F;
    out = warpReduceMax(out);
  }
  return __shfl_sync(0xffffffff, out, 0);
}

__inline__ __device__ float blockReduceSum(float val) {
  __shared__ float shared[32];
  int lane = threadIdx.x & 31;
  int wid  = threadIdx.x >> 5;

  val = warpReduceSum(val);
  if (lane == 0) shared[wid] = val;
  __syncthreads();

  float out = 0.f;
  if (wid == 0) {
    out = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : 0.f;
    out = warpReduceSum(out);
  }
  return __shfl_sync(0xffffffff, out, 0);
}

// -------------------------
// FlashAttention-2 Alg.1 forward (simplified, single-head, FP32)
// Each CUDA block computes ONE row i of O (shape [d]).
// It iterates over K/V in tiles of Bc columns.
//
// For each tile:
//   - compute scores s_j = <Q_i, K_j>/sqrt(d) (masked if causal && j>i)
//   - tile_m = max_j s_j
//   - m_new = max(m_old, tile_m)
//   - rescale old accumulators: l *= exp(m_old-m_new), Otilde *= same
//   - compute p_j = exp(s_j - m_new), tile_l = sum p_j
//   - Otilde += sum_j p_j * V_j
// Finally:
//   O = Otilde / l
//   L = m + log(l)
// -------------------------
template<int Bc, int BLOCK_THREADS>
__global__ void fa2_alg1_forward_rowwise(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    float* __restrict__ L,
    int N, int d,
    int causal
) {
  const int i = (int)blockIdx.x;
  if (i >= N) return;

  const float inv_sqrt_d = rsqrtf((float)d);

  // Shared for scores/probs for this tile
  __shared__ float sh_scores[Bc];
  __shared__ float sh_probs[Bc];

  // Running state (scalar per row)
  float m = -CUDART_INF_F;
  float l = 0.f;

  // Otilde in registers (thread-strided over k)
  // Otilde[k] maintained across tiles
  // We store only the slice handled by this thread.
  // Access pattern: k = tid, tid+BLOCK_THREADS, ...
  for (int k = threadIdx.x; k < d; k += BLOCK_THREADS) {
    O[i * d + k] = 0.f; // temporarily use O as Otilde buffer, we'll normalize at end
  }
  __syncthreads();

  const float* Qi = Q + (size_t)i * d;

  const int numTiles = CEIL_DIV(N, Bc);

  for (int t = 0; t < numTiles; ++t) {
    const int j0 = t * Bc;

    // 1) compute score for each j in tile (parallel over threads)
    float s = -CUDART_INF_F;
    if (threadIdx.x < Bc) {
      const int j = j0 + threadIdx.x;
      if (j < N && !(causal && j > i)) {
        const float* Kj = K + (size_t)j * d;
        float dot = 0.f;
        for (int k = 0; k < d; ++k) dot += Qi[k] * Kj[k];
        s = dot * inv_sqrt_d;
      }
      sh_scores[threadIdx.x] = s;
    }
    __syncthreads();

    // 2) tile max
    float local_max = -CUDART_INF_F;
    if (threadIdx.x < Bc) local_max = sh_scores[threadIdx.x];
    float tile_m = blockReduceMax(local_max);

    // If tile fully masked (tile_m = -inf), skip
    if (!isfinite(tile_m)) {
      __syncthreads();
      continue;
    }

    // 3) update running max and rescale old accumulators
    const float m_old = m;
    const float m_new = fmaxf(m_old, tile_m);

    const float scale_old = isfinite(m_old) ? expf(m_old - m_new) : 0.f;

    // rescale l
    l *= scale_old;

    // rescale Otilde (stored in O buffer)
    for (int k = threadIdx.x; k < d; k += BLOCK_THREADS) {
      float ot = O[i * d + k];
      ot *= scale_old;
      O[i * d + k] = ot;
    }
    __syncthreads();

    // 4) compute p_j = exp(s_j - m_new) and tile_l
    float p = 0.f;
    if (threadIdx.x < Bc) {
      const float sj = sh_scores[threadIdx.x];
      p = isfinite(sj) ? expf(sj - m_new) : 0.f;
      sh_probs[threadIdx.x] = p;
    }
    __syncthreads();

    float local_sum = 0.f;
    if (threadIdx.x < Bc) local_sum = sh_probs[threadIdx.x];
    float tile_l = blockReduceSum(local_sum);

    // 5) Otilde += sum_j p_j * V_j  (parallel over k, loop over j)
    for (int k = threadIdx.x; k < d; k += BLOCK_THREADS) {
      float acc = O[i * d + k];
      // iterate j in tile
      for (int cj = 0; cj < Bc; ++cj) {
        const int j = j0 + cj;
        if (j >= N) break;
        const float pj = sh_probs[cj];
        if (pj != 0.f) {
          acc += pj * V[(size_t)j * d + k];
        }
      }
      O[i * d + k] = acc;
    }
    __syncthreads();

    // 6) update running l and m
    l += tile_l;
    m = m_new;
    __syncthreads();
  }

  // finalize: O = Otilde / l, L = m + log(l)
  if (!(l > 0.f) || !isfinite(l) || !isfinite(m)) {
    for (int k = threadIdx.x; k < d; k += BLOCK_THREADS) {
      O[i * d + k] = 0.f;
    }
    if (threadIdx.x == 0) L[i] = -CUDART_INF_F;
    return;
  }

  const float inv_l = 1.0f / l;
  for (int k = threadIdx.x; k < d; k += BLOCK_THREADS) {
    O[i * d + k] *= inv_l;
  }
  if (threadIdx.x == 0) {
    L[i] = m + logf(l);
  }
}
