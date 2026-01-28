#pragma once

#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <const int BM, const int BN, const int BK, const int TM>
__global__ void sgemm1DBlocktiling(int M, int N, int K, float alpha,
                                  const float *A, const float *B, float beta,
                                  float *C) {
  // Same mapping as the repo code (note x/y flipped for better locality)
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  // thread layout
  const int threadCol = (int)threadIdx.x % BN;
  const int threadRow = (int)threadIdx.x / BN;

  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // Move pointers to this C tile
  const float* Abase = A + (size_t)cRow * BM * (size_t)K;
  const float* Bbase = B + (size_t)cCol * BN;
  float* Cbase = C + (size_t)cRow * BM * (size_t)N + (size_t)cCol * BN;

  // Map each thread to one element in A tile and one element in B tile
  // (works only when BM==BN and BM*BK==blockDim.x)
  const uint innerColA = threadIdx.x % BK;
  const uint innerRowA = threadIdx.x / BK;
  const uint innerColB = threadIdx.x % BN;
  const uint innerRowB = threadIdx.x / BN;

  float threadResults[TM];
#pragma unroll
  for (int i = 0; i < TM; ++i) threadResults[i] = 0.0f;

  // Loop over K tiles
  for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // ----- load As -----
    {
      const int gRowA = (int)cRow * BM + (int)innerRowA;
      const int gColA = bkIdx + (int)innerColA;
      float aVal = 0.0f;
      if (gRowA < M && gColA < K) {
        aVal = Abase[(size_t)innerRowA * (size_t)K + (size_t)gColA];
      }
      As[(size_t)innerRowA * BK + innerColA] = aVal;
    }

    // ----- load Bs -----
    {
      const int gRowB = bkIdx + (int)innerRowB;
      const int gColB = (int)cCol * BN + (int)innerColB;
      float bVal = 0.0f;
      if (gRowB < K && gColB < N) {
        bVal = Bbase[(size_t)gRowB * (size_t)N + (size_t)innerColB];
      }
      Bs[(size_t)innerRowB * BN + innerColB] = bVal;
    }

    __syncthreads();

    // compute
#pragma unroll
    for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
      const float tmpB = Bs[(size_t)dotIdx * BN + (size_t)threadCol];
#pragma unroll
      for (int resIdx = 0; resIdx < TM; ++resIdx) {
        const int r = threadRow * TM + resIdx; // row within BM
        threadResults[resIdx] += As[(size_t)r * BK + (size_t)dotIdx] * tmpB;
      }
    }

    __syncthreads();
  }

  // write back
#pragma unroll
  for (int resIdx = 0; resIdx < TM; ++resIdx) {
    const int gRowC = (int)cRow * BM + (threadRow * TM + resIdx);
    const int gColC = (int)cCol * BN + threadCol;
    if (gRowC < M && gColC < N) {
      float* cptr = Cbase + (size_t)(threadRow * TM + resIdx) * (size_t)N + (size_t)threadCol;
      *cptr = alpha * threadResults[resIdx] + beta * (*cptr);
    }
  }
}
