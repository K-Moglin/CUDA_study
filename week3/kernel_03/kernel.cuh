#pragma once
#include <cuda_runtime.h>

#ifndef BLOCKSIZE
#define BLOCKSIZE 32
#endif

// Kernel 3: SMEM Caching (classic tiled SGEMM)
__global__ void sgemm_smem(int M, int N, int K, float alpha,
                           const float* A, const float* B,
                           float beta, float* C) {
  // Block coordinates
  const int row = (int)blockIdx.x * BLOCKSIZE + (int)threadIdx.y;
  const int col = (int)blockIdx.y * BLOCKSIZE + (int)threadIdx.x;

  // Shared tiles
  __shared__ float As[BLOCKSIZE][BLOCKSIZE];
  __shared__ float Bs[BLOCKSIZE][BLOCKSIZE];

  float tmp = 0.0f;

  // Loop over tiles of K
  const int numTiles = (K + BLOCKSIZE - 1) / BLOCKSIZE;
  for (int t = 0; t < numTiles; ++t) {
    const int a_col = t * BLOCKSIZE + (int)threadIdx.x; // along K
    const int b_row = t * BLOCKSIZE + (int)threadIdx.y; // along K

    // Load A tile: A[row, a_col]
    if (row < M && a_col < K) {
      As[threadIdx.y][threadIdx.x] = A[row * K + a_col];
    } else {
      As[threadIdx.y][threadIdx.x] = 0.0f;
    }

    // Load B tile: B[b_row, col]
    if (b_row < K && col < N) {
      Bs[threadIdx.y][threadIdx.x] = B[b_row * N + col];
    } else {
      Bs[threadIdx.y][threadIdx.x] = 0.0f;
    }

    __syncthreads();

    // Compute partial dot using SMEM
    #pragma unroll
    for (int i = 0; i < BLOCKSIZE; ++i) {
      tmp += As[threadIdx.y][i] * Bs[i][threadIdx.x];
    }

    __syncthreads();
  }

  // Write back
  if (row < M && col < N) {
    C[row * N + col] = alpha * tmp + beta * C[row * N + col];
  }
}
