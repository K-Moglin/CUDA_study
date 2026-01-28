#pragma once
#include <cuda_runtime.h>

#ifndef BLOCKSIZE
#define BLOCKSIZE 32
#endif

// Kernel 2: Global Memory Coalescing
__global__ void sgemm_coalescing(int M, int N, int K, float alpha,
                                 const float* A, const float* B,
                                 float beta, float* C) {
  // 1D block: 1024 threads
  const int x = (int)blockIdx.x * BLOCKSIZE + ((int)threadIdx.x / BLOCKSIZE);
  const int y = (int)blockIdx.y * BLOCKSIZE + ((int)threadIdx.x % BLOCKSIZE);

  if (x < M && y < N) {
    float tmp = 0.0f;
    for (int i = 0; i < K; ++i) {
      tmp += A[x * K + i] * B[i * N + y];
    }
    C[x * N + y] = alpha * tmp + beta * C[x * N + y];
  }
}
