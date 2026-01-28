#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <iostream>

#include "kernel.cuh"

#define CUDA_CHECK(call) do {                               \
  cudaError_t _e = (call);                                  \
  if (_e != cudaSuccess) {                                  \
    fprintf(stderr, "CUDA error %s:%d: %s\n",               \
            __FILE__, __LINE__, cudaGetErrorString(_e));    \
    std::exit(1);                                           \
  }                                                         \
} while(0)

int main(int argc, char** argv) {
  int M = 4096, N = 4096, K = 4096;
  int iters = 50;
  if (argc >= 4) { M = std::atoi(argv[1]); N = std::atoi(argv[2]); K = std::atoi(argv[3]); }
  if (argc >= 5) { iters = std::atoi(argv[4]); }

  float alpha = 1.0f, beta = 0.0f;

  // Kernel5 params
  constexpr int BM = 128;
  constexpr int BN = 128;
  constexpr int BK = 8;
  constexpr int TM = 8;
  constexpr int TN = 8;

  size_t bytesA = (size_t)M * (size_t)K * sizeof(float);
  size_t bytesB = (size_t)K * (size_t)N * sizeof(float);
  size_t bytesC = (size_t)M * (size_t)N * sizeof(float);

  std::vector<float> hA((size_t)M*(size_t)K), hB((size_t)K*(size_t)N), hC((size_t)M*(size_t)N);

  std::mt19937 rng(123);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& v : hA) v = dist(rng);
  for (auto& v : hB) v = dist(rng);
  for (auto& v : hC) v = dist(rng);

  float *dA=nullptr, *dB=nullptr, *dC=nullptr;
  CUDA_CHECK(cudaMalloc(&dA, bytesA));
  CUDA_CHECK(cudaMalloc(&dB, bytesB));
  CUDA_CHECK(cudaMalloc(&dC, bytesC));

  CUDA_CHECK(cudaMemcpy(dA, hA.data(), bytesA, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB.data(), bytesB, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dC, hC.data(), bytesC, cudaMemcpyHostToDevice));

  // grid: (x = N tiles, y = M tiles) to match kernel mapping
  dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM), 1);

  // block: (BN/TN, BM/TM) threads = 16x16=256
  dim3 block(BN / TN, BM / TM, 1);

  // Warmup
  sgemm2DBlocktiling<BM, BN, BK, TM, TN><<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // Timing
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) {
    sgemm2DBlocktiling<BM, BN, BK, TM, TN><<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms_total = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms_total, start, stop));
  float ms = ms_total / iters;

  double flops = 2.0 * (double)M * (double)N * (double)K + 1.0 * (double)M * (double)N;
  double gflops = flops / (ms * 1e-3) / 1e9;

  std::cout << "Kernel5 2D blocktiling"
            << " | BM="<<BM<<" BN="<<BN<<" BK="<<BK<<" TM="<<TM<<" TN="<<TN
            << " | M="<<M<<" N="<<N<<" K="<<K
            << " | avg "<<ms<<" ms"
            << " | "<<gflops<<" GFLOPs/s\n";

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(dA));
  CUDA_CHECK(cudaFree(dB));
  CUDA_CHECK(cudaFree(dC));
  return 0;
}
