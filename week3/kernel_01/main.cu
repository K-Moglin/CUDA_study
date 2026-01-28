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

static inline int ceil_div(int x, int y) { return (x + y - 1) / y; }

int main(int argc, char** argv) {
  // Defaults: moderate size, fast run
  int M = 2048, N = 2048, K = 2048;
  int iters = 50;

  if (argc >= 4) { M = std::atoi(argv[1]); N = std::atoi(argv[2]); K = std::atoi(argv[3]); }
  if (argc >= 5) { iters = std::atoi(argv[4]); }

  float alpha = 1.0f, beta = 0.0f;

  size_t bytesA = (size_t)M * K * sizeof(float);
  size_t bytesB = (size_t)K * N * sizeof(float);
  size_t bytesC = (size_t)M * N * sizeof(float);

  std::vector<float> hA((size_t)M*K), hB((size_t)K*N), hC((size_t)M*N);

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

  dim3 block(32, 32, 1);
  dim3 grid(ceil_div(M, 32), ceil_div(N, 32), 1);

  // Warmup
  sgemm_naive<<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // Timing
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) {
    sgemm_naive<<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms_total = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms_total, start, stop));
  float ms = ms_total / iters;

  // FLOPs for GEMM: approx 2*M*N*K (+ M*N for beta*C add;这里 beta=0 默认影响不大)
  double flops = 2.0 * (double)M * (double)N * (double)K + 1.0 * (double)M * (double)N;
  double gflops = flops / (ms * 1e-3) / 1e9;

  std::cout << "Kernel1 naive | M=" << M << " N=" << N << " K=" << K
            << " | avg " << ms << " ms"
            << " | " << gflops << " GFLOPs/s\n";

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(dA));
  CUDA_CHECK(cudaFree(dB));
  CUDA_CHECK(cudaFree(dC));
  return 0;
}
