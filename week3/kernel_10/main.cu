#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

#include "kernel.cuh"

static void ck(cudaError_t e, const char* msg) {
  if (e != cudaSuccess) {
    std::fprintf(stderr, "CUDA error: %s: %s\n", msg, cudaGetErrorString(e));
    std::exit(1);
  }
}

static float rand01() {
  return (float)std::rand() / (float)RAND_MAX - 0.5f;
}

int main(int argc, char** argv) {
  int M = 4096, N = 4096, K = 4096, iters = 50;
  if (argc >= 5) {
    M = std::atoi(argv[1]);
    N = std::atoi(argv[2]);
    K = std::atoi(argv[3]);
    iters = std::atoi(argv[4]);
  }
  const float alpha = 1.0f, beta = 0.0f;

  std::printf("Kernel10 (Warp-tiling)  M=%d N=%d K=%d  iters=%d\n", M, N, K, iters);

  size_t bytesA = (size_t)M * K * sizeof(float);
  size_t bytesB = (size_t)K * N * sizeof(float);
  size_t bytesC = (size_t)M * N * sizeof(float);

  std::vector<float> hA((size_t)M * K), hB((size_t)K * N), hC((size_t)M * N);

  std::srand(0);
  for (auto& x : hA) x = rand01();
  for (auto& x : hB) x = rand01();
  for (auto& x : hC) x = 0.0f;

  float *dA=nullptr, *dB=nullptr, *dC=nullptr;
  ck(cudaMalloc(&dA, bytesA), "malloc A");
  ck(cudaMalloc(&dB, bytesB), "malloc B");
  ck(cudaMalloc(&dC, bytesC), "malloc C");
  ck(cudaMemcpy(dA, hA.data(), bytesA, cudaMemcpyHostToDevice), "copy A");
  ck(cudaMemcpy(dB, hB.data(), bytesB, cudaMemcpyHostToDevice), "copy B");
  ck(cudaMemcpy(dC, hC.data(), bytesC, cudaMemcpyHostToDevice), "copy C");

  // Params (can tune later)
  constexpr int BM = 128;
  constexpr int BN = 128;
  constexpr int BK = 8;
  constexpr int WM = 64;
  constexpr int WN = 32;
  constexpr int TM = 8;
  constexpr int TN = 8;

  dim3 block(BN / TN, BM / TM);               // (16, 16) => 256 threads
  dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));

  // Warmup
  sgemmWarptiling<BM, BN, BK, WM, WN, TM, TN><<<grid, block>>>(
      M, N, K, alpha, dA, dB, beta, dC);
  ck(cudaGetLastError(), "kernel launch warmup");
  ck(cudaDeviceSynchronize(), "sync warmup");

  cudaEvent_t start, stop;
  ck(cudaEventCreate(&start), "event create start");
  ck(cudaEventCreate(&stop), "event create stop");

  ck(cudaEventRecord(start), "record start");
  for (int i = 0; i < iters; ++i) {
    sgemmWarptiling<BM, BN, BK, WM, WN, TM, TN><<<grid, block>>>(
        M, N, K, alpha, dA, dB, beta, dC);
  }
  ck(cudaEventRecord(stop), "record stop");
  ck(cudaEventSynchronize(stop), "sync stop");

  float ms = 0.0f;
  ck(cudaEventElapsedTime(&ms, start, stop), "elapsed time");

  double avg_ms = (double)ms / iters;
  double flops = 2.0 * (double)M * (double)N * (double)K;
  double gflops = (flops * 1e-9) / (avg_ms * 1e-3);

  std::printf("Avg time: %.4f ms, GFLOPs: %.2f\n", avg_ms, gflops);

  ck(cudaEventDestroy(start), "destroy start");
  ck(cudaEventDestroy(stop), "destroy stop");
  ck(cudaFree(dA), "free A");
  ck(cudaFree(dB), "free B");
  ck(cudaFree(dC), "free C");
  return 0;
}
