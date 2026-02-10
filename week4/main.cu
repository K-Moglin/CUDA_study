#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>

#include "kernel.cuh"

static void ck(cudaError_t e, const char* msg) {
  if (e != cudaSuccess) {
    fprintf(stderr, "CUDA error: %s: %s\n", msg, cudaGetErrorString(e));
    std::exit(1);
  }
}

static float frand01() { return (float)rand() / (float)RAND_MAX; }

static void fill_rand(std::vector<float>& x) {
  for (auto& v : x) v = frand01() * 2.f - 1.f;
}

// CPU naive attention for correctness (single-head)
static void cpu_naive_attention(
    const std::vector<float>& Q,
    const std::vector<float>& K,
    const std::vector<float>& V,
    std::vector<float>& O,
    std::vector<float>& L,
    int N, int d, int causal
) {
  const float inv_sqrt_d = 1.f / std::sqrt((float)d);
  std::vector<float> scores(N);

  for (int i = 0; i < N; ++i) {
    float m = -INFINITY;
    for (int j = 0; j < N; ++j) {
      if (causal && j > i) { scores[j] = -INFINITY; continue; }
      float dot = 0.f;
      for (int k = 0; k < d; ++k) dot += Q[i*d+k] * K[j*d+k];
      float s = dot * inv_sqrt_d;
      scores[j] = s;
      m = std::max(m, s);
    }
    float l = 0.f;
    for (int j = 0; j < N; ++j) {
      if (std::isfinite(scores[j])) l += std::exp(scores[j] - m);
    }

    for (int k = 0; k < d; ++k) O[i*d+k] = 0.f;
    if (l > 0.f) {
      for (int j = 0; j < N; ++j) {
        if (!std::isfinite(scores[j])) continue;
        float p = std::exp(scores[j] - m) / l;
        for (int k = 0; k < d; ++k) O[i*d+k] += p * V[j*d+k];
      }
      L[i] = m + std::log(l);
    } else {
      L[i] = -INFINITY;
    }
  }
}

int main(int argc, char** argv) {
  // Usage: ./fa2_cuda [N] [d] [Br(not used)] [Bc] [iters] [causal]
  int N = (argc > 1) ? std::atoi(argv[1]) : 256;
  int d = (argc > 2) ? std::atoi(argv[2]) : 64;
  int Bc = (argc > 3) ? std::atoi(argv[3]) : 128;
  int iters = (argc > 4) ? std::atoi(argv[4]) : 50;
  int causal = (argc > 5) ? std::atoi(argv[5]) : 1;

  if (!(Bc == 64 || Bc == 128 || Bc == 256)) {
    printf("For simplicity, set Bc to 64/128/256\n");
    return 0;
  }
  if (d <= 0 || N <= 0) return 0;

  srand(0);

  std::vector<float> hQ(N*d), hK(N*d), hV(N*d);
  std::vector<float> hO(N*d), hL(N);
  std::vector<float> refO(N*d), refL(N);

  fill_rand(hQ); fill_rand(hK); fill_rand(hV);

  float *dQ=nullptr, *dK=nullptr, *dV=nullptr, *dO=nullptr, *dL=nullptr;
  ck(cudaMalloc(&dQ, (size_t)N*d*sizeof(float)), "malloc Q");
  ck(cudaMalloc(&dK, (size_t)N*d*sizeof(float)), "malloc K");
  ck(cudaMalloc(&dV, (size_t)N*d*sizeof(float)), "malloc V");
  ck(cudaMalloc(&dO, (size_t)N*d*sizeof(float)), "malloc O");
  ck(cudaMalloc(&dL, (size_t)N*sizeof(float)), "malloc L");

  ck(cudaMemcpy(dQ, hQ.data(), (size_t)N*d*sizeof(float), cudaMemcpyHostToDevice), "cpy Q");
  ck(cudaMemcpy(dK, hK.data(), (size_t)N*d*sizeof(float), cudaMemcpyHostToDevice), "cpy K");
  ck(cudaMemcpy(dV, hV.data(), (size_t)N*d*sizeof(float), cudaMemcpyHostToDevice), "cpy V");

  // Launch config: one block per row, BLOCK_THREADS threads
  constexpr int BLOCK_THREADS = 256;
  dim3 block(BLOCK_THREADS);
  dim3 grid(N);

  // Warmup + timing
  cudaEvent_t start, stop;
  ck(cudaEventCreate(&start), "event create");
  ck(cudaEventCreate(&stop), "event create");

  auto launch = [&](){
    if (Bc == 64) {
      fa2_alg1_forward_rowwise<64, BLOCK_THREADS><<<grid, block>>>(dQ,dK,dV,dO,dL,N,d,causal);
    } else if (Bc == 128) {
      fa2_alg1_forward_rowwise<128, BLOCK_THREADS><<<grid, block>>>(dQ,dK,dV,dO,dL,N,d,causal);
    } else {
      fa2_alg1_forward_rowwise<256, BLOCK_THREADS><<<grid, block>>>(dQ,dK,dV,dO,dL,N,d,causal);
    }
  };

  launch();
  ck(cudaGetLastError(), "kernel launch");
  ck(cudaDeviceSynchronize(), "sync warmup");

  ck(cudaEventRecord(start), "record start");
  for (int it = 0; it < iters; ++it) launch();
  ck(cudaEventRecord(stop), "record stop");
  ck(cudaEventSynchronize(stop), "sync stop");

  float ms = 0.f;
  ck(cudaEventElapsedTime(&ms, start, stop), "elapsed");
  ms /= iters;

  ck(cudaMemcpy(hO.data(), dO, (size_t)N*d*sizeof(float), cudaMemcpyDeviceToHost), "cpy O");
  ck(cudaMemcpy(hL.data(), dL, (size_t)N*sizeof(float), cudaMemcpyDeviceToHost), "cpy L");

  // CPU reference
  cpu_naive_attention(hQ, hK, hV, refO, refL, N, d, causal);

  // Compare errors
  float max_abs = 0.f, max_rel = 0.f;
  for (int i = 0; i < N*d; ++i) {
    float a = hO[i], b = refO[i];
    float ae = std::fabs(a - b);
    max_abs = std::max(max_abs, ae);
    float denom = std::max(1e-6f, std::fabs(b));
    max_rel = std::max(max_rel, ae / denom);
  }
  float max_L_abs = 0.f;
  for (int i = 0; i < N; ++i) {
    float ae = std::fabs(hL[i] - refL[i]);
    max_L_abs = std::max(max_L_abs, ae);
  }

  // Rough FLOPs (for reference only): QK (N*N*d*2) + PV (N*N*d*2) ~= 4*N*N*d
  double flops = 4.0 * (double)N * (double)N * (double)d;
  double gflops = flops / (ms * 1e-3) / 1e9;

  printf("FlashAttention-2 Alg1 CUDA (rowwise) test\n");
  printf("N=%d d=%d Bc=%d causal=%d iters=%d\n", N, d, Bc, causal, iters);
  printf("time = %.3f ms   approx GFLOPs = %.2f\n", ms, gflops);
  printf("max |O_gpu - O_cpu| = %.3e\n", max_abs);
  printf("max relative error  = %.3e\n", max_rel);
  printf("max |L_gpu - L_cpu| = %.3e\n", max_L_abs);
  printf("sample L[0] gpu=%.6f cpu=%.6f\n", hL[0], refL[0]);

  cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO); cudaFree(dL);
  cudaEventDestroy(start); cudaEventDestroy(stop);
  return 0;
}
