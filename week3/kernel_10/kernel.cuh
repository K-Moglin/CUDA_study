#pragma once
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

// Kernel 10: Warp-tiling (simple, correctness-first version)
// Block tile: BM x BN
// Warp tile : WM x WN   (WM divides BM, WN divides BN)
// Per-thread: TM x TN   (TM divides WM, TN divides WN)
// Warp threads = (WM/TM) * (WN/TN)  must be 32 (1 warp)
//
// Recommended parameters used below in main.cu:
// BM=128, BN=128, BK=8, WM=64, WN=32, TM=8, TN=8
// - WM/TM = 8
// - WN/TN = 4
// - 8*4 = 32 threads per warp 
// - warps per block = (BM/WM)*(BN/WN) = 2*4 = 8 warps -> 256 threads 

template <int BM, int BN, int BK, int WM, int WN, int TM, int TN>
__global__ void sgemmWarptiling(int M, int N, int K, float alpha,
                                const float* __restrict__ A,
                                const float* __restrict__ B,
                                float beta,
                                float* __restrict__ C) {
  // 2D threadblock:
  // blockDim.x = BN/TN, blockDim.y = BM/TM
  const int tx = (int)threadIdx.x;  // 0..(BN/TN-1)
  const int ty = (int)threadIdx.y;  // 0..(BM/TM-1)

  const int cRow = (int)blockIdx.y; // block tile row id
  const int cCol = (int)blockIdx.x; // block tile col id

  const int blockRowBase = cRow * BM;
  const int blockColBase = cCol * BN;

  // warp tiling decomposition inside the block
  constexpr int warpTilesN = BN / WN;         // number of warp tiles along N inside a block
  constexpr int warpTilesM = BM / WM;         // number of warp tiles along M inside a block
  constexpr int warpThreadsX = WN / TN;       // threads along N inside a warp
  constexpr int warpThreadsY = WM / TM;       // threads along M inside a warp
  static_assert(warpThreadsX * warpThreadsY == 32, "Warp tile must map to exactly 32 threads");

  const int warpCol = tx / warpThreadsX;      // 0..(warpTilesN-1)
  const int warpRow = ty / warpThreadsY;      // 0..(warpTilesM-1)

  const int laneX = tx % warpThreadsX;        // 0..(warpThreadsX-1)
  const int laneY = ty % warpThreadsY;        // 0..(warpThreadsY-1)

  // This thread's output tile base (in C coordinates)
  const int warpRowBase = blockRowBase + warpRow * WM;
  const int warpColBase = blockColBase + warpCol * WN;

  const int threadRowBase = warpRowBase + laneY * TM;
  const int threadColBase = warpColBase + laneX * TN;

  // Shared memory tiles
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // Register accumulator
  float acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i) {
#pragma unroll
    for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;
  }

  // Flatten thread id for cooperative loads
  const int tLinear = ty * (BN / TN) + tx;
  const int numThreads = (BM / TM) * (BN / TN); // should be 256
  (void)numThreads;

  // How many floats to load into As and Bs each iteration
  // As: BM*BK floats, Bs: BK*BN floats
  // We do simple scalar loads with bounds checks (correctness-first).
  for (int kBase = 0; kBase < K; kBase += BK) {
    // Load A tile to shared: As[m, k] = A[(blockRowBase+m), (kBase+k)]
    for (int idx = tLinear; idx < BM * BK; idx += (BM / TM) * (BN / TN)) {
      const int m = idx / BK;
      const int k = idx % BK;
      const int gRow = blockRowBase + m;
      const int gCol = kBase + k;
      As[m * BK + k] = (gRow < M && gCol < K) ? A[(size_t)gRow * K + gCol] : 0.0f;
    }

    // Load B tile to shared: Bs[k, n] = B[(kBase+k), (blockColBase+n)]
    for (int idx = tLinear; idx < BK * BN; idx += (BM / TM) * (BN / TN)) {
      const int k = idx / BN;
      const int n = idx % BN;
      const int gRow = kBase + k;
      const int gCol = blockColBase + n;
      Bs[k * BN + n] = (gRow < K && gCol < N) ? B[(size_t)gRow * N + gCol] : 0.0f;
    }

    __syncthreads();

    // Compute: this thread updates TMxTN
#pragma unroll
    for (int k = 0; k < BK; ++k) {
      float aReg[TM];
#pragma unroll
      for (int i = 0; i < TM; ++i) {
        const int r = threadRowBase + i;
        const int mLocal = r - blockRowBase; // 0..BM-1
        aReg[i] = (r < M && mLocal >= 0 && mLocal < BM) ? As[mLocal * BK + k] : 0.0f;
      }

      float bReg[TN];
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        const int c = threadColBase + j;
        const int nLocal = c - blockColBase; // 0..BN-1
        bReg[j] = (c < N && nLocal >= 0 && nLocal < BN) ? Bs[k * BN + nLocal] : 0.0f;
      }

#pragma unroll
      for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
          acc[i][j] += aReg[i] * bReg[j];
        }
      }
    }

    __syncthreads();
  }

  // Write back C = alpha*(A@B) + beta*C
#pragma unroll
  for (int i = 0; i < TM; ++i) {
    const int r = threadRowBase + i;
    if (r < M) {
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        const int c = threadColBase + j;
        if (c < N) {
          const size_t idx = (size_t)r * N + c;
          C[idx] = alpha * acc[i][j] + beta * C[idx];
        }
      }
    }
  }
}
