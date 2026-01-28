#pragma once
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemm2DBlocktiling(int M, int N, int K, float alpha,
                                  const float* A, const float* B, float beta,
                                  float* C) {
  // (x=tile-col, y=tile-row) — same “flip x/y” idea as kernel4 for better B locality
  const int cRow = (int)blockIdx.y;
  const int cCol = (int)blockIdx.x;

  // thread coordinates inside the threadblock (2D)
  const int tCol = (int)threadIdx.x; // 0..(BN/TN - 1)
  const int tRow = (int)threadIdx.y; // 0..(BM/TM - 1)

  // this thread computes a TMxTN output tile
  const int rowBase = cRow * BM + tRow * TM;
  const int colBase = cCol * BN + tCol * TN;

  __shared__ float As[BM * BK]; // [BM, BK]
  __shared__ float Bs[BK * BN]; // [BK, BN]

  float acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i) {
#pragma unroll
    for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;
  }

  const int numTiles = CEIL_DIV(K, BK);

  const int threadsPerBlock = (BM / TM) * (BN / TN);
  const int tid = tRow * (BN / TN) + tCol; // linear thread id

  // Base pointers for this C tile
  const float* Abase = A + (size_t)cRow * BM * (size_t)K;
  const float* Bbase = B + (size_t)cCol * BN;
  float* Cbase = C + (size_t)cRow * BM * (size_t)N + (size_t)cCol * BN;

  for (int tile = 0; tile < numTiles; ++tile) {
    const int kBase = tile * BK;

    // ----------------------------
    // Load A tile: [BM, BK] into As
    // Each thread loads multiple elements by striding over linear index.
    // ----------------------------
    for (int idx = tid; idx < BM * BK; idx += threadsPerBlock) {
      const int aRow = idx / BK;   // 0..BM-1
      const int aCol = idx % BK;   // 0..BK-1
      const int gRow = cRow * BM + aRow;
      const int gCol = kBase + aCol;

      As[idx] = (gRow < M && gCol < K) ? Abase[(size_t)aRow * (size_t)K + (size_t)gCol] : 0.0f;
    }

    // ----------------------------
    // Load B tile: [BK, BN] into Bs
    // ----------------------------
    for (int idx = tid; idx < BK * BN; idx += threadsPerBlock) {
      const int bRow = idx / BN;   // 0..BK-1
      const int bCol = idx % BN;   // 0..BN-1
      const int gRow = kBase + bRow;
      const int gCol = cCol * BN + bCol;

      Bs[idx] = (gRow < K && gCol < N) ? Bbase[(size_t)gRow * (size_t)N + (size_t)bCol] : 0.0f;
    }

    __syncthreads();

    // ----------------------------
    // Compute on the tiles from SMEM
    // ----------------------------
#pragma unroll
    for (int kk = 0; kk < BK; ++kk) {
      float aFrag[TM];
      float bFrag[TN];

#pragma unroll
      for (int i = 0; i < TM; ++i) {
        const int r = tRow * TM + i;             // row within BM
        aFrag[i] = As[r * BK + kk];
      }

#pragma unroll
      for (int j = 0; j < TN; ++j) {
        const int c = tCol * TN + j;             // col within BN
        bFrag[j] = Bs[kk * BN + c];
      }

#pragma unroll
      for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
          acc[i][j] += aFrag[i] * bFrag[j];
        }
      }
    }

    __syncthreads();
  }

  // ----------------------------
  // Write back TMxTN results
  // ----------------------------
#pragma unroll
  for (int i = 0; i < TM; ++i) {
    const int gRow = rowBase + i;
    if (gRow < M) {
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        const int gCol = colBase + j;
        if (gCol < N) {
          float* cptr = Cbase + (size_t)(tRow * TM + i) * (size_t)N + (size_t)(tCol * TN + j);
          *cptr = alpha * acc[i][j] + beta * (*cptr);
        }
      }
    }
  }
}
