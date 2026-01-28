#pragma once
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

// Kernel 6: Vectorize SMEM and GMEM Accesses
// Params follow the article's kernel-6 setting: BM=BN=128, BK=TM=TN=8.
// - As is stored transposed in SMEM: As[k][m] laid out as As[k*BM + m]
// - GMEM loads/stores use float4 where possible, with safe tail handling.
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemmVectorized(int M, int N, int K, float alpha,
                                const float* A, const float* B, float beta,
                                float* C) {
  // (x=tile-col, y=tile-row) like the repo/article for better B locality
  const int cRow = (int)blockIdx.y;
  const int cCol = (int)blockIdx.x;

  // 2D threadblock: (BN/TN, BM/TM)
  const int tCol = (int)threadIdx.x;  // 0..(BN/TN-1)
  const int tRow = (int)threadIdx.y;  // 0..(BM/TM-1)

  const int rowBase = cRow * BM + tRow * TM;
  const int colBase = cCol * BN + tCol * TN;

  // Shared tiles:
  // As is transposed: (BK x BM)
  __shared__ float As[BK * BM];
  // Bs normal: (BK x BN)
  __shared__ float Bs[BK * BN];

  // Per-thread accumulators
  float acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i) {
#pragma unroll
    for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;
  }

  // Linear thread id inside block
  const int numThreads = (BM / TM) * (BN / TN);
  const int tid = tRow * (BN / TN) + tCol;

  // Base pointers for this C tile
  const float* Abase = A + (size_t)cRow * BM * (size_t)K;
  const float* Bbase = B + (size_t)cCol * BN;
  float* Cbase = C + (size_t)cRow * BM * (size_t)N + (size_t)cCol * BN;

  const int numTiles = CEIL_DIV(K, BK);

  // Mapping threads to vectorized loads:
  static_assert((BK % 4) == 0, "BK must be divisible by 4 for float4 loads of A");
  static_assert((BN % 4) == 0, "BN must be divisible by 4 for float4 stores/loads of B");
  const int aVecPerRow = BK / 4;   // float4 segments across BK
  const int bVecPerRow = BN / 4;   // float4 segments across BN

  // A: total float4 loads per tile = BM * (BK/4)
  const int innerRowA = tid / aVecPerRow;   // 0..BM-1
  const int innerColA = tid % aVecPerRow;   // 0..(BK/4-1)

  // B: total float4 loads per tile = BK * (BN/4)
  const int innerRowB = tid / bVecPerRow;   // 0..BK-1
  const int innerColB = tid % bVecPerRow;   // 0..(BN/4-1)

  // Register caches (same idea as kernel5, but we load them in vector chunks)
  float regM[TM];
  float regN[TN];

  for (int tile = 0; tile < numTiles; ++tile) {
    const int kBase = tile * BK;

    // ----------------------------
    // GMEM -> SMEM for A (vectorized float4), while transposing into As
    // Article pattern:
    // float4 tmp = reinterpret_cast<float4*>(&A[row*K + col*4])[0];
    // As[(col*4 + j)*BM + row] = tmp.(xyzw)
    // ----------------------------
    {
      const int gRowA = cRow * BM + innerRowA;
      const int gColA0 = kBase + innerColA * 4;

      float4 tmp = make_float4(0.f, 0.f, 0.f, 0.f);
      if (gRowA < M && (gColA0 + 3) < K) {
        tmp = reinterpret_cast<const float4*>(
                &A[(size_t)gRowA * (size_t)K + (size_t)gColA0]
              )[0];
      } else {
        // safe tail
        float v0=0.f, v1=0.f, v2=0.f, v3=0.f;
        if (gRowA < M && (gColA0 + 0) < K) v0 = A[(size_t)gRowA * (size_t)K + (size_t)(gColA0 + 0)];
        if (gRowA < M && (gColA0 + 1) < K) v1 = A[(size_t)gRowA * (size_t)K + (size_t)(gColA0 + 1)];
        if (gRowA < M && (gColA0 + 2) < K) v2 = A[(size_t)gRowA * (size_t)K + (size_t)(gColA0 + 2)];
        if (gRowA < M && (gColA0 + 3) < K) v3 = A[(size_t)gRowA * (size_t)K + (size_t)(gColA0 + 3)];
        tmp = make_float4(v0, v1, v2, v3);
      }

      // transpose during store to SMEM: As[k][m]
      const int k0 = innerColA * 4 + 0;
      const int k1 = innerColA * 4 + 1;
      const int k2 = innerColA * 4 + 2;
      const int k3 = innerColA * 4 + 3;
      if (k0 < BK) As[(size_t)k0 * BM + (size_t)innerRowA] = tmp.x;
      if (k1 < BK) As[(size_t)k1 * BM + (size_t)innerRowA] = tmp.y;
      if (k2 < BK) As[(size_t)k2 * BM + (size_t)innerRowA] = tmp.z;
      if (k3 < BK) As[(size_t)k3 * BM + (size_t)innerRowA] = tmp.w;
    }

    // ----------------------------
    // GMEM -> SMEM for B (vectorized float4) into Bs
    // Article pattern:
    // reinterpret_cast<float4*>(&Bs[row*BN + col*4])[0] =
    //   reinterpret_cast<float4*>(&B[row*N  + col*4])[0];
    // ----------------------------
    {
      const int gRowB = kBase + innerRowB;
      const int gColB0 = cCol * BN + innerColB * 4;

      float4 tmp = make_float4(0.f, 0.f, 0.f, 0.f);
      if (gRowB < K && (gColB0 + 3) < N) {
        tmp = reinterpret_cast<const float4*>(
                &B[(size_t)gRowB * (size_t)N + (size_t)gColB0]
              )[0];
      } else {
        float v0=0.f, v1=0.f, v2=0.f, v3=0.f;
        if (gRowB < K && (gColB0 + 0) < N) v0 = B[(size_t)gRowB * (size_t)N + (size_t)(gColB0 + 0)];
        if (gRowB < K && (gColB0 + 1) < N) v1 = B[(size_t)gRowB * (size_t)N + (size_t)(gColB0 + 1)];
        if (gRowB < K && (gColB0 + 2) < N) v2 = B[(size_t)gRowB * (size_t)N + (size_t)(gColB0 + 2)];
        if (gRowB < K && (gColB0 + 3) < N) v3 = B[(size_t)gRowB * (size_t)N + (size_t)(gColB0 + 3)];
        tmp = make_float4(v0, v1, v2, v3);
      }

      // store into SMEM
      const int smemIdx = innerRowB * BN + innerColB * 4;
      if ((smemIdx + 3) < (BK * BN)) {
        reinterpret_cast<float4*>(&Bs[smemIdx])[0] = tmp;
      } else {
        // extremely unlikely with valid params, but keep safe
        if ((smemIdx + 0) < BK * BN) Bs[smemIdx + 0] = tmp.x;
        if ((smemIdx + 1) < BK * BN) Bs[smemIdx + 1] = tmp.y;
        if ((smemIdx + 2) < BK * BN) Bs[smemIdx + 2] = tmp.z;
        if ((smemIdx + 3) < BK * BN) Bs[smemIdx + 3] = tmp.w;
      }
    }

    __syncthreads();

    // ----------------------------
    // Compute: dotIdx outer, load regM/regN, outer product
    // With As transposed, regM loads are contiguous in SMEM.
    // ----------------------------
#pragma unroll
    for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // regM: TM contiguous elements from As[dotIdx*BM + row]
      const int mBase = (dotIdx * BM) + (tRow * TM);

      // load TM=8 as two float4
      float4 m0 = reinterpret_cast<const float4*>(&As[mBase + 0])[0];
      float4 m1 = reinterpret_cast<const float4*>(&As[mBase + 4])[0];
      regM[0]=m0.x; regM[1]=m0.y; regM[2]=m0.z; regM[3]=m0.w;
      regM[4]=m1.x; regM[5]=m1.y; regM[6]=m1.z; regM[7]=m1.w;

      // regN: TN contiguous elements from Bs[dotIdx*BN + col]
      const int nBase = (dotIdx * BN) + (tCol * TN);
      float4 n0 = reinterpret_cast<const float4*>(&Bs[nBase + 0])[0];
      float4 n1 = reinterpret_cast<const float4*>(&Bs[nBase + 4])[0];
      regN[0]=n0.x; regN[1]=n0.y; regN[2]=n0.z; regN[3]=n0.w;
      regN[4]=n1.x; regN[5]=n1.y; regN[6]=n1.z; regN[7]=n1.w;

#pragma unroll
      for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
          acc[i][j] += regM[i] * regN[j];
        }
      }
    }

    __syncthreads();
  }

  // ----------------------------
  // Write back TMxTN
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
