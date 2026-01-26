#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>

static void cuda_check(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error %s: %s\n", msg, cudaGetErrorString(e));
        std::exit(1);
    }
}

// Row-major storage.
// BLAS-like semantics:
// op(A) is m x k
// op(B) is k x n
// If transA=false: A stored as (m x k)
// If transA=true : A stored as (k x m)  (so op(A)=A^T has shape m x k)
// If transB=false: B stored as (k x n)
// If transB=true : B stored as (n x k)  (so op(B)=B^T has shape k x n)

__device__ __forceinline__ float getA(const float* A, int m, int k, bool transA, int i, int t) {
    // want op(A)[i, t]
    // transA=false: A is m x k -> A[i*k + t]
    // transA=true : A is k x m -> op(A)=A^T, so op(A)[i,t] = A[t*m + i]
    return transA ? A[t * m + i] : A[i * k + t];
}

__device__ __forceinline__ float getB(const float* B, int k, int n, bool transB, int t, int j) {
    // want op(B)[t, j]
    // transB=false: B is k x n -> B[t*n + j]
    // transB=true : B is n x k -> op(B)=B^T, so op(B)[t,j] = B[j*k + t]
    return transB ? B[j * k + t] : B[t * n + j];
}

__global__ void gemm_kernel(
    int m, int n, int k,
    float alpha, const float* A, bool transA,
    const float* B, bool transB,
    float beta, float* C
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m || col >= n) return;

    float acc = 0.0f;
    for (int t = 0; t < k; ++t) {
        acc += getA(A, m, k, transA, row, t) * getB(B, k, n, transB, t, col);
    }

    float old = C[row * n + col];
    C[row * n + col] = alpha * acc + beta * old;
}

static void cpu_gemm_ref(
    int m, int n, int k,
    float alpha, const float* A, bool transA,
    const float* B, bool transB,
    float beta, float* C
) {
    auto getA_h = [&] (int i, int t) -> float {
        return transA ? A[t * m + i] : A[i * k + t];
    };
    auto getB_h = [&] (int t, int j) -> float {
        return transB ? B[j * k + t] : B[t * n + j];
    };

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float acc = 0.0f;
            for(int t = 0; t < k; ++t) acc += getA_h(i, t) * getB_h(t, j);
            C[i * n + j] = alpha * acc + beta * C[i * n + j];
        }
    }
}

static void fill_rand(std::vector<float>& v, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : v) {
        x = dist(rng);
    }
}

static bool nearly_equal(const std::vector<float>& a, const std::vector<float>& b, float eps) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); i++) {
        float diff = std::fabs(a[i] - b[i]);
        if (diff > eps) return false;
    }
    return true;
}

static void run_one_case(int m, int n, int k, bool transA, bool transB) {
    // Allocate host
    // A storage shape depends on transA:
    // transA=false => A(m,k), transA=true => A(k,m)
    int A_rows = transA ? k : m;
    int A_cols = transA ? m : k;
    
    int B_rows = transB ? n : k;
    int B_cols = transB ? k : n;

    std::vector<float> hA((size_t)A_rows * A_cols);
    std::vector<float> hB((size_t)B_rows * B_cols);
    std::vector<float> hC((size_t)m * n);
    std::vector<float> hC_ref;

    fill_rand(hA, 123u + m*3 + n*5 + k*7 + (int)transA*11);
    fill_rand(hB, 456u + m*13 + n*17 + k*19 + (int)transB*23);
    fill_rand(hC, 789u + m*29 + n*31 + k*37);

    hC_ref = hC; // copy for reference

    float alpha = 1.25f;
    float beta = 0.75f;

    //CPU reference
    cpu_gemm_ref(m, n, k, alpha, hA.data(), transA, hB.data(), transB, beta, hC_ref.data());

    // Device alloc
    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    cuda_check(cudaMalloc(&dA, hA.size() * sizeof(float)), "malloc A");
    cuda_check(cudaMalloc(&dB, hB.size() * sizeof(float)), "malloc B");
    cuda_check(cudaMalloc(&dC, hC.size() * sizeof(float)), "malloc C");

    cuda_check(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(float), cudaMemcpyHostToDevice), "copy A");
    cuda_check(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(float), cudaMemcpyHostToDevice), "copy B");
    cuda_check(cudaMemcpy(dC, hC.data(), hC.size() * sizeof(float), cudaMemcpyHostToDevice), "copy C");

    dim3 block(16, 16);
    dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);

    gemm_kernel<<<grid, block>>>(m, n, k, alpha, dA, transA, dB, transB, beta, dC);
    cuda_check(cudaGetLastError(), "kernel launch");
    cuda_check(cudaDeviceSynchronize(), "kernel sync");

    std::vector<float> hC_out(hC.size());
    cuda_check(cudaMemcpy(hC_out.data(), dC, hC.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy C back");

    cudaFree(dA); cudaFree(dB); cudaFree(dC);

    if (!nearly_equal(hC_out, hC_ref, 1e-3f)) {
        fprintf(stderr, "[FAIL] m=%d n=%d k=%d transA=%d transB=%d\n", m, n, k, (int)transA, (int)transB);
        std::exit(2);
    } else {
        printf("[PASS] m=%d n=%d k=%d transA=%d transB=%d\n", m, n, k, (int)transA, (int)transB);
    }
}

static void run_tests() {
    // A few corner-ish cases; keep k>=1 etc.
    struct Case { int m,n,k; };
    std::vector<Case> cases = {
        {1,1,1},
        {1,5,1},
        {2,3,1},
        {2,2,2},
        {7,7,7},
        {31,29,17},
        {64,64,64},
        {128,256,64},
    };

    for (auto c : cases) {
        for (int ta = 0; ta <= 1; ta++) {
            for (int tb = 0; tb <= 1; tb++) {
                run_one_case(c.m, c.n, c.k, ta, tb);
            }
        }
    }
}

static double now_sec() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

static void bench(int m, int n, int k, bool transA, bool transB, int iters) {
    int A_rows = transA ? k : m;
    int A_cols = transA ? m : k;
    int B_rows = transB ? n : k;
    int B_cols = transB ? k : n;

    std::vector<float> hA((size_t)A_rows * A_cols);
    std::vector<float> hB((size_t)B_rows * B_cols);
    std::vector<float> hC((size_t)m * n);

    fill_rand(hA, 1);
    fill_rand(hB, 2);
    fill_rand(hC, 3);

    float alpha = 1.0f;
    float beta = 0.0f;

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    cuda_check(cudaMalloc(&dA, hA.size() * sizeof(float)), "malloc A");
    cuda_check(cudaMalloc(&dB, hB.size() * sizeof(float)), "malloc B");
    cuda_check(cudaMalloc(&dC, hC.size() * sizeof(float)), "malloc C");
    cuda_check(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(float), cudaMemcpyHostToDevice), "copy A");
    cuda_check(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(float), cudaMemcpyHostToDevice), "copy B");
    cuda_check(cudaMemcpy(dC, hC.data(), hC.size() * sizeof(float), cudaMemcpyHostToDevice), "copy C");

    dim3 block(16, 16);
    dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);

    //warmup
    gemm_kernel<<<grid, block>>>(m, n, k,alpha, dA, transA, dB, transB, beta, dC);
    cuda_check(cudaDeviceSynchronize(), "warmup sync");

    double t0 = now_sec();
    for (int i = 0; i < iters; ++i) {
        gemm_kernel<<<grid, block>>>(m, n, k, alpha, dA, transA, dB, transB, beta, dC);
    }
    cuda_check(cudaDeviceSynchronize(), "bench sync");
    double t1 = now_sec();

    double avg_ms = (t1 - t0) * 1e3 / iters;
    printf("BENCH m=%d n=%d k=%d transA=%d transB=%d: %.3f ms\n",
        m, n, k, (int)transA, (int)transB, avg_ms);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
}

int main(int argc, char** argv) {
    if (argc >= 2 && std::strcmp(argv[1], "--test") == 0) {
        run_tests();
        return 0;
    }
    if (argc >= 2 && std::strcmp(argv[1], "--bench") == 0) {
        if (argc != 8) {
            fprintf(stderr, "Usage: %s --bench m n k transA transB iters\n", argv[0]);
            return 1;
        }
        int m = std::atoi(argv[2]);
        int n = std::atoi(argv[3]);
        int k = std::atoi(argv[4]);
        bool transA = std::atoi(argv[5]) != 0;
        bool transB = std::atoi(argv[6]) != 0;
        int iters = std::atoi(argv[7]);
        bench(m, n, k, transA, transB, iters);
        return 0;
    }

    fprintf(stderr, "Usage: \n");
    fprintf(stderr, " %s --test\n", argv[0]);
    fprintf(stderr, " %s --bench m n k transA transB iters\n", argv[0]);
    return 1;
}