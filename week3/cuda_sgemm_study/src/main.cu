#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) do {                                      \
    cudaError_t err = (call);                                      \
    if (err != cudaSuccess) {                                      \
        printf("CUDA error at %s:%d - %s\n",                       \
               __FILE__, __LINE__, cudaGetErrorString(err));       \
        exit(1);                                                   \
    }                                                              \
} while (0)

void random_matrix(int rows, int cols, float *a) {
    for (int i = 0; i < rows * cols; ++i) {
        a[i] = 2.0f * (float(rand()) / RAND_MAX) - 1.0f;
    }
}

void cpu_sgemm(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = sum;
        }
    }
}

float compare_matrices(int rows, int cols, const float* a, const float* b) {
    float max_diff = 0.0f;
    for (int i = 0; i < rows * cols; ++i) {
        float diff = fabsf(a[i] - b[i]);
        if (diff > max_diff) max_diff = diff;
    }
    return max_diff;
}

// baseline naive：1 thread -> 1 output
__global__ void sgemm_naive(const float *A, const float *B, float *C, int M, int N, int K) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// shared memory tiling：1 block -> 1 tile
template <int TILE>
__global__ void sgemm_tiled(const float *A, const float *B, float *C, int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int col = blockIdx.x * TILE + threadIdx.x;
    int row = blockIdx.y * TILE + threadIdx.y;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;

        if (row < M && a_col < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + a_col];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (b_row < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

float benchmark_naive(const float *dA, const float *dB, float *dC, int M, int N, int K) {
    constexpr int BLOCK = 16;
    dim3 block(BLOCK, BLOCK);
    dim3 grid((N + BLOCK - 1) / BLOCK, (M + BLOCK - 1) / BLOCK);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // 预热
    for (int i = 0; i < 5; ++i) {
        sgemm_naive<<<grid, block>>>(dA, dB, dC, M, N, K);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < 10; ++i) {
        sgemm_naive<<<grid, block>>>(dA, dB, dC, M, N, K);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    ms /= 10.0f;

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    return ms;
}

int main() {
    int M = 1024, N = 1024, K = 1024;

    size_t sizeA = size_t(M) * K * sizeof(float);
    size_t sizeB = size_t(K) * N * sizeof(float);
    size_t sizeC = size_t(M) * N * sizeof(float);

    float *hA = (float*)malloc(sizeA);
    float *hB = (float*)malloc(sizeB);
    float *hC_cpu = (float*)malloc(sizeC);
    float *hC_gpu = (float*)malloc(sizeC);

    random_matrix(M, K, hA);
    random_matrix(K, N, hB);
    memset(hC_cpu, 0, sizeC);
    memset(hC_gpu, 0, sizeC);

    float *dA, *dB, *dC;
    CHECK_CUDA(cudaMalloc(&dA, sizeA));
    CHECK_CUDA(cudaMalloc(&dB, sizeB));
    CHECK_CUDA(cudaMalloc(&dC, sizeC));

    CHECK_CUDA(cudaMemcpy(dA, hA, sizeA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB, sizeB, cudaMemcpyHostToDevice));

    cpu_sgemm(hA, hB, hC_cpu, M, N, K);

    float ms = benchmark_naive(dA, dB, dC, M, N, K);

    CHECK_CUDA(cudaMemcpy(hC_gpu, dC, sizeC, cudaMemcpyDeviceToHost));

    float max_diff = compare_matrices(M, N, hC_gpu, hC_cpu);
    double gflops = 2.0 * M * N * K / (ms * 1e6);

    printf("Kernel: naive\n");
    printf("Time: %.3f ms\n", ms);
    printf("GFLOPS: %.2f\n", gflops);
    printf("Max diff: %f\n", max_diff);

    free(hA);
    free(hB);
    free(hC_cpu);
    free(hC_gpu);
    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));
    return 0;
}