#include <cstdio>
#include <cstdlib>
#include <cmath>

#define A(i, j) a[(i) * n + (j)]
#define B(i, j) b[(i) * n + (j)]
//生成随机矩阵
void random_matrix(int m, int n, float *a) {
    for (int i = 0; i <m; ++i) 
        for (int j = 0; j < n; ++j) 
#if 1
            A(i,j) = 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
#else
            A(i,j) = (j - i) % 3;
#endif
}

//cpu端矩阵乘法
void cpu_sgemm(float* A_ptr, float* B_ptr, float* C_ptr, const int M, const int N, const int K) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A_ptr[m * K + k] * B_ptr[k * N + n];
            }
            C_ptr[m * N + n] = sum;
        }
    }
}

//比较矩阵
float compare_matrices(int m, int n, float* a, float* b) {
    int i, j;
    float max_diff = 0.0f, diff;
    int printed = 0;

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            diff = abs(A(i, j) - B(i, j));
            max_diff = (diff > max_diff ? diff : max_diff);
            if (0 == printed) {
                if (max_diff > 0.5f || max_diff < -0.5f) {
                    printf("\n error: i %d, j %d diff %f got %f expect %f", i, j, diff, A(i, j), B(i, j));
                    printed = 1;
                }
            }
        }
    }
    return max_diff;
}

//gpu端矩阵乘法
__global__ void cuda_sgemm(float *A_ptr, float *B_ptr, float *C_ptr, const int M, const int N, const int K) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float tmp = 0.0f;
        for (int i = 0; i < K; ++i) {
            tmp += A_ptr[row * K + i] * B_ptr[i * N + col];
        }
        C_ptr[row * N + col] = tmp;
    }
}


int main() {
    int m = 2048, n = 2048, k = 2048;

    //预设大小
    const size_t mem_size_A = m * k * sizeof(float);
    const size_t mem_size_B = k * n * sizeof(float);
    const size_t mem_size_C = m * n * sizeof(float);

    //cpu端内存分配
    float *matrix_A_host = (float *)malloc(mem_size_A);
    float *matrix_B_host = (float *)malloc(mem_size_B);
    float *matrix_C_host_gpu_calc = (float *)malloc(mem_size_C);
    float *matrix_C_host_cpu_calc = (float *)malloc(mem_size_C);

    //初始化矩阵
    random_matrix(m, k, matrix_A_host);
    random_matrix(k, n, matrix_B_host);
    memset(matrix_C_host_gpu_calc, 0, mem_size_C);
    memset(matrix_C_host_cpu_calc, 0, mem_size_C);

    //gpu端内存分配
    float* matrix_A_device, *matrix_B_device, *matrix_C_device;
    cudaMalloc((void**)&matrix_A_device, mem_size_A);
    cudaMalloc((void**)&matrix_B_device, mem_size_B);
    cudaMalloc((void**)&matrix_C_device, mem_size_C);

    //将数据从cpu端复制到gpu端
    cudaMemcpy(matrix_A_device, matrix_A_host, mem_size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(matrix_B_device, matrix_B_host, mem_size_B, cudaMemcpyHostToDevice);
    //cpu端计算
    cpu_sgemm(matrix_A_host, matrix_B_host, matrix_C_host_cpu_calc, m, n, k);

    //调用核函数
    constexpr int BLOCK = 32;
    dim3 block(BLOCK, BLOCK);
    dim3 grid((m + BLOCK - 1) / BLOCK, (n + BLOCK - 1) / BLOCK);
    cuda_sgemm<<<grid, block>>>(matrix_A_device, matrix_B_device, matrix_C_device, m, n, k);

    cudaMemcpy(matrix_C_host_gpu_calc, matrix_C_device, mem_size_C, cudaMemcpyDeviceToHost);

    //比较结果
    float diff = compare_matrices(m, n, matrix_C_host_gpu_calc, matrix_C_host_cpu_calc);
    if (diff > 0.5 || diff < -0.5) {
        printf(" diff too big !\n");
    } else {
        printf("right !\n");
    }
    
    //释放内存
    free(matrix_A_host);
    free(matrix_B_host);
    free(matrix_C_host_gpu_calc);
    free(matrix_C_host_cpu_calc);
    cudaFree(matrix_A_device);
    cudaFree(matrix_B_device);
    cudaFree(matrix_C_device);

    return 0;
}