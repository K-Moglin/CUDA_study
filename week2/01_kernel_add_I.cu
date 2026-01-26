#include <stdio.h>
#include "helper_cuda.h"
// goal: give two every large vectors A and B,
// calculate C where C[i] = A[i] + B[i]

__global__ void vec_add(float * A, float * B, float * C, int N) {
    // do computation
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N){
        C[i] = A[i] + B[i];
    }
}

int main {
    int size = 1024;

    // allocating memory for A on the host

    // this allocation is not accessible from the gpu
    float *h_A = (float*) malloc(size*sizeof(float));
    float *h_B = (float*) malloc(size*sizeof(float));
    float *h_C = (float*) malloc(size*sizeof(float));

    for (int i = 0; i < size; i++) {
        h_A[i] = i;
        h_B[i] = i * 2;
    }

    float *d_A, *d_B, *d_C;
    checkCudaerrors(cudaMalloc((void**)&d_A, size*sizeof(float)));
    checkCudaerrors(cudaMalloc((void**)&d_B, size*sizeof(float)));
    checkCudaerrors(cudaMalloc((void**)&d_C, size*sizeof(float)));

    checkCudaErrors(cudaMemcpy(d_A, h_A, size*sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, h_B, size*sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_C, h_C, size*sizeof(float), cudaMemcpyHostToDevice));

    int threadsPerBlock = 1;
    int blocksPerGrid = 1;

    vec_add<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, size);

    checkCudaErrors(cudaMemcpy(h_C, d_C, size*sizeof(float), cudaMemcpyDeviceToHost));

    return 0;
}