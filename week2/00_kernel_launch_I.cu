#include <stdio.h>
#inlcude "helper_cuda.h"

// __global__ attribute tells the nvidia compiler
// that this function can run the device(aka gpu)
// as well as the host

__global__ void myKernel(int x) {
    //threadIdx is a built-in object that gives you access to the thread number

    //what if we want to give each thread a unique id number?
    //blockDim is the number of threads inside a block
    int threadNum = blockIdx.x * blockDim.x + threadIdx.x;

    printf("inside the device: blockIdx=%d threadIdx=%d threadNum=%d %d\n", blockIdx.x, threadIdx.x, threadNum, x);
}

int main() {
    int threadsPerBlock = 1;
    int blocksPerGrid = 1;
    int x = 42;

    //mykernel is the name of a function that is run on the device
    //"parallel kernel" in the diagram that launches a grid on the gpu
    myKernel<<<blocksPerGrid, threadsPerBlock>>>(x);

    //throw an exception if cudaGetLastError returns an error
    //this error check only checks for errors on kernel launch
    checkCudaErrors(cudaGetLastError());

    //force the host to wait until the last kernel call finishes
    // also throws an exception if there is an error during kernel execution
    checkCudaErrors(cudaDeviceSynchronize());

    return 0;
}