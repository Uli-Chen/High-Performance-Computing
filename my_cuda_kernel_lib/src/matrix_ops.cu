#include <cuda_runtime.h>
#include <iostream>
#include "matrix_ops.h"

// CUDA kernel: C = A + B
__global__ void matrixAddKernel(const float* A, const float* B, float* C,
                                int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = row * cols + col;

    if (row < rows && col < cols) {
        C[idx] = A[idx] + B[idx];
    }
}

// Host wrapper
void matrixAdd(const float* A, const float* B, float* C,
               int rows, int cols) {
    int size = rows * cols * sizeof(float);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((cols + block.x - 1) / block.x,
              (rows + block.y - 1) / block.y);

    matrixAddKernel<<<grid, block>>>(d_A, d_B, d_C, rows, cols);

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
