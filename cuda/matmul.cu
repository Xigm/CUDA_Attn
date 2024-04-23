#include <cuda_runtime.h>

__global__ void matmulKernel(float* A, float* B, float* C, int m, int n, int p) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < p) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[k * p + col];
        }
        C[row * p + col] = sum;
    }
}

void matmul(float** A, float** B, float** C, int m, int n, int p) {
    // Allocate device memory
    float* d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, m * n * sizeof(float));
    cudaMalloc((void**)&d_B, n * p * sizeof(float));
    cudaMalloc((void**)&d_C, m * p * sizeof(float));

    // Copy input matrices from host to device
    cudaMemcpy(d_A, A[0], m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B[0], n * p * sizeof(float), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((p + blockDim.x - 1) / blockDim.x, (m + blockDim.y - 1) / blockDim.y);

    // Launch the CUDA kernel
    matmulKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, m, n, p);

    // Copy the result matrix from device to host
    cudaMemcpy(C[0], d_C, m * p * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}