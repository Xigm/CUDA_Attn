#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuda.h>

#include "utils.cu"

__global__ void matmul_kernel(float* A, float* B, float* C, int m, int n, int p) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    //print blockDim
    printf("BlockDim: %d, %d\n", blockDim.x, blockDim.y);

    printf("Row, Col: %d, %d\n", row, col);
    printf("C index: %d\n", row * p + col);
    
    if (row < m && col < p) {
        // printf("Enters kernel\n");
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            printf("Entering A: %d, B: %d\n", row * n + k, k * p + col);
            if (row * n + k == 4) {
                printf("Enter problem \n");
                printf("A: %f, B: %f\n", A[row * n + k], B[k * p + col]);
            }
            sum += A[row * n + k] * B[k * p + col];
        }
        printf("Sum: %f\n", sum);   
        C[row * p + col] = sum;
    }
}

// kernel matmul for transposed B matrix
__global__ void matmul_kernel_transposed(float* A, float* B, float* C, int m, int n, int p) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < p) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[col * n + k];
        }
        C[row * p + col] = sum;
    }
}

// kernel which performs divide by sqrt(dk)
__global__ void normalize(float* A, int m, int n, int dk) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        A[row * n + col] /= sqrt((float) dk);
    }
}

// kernel to fill the upper triangular matrix with -inf
__global__ void masked_fill(float* A, int m, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        if (row > col) {
            A[row * n + col] = -INFINITY;
        }
    }
}


void matmul(float* A, float* B, float* C, int m, int n, int p) {
    // Allocate device memory
    float* d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, m * n * sizeof(float));
    cudaMalloc((void**)&d_B, n * p * sizeof(float));
    cudaMalloc((void**)&d_C, m * p * sizeof(float));

    // Copy input matrices from host to device
    cudaMemcpy(d_A, A, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, n * p * sizeof(float), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockDim(2, 8);
    dim3 gridDim((p + blockDim.x - 1) / blockDim.x, (m + blockDim.y - 1) / blockDim.y);

    // Launch the matrix multiplication kernel
    matmul_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, m, n, p);

    // Copy result matrix from device to host
    cudaMemcpy(C, d_C, m * p * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void attention(float** input, int num_inputs, int dk, float** output, float** Wq, float** Wk, float** Wv, float **W_cproj) {
    // allocate device memory for inputs, outputs, weights, and intermediate results attn
    float* d_input, *d_output, *d_Wq, *d_Wk, *d_Wv, *d_W_cproj, *attn, *Q, *K, *V;

    loadMatrixToGPU(input, d_input, num_inputs, dk);
    loadMatrixToGPU(Wq, d_Wq, dk, dk);
    loadMatrixToGPU(Wk, d_Wk, dk, dk);
    loadMatrixToGPU(Wv, d_Wv, dk, dk);
    loadMatrixToGPU(W_cproj, d_W_cproj, dk, dk);
    
    // cudaMalloc(&d_input, num_inputs * dk * sizeof(float));
    // cudaError_t err1 = cudaMalloc(&d_Wq, dk * dk * sizeof(float));
    // cudaError_t err2 = cudaMalloc(&d_Wk, dk * dk * sizeof(float));
    // cudaError_t err3 = cudaMalloc(&d_output, num_inputs * dk * sizeof(float));
    // cudaError_t err4 = cudaMalloc(&d_Wv, dk * dk * sizeof(float));
    // cudaError_t err5 = cudaMalloc(&d_W_cproj, dk * dk * sizeof(float));
    cudaError_t err6 = cudaMalloc(&attn, num_inputs * num_inputs * sizeof(float));
    cudaError_t err7 = cudaMalloc(&Q, num_inputs * dk * sizeof(float));
    cudaError_t err8 = cudaMalloc(&K, num_inputs * dk * sizeof(float));
    cudaError_t err9 = cudaMalloc(&V, num_inputs * dk * sizeof(float));

    // copy input matrices from host to device
    // cudaMemcpy(d_Wq, Wq[0], dk * dk * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_Wk, Wk[0], dk * dk * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_Wv, Wv[0], dk * dk * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_W_cproj, W_cproj[0], dk * dk * sizeof(float), cudaMemcpyHostToDevice);

    if (err6 != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err6));
        exit(EXIT_FAILURE);
    }

    // define grid and block dimensions
    dim3 blockDim_d_q(dk, num_inputs);
    dim3 blockDim(16,16);
    dim3 gridDim((dk + blockDim.x - 1) / blockDim.x, (num_inputs + blockDim.y - 1) / blockDim.y);

    // print blockdim and grid dim
    printf("blockDim: (%d, %d)\n", blockDim_d_q.x, blockDim_d_q.y);
    // printf("gridDim: (%d, %d)\n", gridDim.x, gridDim.y);

    // launch the matrix multiplication kernel for Wq
    matmul_kernel<<<1, blockDim_d_q>>>(d_input, d_Wq, Q, num_inputs, dk, dk);
    CHECK_KERNELCALL();

    // matmul_kernel<<<gridDim, blockDim>>>(d_input, d_Wk, K, num_inputs, dk, dk);
    // CHECK_KERNELCALL();
    // matmul_kernel<<<gridDim, blockDim>>>(d_input, d_Wv, V, num_inputs, dk, dk);
    // CHECK_KERNELCALL();

    CHECK(cudaDeviceSynchronize());

    // launch the matrix multiplication kernel Q * K^T
    matmul_kernel_transposed<<<gridDim, blockDim>>>(Q, K, attn, num_inputs, dk, dk);
    CHECK_KERNELCALL();

    // launch the kernel to perform normalization by sqrt(dk)
    normalize<<<gridDim, blockDim>>>(attn, num_inputs, num_inputs, dk);
    CHECK_KERNELCALL();

    // launch the kernel to fill the triangular upper matrix with -inf
    masked_fill<<<gridDim, blockDim>>>(attn, num_inputs, num_inputs);
    CHECK_KERNELCALL();

    // launch the matrix multiplication kernel attn * V
    matmul_kernel<<<gridDim, blockDim>>>(attn, V, d_output, num_inputs, num_inputs, dk);
    CHECK_KERNELCALL();

    // launch the matrix multiplication kernel for W_cproj
    matmul_kernel<<<gridDim, blockDim>>>(d_output, d_W_cproj, d_output, num_inputs, dk, dk);
    CHECK_KERNELCALL();

    // copy the result matrix from device to host
    cudaMemcpy(output[0], d_output, num_inputs * dk * sizeof(float), cudaMemcpyDeviceToHost);
    

    // free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_Wq);
    cudaFree(d_Wk);
    cudaFree(d_Wv);
    cudaFree(d_W_cproj);
    cudaFree(attn);
    cudaFree(Q);
    cudaFree(K);
    cudaFree(V);

}