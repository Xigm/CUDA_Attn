#include <cuda.h>
#include <cuda_runtime.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "utils.cu"

__global__ void matmul_kernel(float* A, float* B, float* C, int m, int n, int p) {
    int y = blockIdx.y * blockDim.y + threadIdx.y;    
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x < m && y < p) {
        // printf("Enters kernel\n");
        float sum = 0.0f;
        for (int k = 0; k < p; k++) {
            sum += A[x * p + k] * B[k * n + y];
        } 
        // printf("%d \n",row * p + col);
        C[x * n + y] = sum;

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
        if (row < col) {
            A[row * n + col] = -INFINITY;
        }
    }
}

__device__ inline void atomicMaxFloat(float *address, float val) {
    int *address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_i, assumed,
                        __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
}

__global__ void softmax(float *input, int rows, int cols) {
    int row = blockIdx.x;  // Each block handles one row
    int tid = threadIdx.x;
    int idx = row * cols + tid;

    if (tid >= cols) return;  // Safeguard against excess threads

    // Step 1: Find max for numerical stability
    extern __shared__ float shared[];
    float *max_val = &shared[0];
    float *exp_sum = &shared[1];
    if (tid == 0) {
        *max_val = input[idx];
        *exp_sum = 0.0;
    }
    __syncthreads();

    for (int i = tid; i < cols; i += blockDim.x) {
        atomicMaxFloat(max_val, input[row * cols + i]);
    }
    __syncthreads();

    // Step 2: Compute sum of exponentials
    float sum_exp = 0.0;
    for (int i = tid; i < cols; i += blockDim.x) {
        sum_exp += expf(input[row * cols + i] - *max_val);
    }
    atomicAdd(exp_sum, sum_exp);
    __syncthreads();

    // Step 3: Calculate softmax output
    for (int i = tid; i < cols; i += blockDim.x) {
        input[row * cols + i] = expf(input[row * cols + i] - *max_val) / *exp_sum;
    }
}

// void matmul(float* A, float* B, float* C, int m, int n, int p) {
//     // Allocate device memory
//     float* d_A, *d_B, *d_C;
//     cudaMalloc((void**)&d_A, m * n * sizeof(float));
//     cudaMalloc((void**)&d_B, n * p * sizeof(float));
//     cudaMalloc((void**)&d_C, m * p * sizeof(float));

//     // Copy input matrices from host to device
//     cudaMemcpy(d_A, A, m * n * sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_B, B, n * p * sizeof(float), cudaMemcpyHostToDevice);

//     // Define grid and block dimensions
//     dim3 blockDim(2, 8);
//     dim3 gridDim((p + blockDim.x - 1) / blockDim.x, (m + blockDim.y - 1) / blockDim.y);

//     // Launch the matrix multiplication kernel
//     // matmul_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, m, n, p);

//     // Copy result matrix from device to host
//     cudaMemcpy(C, d_C, m * p * sizeof(float), cudaMemcpyDeviceToHost);

//     // Free device memory
//     cudaFree(d_A);
//     cudaFree(d_B);
//     cudaFree(d_C);
// }

void attention(float** input, int num_inputs, int dk, float** output, float** Wq, float** Wk, float** Wv, float **W_cproj) {
    // allocate device memory for inputs, outputs, weights, and intermediate results attn
    float* d_input, *d_output, *d_output_2, *d_Wq, *d_Wk, *d_Wv, *d_W_cproj, *attn, *Q, *K, *V;
    float ** p_d_input = &d_input;
    float ** p_d_Wq = &d_Wq;
    float ** p_d_Wk = &d_Wk;
    float ** p_d_Wv = &d_Wv;
    float ** p_d_W_cproj = &d_W_cproj;

    loadMatrixToGPU(input, p_d_input, num_inputs, dk);
    loadMatrixToGPU(Wq, p_d_Wq, dk, dk);
    loadMatrixToGPU(Wk, p_d_Wk, dk, dk);
    loadMatrixToGPU(Wv, p_d_Wv, dk, dk);
    loadMatrixToGPU(W_cproj, p_d_W_cproj, dk, dk);
    
    cudaMalloc(&attn, num_inputs * num_inputs * sizeof(float));
    cudaMalloc(&Q, num_inputs * dk * sizeof(float));
    cudaMalloc(&K, num_inputs * dk * sizeof(float));
    cudaMalloc(&V, num_inputs * dk * sizeof(float));
    cudaMalloc(&d_output, num_inputs * dk * sizeof(float));
    cudaMalloc(&d_output_2, num_inputs * dk * sizeof(float));

    // if (err6 != cudaSuccess) {
    //     fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err6));
    //     exit(EXIT_FAILURE);
    // }

    printf("Creating kernel of size %d, %d \n", dk, num_inputs);
    // define grid and block dimensions
    dim3 blockDim_d_q(dk, num_inputs);
    dim3 blockDim_d_q_mod(32, 32);
    dim3 gridDim((int) dk/(32-1) + 1.0f, (int) num_inputs/(32-1) + 1.0f);

    // print blockdim and grid dim
    printf("blockDim: (%d, %d)\n", blockDim_d_q_mod.x, blockDim_d_q_mod.y);
    printf("gridDim: (%d, %d)\n", gridDim.x, gridDim.y);

    // launch the matrix multiplication kernel for Wq
    // print dimensions d_input, d_Wq, Q

    matmul_kernel<<<gridDim, blockDim_d_q_mod>>>(d_input, d_Wq, Q, num_inputs, dk, dk);
    matmul_kernel<<<gridDim, blockDim_d_q_mod>>>(d_input, d_Wk, K, num_inputs, dk, dk);
    matmul_kernel<<<gridDim, blockDim_d_q_mod>>>(d_input, d_Wv, V, num_inputs, dk, dk);
    CHECK_KERNELCALL();

    // Copy the first 3x3 submatrix from Q to the host
    float* Q_submatrix = (float*)malloc(3 * 3 * sizeof(float));
    cudaMemcpy(Q_submatrix, Q, 3 * 3 * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the submatrix
    printf("First 3x3 submatrix from Q:\n");
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            printf("%f ", Q_submatrix[i * 3 + j]);
        }
        printf("\n");
    }

    // Free the host memory
    free(Q_submatrix);

    // launch the matrix multiplication kernel Q * K^T
    dim3 attn_dim(num_inputs,num_inputs);
    matmul_kernel_transposed<<<1, attn_dim>>>(Q, K, attn, num_inputs, dk, num_inputs);
    CHECK_KERNELCALL();

    // launch the kernel to perform normalization by sqrt(dk)
    normalize<<<1, attn_dim>>>(attn, num_inputs, num_inputs, dk);
    CHECK_KERNELCALL();

    // launch the kernel to fill the triangular upper matrix with -inf
    masked_fill<<<1, attn_dim>>>(attn, num_inputs, num_inputs);
    CHECK_KERNELCALL();

    // launch the kernel to fill the triangular upper matrix with -inf
    masked_fill<<<1, attn_dim>>>(attn, num_inputs, num_inputs);
    CHECK_KERNELCALL();

    // launch the kernel for softmaxing the attn matrix
    softmax<<<num_inputs, num_inputs>>>(attn, num_inputs, num_inputs);
    CHECK_KERNELCALL();

    // launch the matrix multiplication kernel attn * V
    matmul_kernel<<<gridDim, blockDim_d_q_mod>>>(attn, V, d_output, num_inputs, num_inputs, dk);
    CHECK_KERNELCALL();

    // launch the matrix multiplication kernel for W_cproj
    matmul_kernel<<<gridDim, blockDim_d_q_mod>>>(d_output, d_W_cproj, d_output_2, num_inputs, dk, dk);
    CHECK_KERNELCALL();

    // copy the result matrix from device to host
    float * output_array = (float *) malloc(num_inputs * dk * sizeof(float));
    cudaMemcpy(output_array, d_output_2, num_inputs * dk * sizeof(float), cudaMemcpyDeviceToHost);
    unflattenMatrix(output_array, num_inputs, dk, output);
    
    // free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_output_2);
    cudaFree(d_Wq);
    cudaFree(d_Wk);
    cudaFree(d_Wv);
    cudaFree(d_W_cproj);
    cudaFree(attn);
    cudaFree(Q);
    cudaFree(K);
    cudaFree(V);

}