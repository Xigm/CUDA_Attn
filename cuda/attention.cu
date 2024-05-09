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
        for (int k = 0; k < n; k++) {
            sum += A[x * n + k] * B[k * p + y];
        }
        // printf("%d sum %f \n", x * p + y, sum);
        C[x * p + y] = sum;

    }
}

__global__ void debug_matmul_kernel(float* A, float* B, float* C, int m, int n, int p) {
    int y = blockIdx.y * blockDim.y + threadIdx.y;    
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x < m && y < p) {
        // printf("Enters kernel\n");
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            sum += A[x * n + k] * B[k * p + y];
        }
        printf("%d sum %f \n", x * p + y, sum);
        C[x * p + y] = sum;

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

__device__ inline void atomicMaxfloat(float *address, float val) {
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
        atomicMaxfloat(max_val, input[row * cols + i]);
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

__device__ inline void atomicExp(float *address) {
    int *address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_i, assumed,
                        __float_as_int(expf(__int_as_float(assumed))));
    } while (assumed != old);
}

__device__ inline void atomicDivide(float *address, float val) {
    int *address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_i, assumed,
                        __float_as_int(__int_as_float(assumed) / val));
    } while (assumed != old);
}

__global__ void kernel_exp(float * z, int num_inputs){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = row * num_inputs + col;
    atomicExp(&z[idx]);
}

__global__ void kernel_accumulate(float * z, float * sum, int num_inputs){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = row * num_inputs + col;
    int idx_sum = idx/num_inputs;

    atomicAdd(&sum[idx_sum], z[idx]);
}

__global__ void kernel_divide(float * z, float * sum, int num_inputs){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = row * num_inputs + col;
    int idx_sum = idx/num_inputs;

    atomicDivide(&z[idx], sum[idx_sum]);
}

void softmax_mig(float *input, int num_inputs, dim3 ks_exp_grid, dim3 ks_exp_block) {

    // Print input array
    // float *input_array = (float *)malloc(num_inputs * num_inputs * sizeof(float));
    // cudaMemcpy(input_array, input, num_inputs * num_inputs * sizeof(float), cudaMemcpyDeviceToHost);
    // for (int i = 0; i < num_inputs; i++) {
    //     for (int j = 0; j < num_inputs; j++) {
    //         printf("%f ", input_array[i * num_inputs + j]);
    //     }
    //     printf("\n");
    // }
    // free(input_array);
    
    // exponentiate all elements
    kernel_exp<<<ks_exp_grid, ks_exp_block>>>(input, num_inputs);

    // // Print input array
    // float *input_array2 = (float *)malloc(num_inputs * num_inputs * sizeof(float));
    // cudaMemcpy(input_array2, input, num_inputs * num_inputs * sizeof(float), cudaMemcpyDeviceToHost);
    // for (int i = 0; i < num_inputs; i++) {
    //     for (int j = 0; j < num_inputs; j++) {
    //         printf("%f ", input_array2[i * num_inputs + j]);
    //     }
    //     printf("\n");
    // }
    // free(input_array2);

    // sum all elements from a row
    float *sum;
    cudaMalloc(&sum, num_inputs * sizeof(float));
    kernel_accumulate<<<ks_exp_grid, ks_exp_block>>>(input, sum, num_inputs);

    // Copy sum vector from device to host
    // float *sum_array = (float *)malloc(num_inputs * sizeof(float));
    // cudaMemcpy(sum_array, sum, num_inputs * sizeof(float), cudaMemcpyDeviceToHost);
    // // Print sum vector
    // for (int i = 0; i < num_inputs; i++) {
    //     printf("%f ", sum_array[i]);
    // }
    // free(sum_array);

    // divide each element by the sum
    kernel_divide<<<ks_exp_grid, ks_exp_block>>>(input, sum, num_inputs);

    // printf("\n OUTPUT SOFTMAX \n");
    // // Print input array
    // float *input_array3 = (float *)malloc(num_inputs * num_inputs * sizeof(float));
    // cudaMemcpy(input_array3, input, num_inputs * num_inputs * sizeof(float), cudaMemcpyDeviceToHost);
    // for (int i = 0; i < num_inputs; i++) {
    //     for (int j = 0; j < num_inputs; j++) {
    //         printf("%f ", input_array3[i * num_inputs + j]);
    //     }
    //     printf("\n");
    // }
    // free(input_array3);

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
    // define grid and block dimensions
    dim3 blockDim_d_q(dk, num_inputs);
    int d1 = 32;
    int d2 = 32;
    if( num_inputs < d1) {
        d1 = num_inputs;
    }
    if (dk < d2) {
        d2 = dk;
    }

    dim3 blockDim_d_q_mod(d1, d2);

    int d1g = 1;
    int d2g = 1;
    if (d1 == 32) {
        d1g = ((int) num_inputs/(32 + 1)) + 1;
    }
    if (d2 == 32) {
        d2g = ((int) dk/(32 + 1)) + 1;
    }
    dim3 gridDim(d1g, d2g);

    // print blockdim and grid dim
    printf("blockDim: (%d, %d)\n", blockDim_d_q_mod.x, blockDim_d_q_mod.y);
    printf("gridDim: (%d, %d)\n", gridDim.x, gridDim.y);

    // launch the matrix multiplication kernel for Wq
    // print dimensions d_input, d_Wq, Q

    matmul_kernel<<<gridDim, blockDim_d_q_mod>>>(d_input, d_Wq, Q, num_inputs, dk, dk);
    matmul_kernel<<<gridDim, blockDim_d_q_mod>>>(d_input, d_Wk, K, num_inputs, dk, dk);
    matmul_kernel<<<gridDim, blockDim_d_q_mod>>>(d_input, d_Wv, V, num_inputs, dk, dk);
    CHECK_KERNELCALL();

    // Copy Q matrix from device to host
    // float *Q_array = (float *)malloc(num_inputs * dk * sizeof(float));
    // cudaMemcpy(Q_array, Q, num_inputs * dk * sizeof(float), cudaMemcpyDeviceToHost);
    // // Print Q matrix
    // for (int i = 0; i < num_inputs; i++) {
    //     for (int j = 0; j < dk; j++) {
    //         printf("%f ", Q_array[i * dk + j]);
    //     }
    //     printf("\n");
    // }
    // free(Q_array);

    // launch the matrix multiplication kernel Q * K^T
    // dim3 attn_grid_dim(num_inputs,num_inputs);
    int d = 32;
    if( num_inputs < d) {
        d = num_inputs;
    }

    dim3 attn_block_dim(d, d);

    int dg = 1;
    if (d == 32) {
        dg = ((int) num_inputs/(32 + 1)) + 1;
    }

    dim3 attn_gridDim(dg, dg);

    dim3 attn_dim(num_inputs, num_inputs);

    matmul_kernel_transposed<<<attn_gridDim, attn_block_dim>>>(Q, K, attn, num_inputs, dk, num_inputs);
    CHECK_KERNELCALL();

    // launch the kernel to perform normalization by sqrt(dk)
    normalize<<<attn_gridDim, attn_block_dim>>>(attn, num_inputs, num_inputs, dk);
    CHECK_KERNELCALL();

    // launch the kernel to fill the triangular upper matrix with -inf
    masked_fill<<<attn_gridDim, attn_block_dim>>>(attn, num_inputs, num_inputs);
    CHECK_KERNELCALL();

    // launch the kernel for softmaxing the attn matrix
    // softmax<<<num_inputs, num_inputs>>>(attn, num_inputs, num_inputs);
    softmax_mig(attn, num_inputs, attn_gridDim, attn_block_dim);
    CHECK_KERNELCALL();

    // Copy attn matrix from device to host
    // float *attn_array = (float *)malloc(num_inputs * num_inputs * sizeof(float));
    // cudaMemcpy(attn_array, attn, num_inputs * num_inputs * sizeof(float), cudaMemcpyDeviceToHost);
    // // Print attn matrix
    // for (int i = 0; i < num_inputs; i++) {
    //     for (int j = 0; j < num_inputs; j++) {
    //         printf("%f ", attn_array[i * num_inputs + j]);
    //     }
    //     printf("\n");
    // }
    // free(attn_array);

    // launch the matrix multiplication kernel for attn * V
    matmul_kernel<<<gridDim, blockDim_d_q_mod>>>(attn, V, d_output, num_inputs, num_inputs, dk);
    CHECK_KERNELCALL();

    // // print d_output here
    // float *d_output_array = (float *)malloc(num_inputs * dk * sizeof(float));
    // cudaMemcpy(d_output_array, d_output, num_inputs * dk * sizeof(float), cudaMemcpyDeviceToHost);
    // for (int i = 0; i < num_inputs; i++) {
    //     for (int j = 0; j < dk; j++) {
    //         printf("%f ", d_output_array[i * dk + j]);
    //     }
    //     printf("\n");
    // }
    // free(d_output_array);

    
    // launch the matrix multiplication kernel for W_cproj
    matmul_kernel<<<gridDim, blockDim_d_q_mod>>>(d_output, d_W_cproj, d_output_2, num_inputs, dk, dk);
    CHECK_KERNELCALL();

    // // print d_output
    // float *d_output_array = (float *)malloc(num_inputs * dk * sizeof(float));
    // cudaMemcpy(d_output_array, d_output_2, num_inputs * dk * sizeof(float), cudaMemcpyDeviceToHost);
    // for (int i = 0; i < num_inputs; i++) {
    //     for (int j = 0; j < dk; j++) {
    //         printf("%f ", d_output_array[i * dk + j]);
    //     }
    //     printf("\n");
    // }
    // free(d_output_array);

    // copy the result matrix from device to host
    float * output_array = (float *) malloc(num_inputs * dk * sizeof(float));
    cudaMemcpy(output_array, d_output_2, num_inputs * dk * sizeof(float), cudaMemcpyDeviceToHost);
    unflattenMatrix(output_array, num_inputs, dk, output);

    // printf("\n\n");
    // //print output_array
    // for (int i = 0; i < num_inputs; i++) {
    //     for (int j = 0; j < dk; j++) {
    //         printf("%f ", output[i][j]);
    //     }
    //     printf("\n");
    // }
    
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