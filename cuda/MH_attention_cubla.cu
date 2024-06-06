#include <cuda.h>
#include <cuda_runtime.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cublas_v2.h"

#include "utils.cu"
#include "softmax.cu"
#define CUDART_INF_F            __int_as_float(0x7f800000)


/* Atomic operations for floating point numbers 

    Definitions at the beginning of the file, if not the compiler explodes

*/

// __device__ inline void atomicSet(float *address, float value) {
//     int *address_as_i = (int*) address;
//     int old = *address_as_i, assumed;
//     do {
//         assumed = old;
//         old = atomicCAS(address_as_i, assumed,
//                         value);
//     } while (assumed != old);
// }

__device__ inline void atomicSet(float *address, const float value) {
    unsigned int *address_as_ui = (unsigned int*)address;
    unsigned int value_as_ui = __float_as_uint(value);
    atomicExch(address_as_ui, value_as_ui);
}


/* Kernel functions 

    matmul_kernel_batched: kernel for matrix multiplication with batch dimension
    matmul_kernel_semibatched: kernel for matrix multiplication with batch dimension and B matrix without batch dimension
    matmul_kernel_semibatched_debug: kernel for matrix multiplication with batch dimension and B matrix without batch dimension, with debug prints
    matmul_kernel_transposed_batched: kernel for matrix multiplication with transposed B matrix and batch dimension
    normalize: kernel to perform divide by sqrt(dk)
    normalize_atomic: kernel to perform divide by sqrt(dk) with atomic operations
    masked_fill: kernel to fill the upper triangular matrix with -inf
    softmax: kernel to perform softmax
    kernel_exp: kernel to perform exponentiation
    kernel_accumulate: kernel to accumulate the sum of exponentials
    kernel_divide: kernel to divide each element by the sum of exponentials
    
*/

__global__ void matmul_kernel_batched(const float* A, const float* B, float* C,const int m,const int n,const int p,const int batch_size) {
    int y = blockIdx.y * blockDim.y + threadIdx.y;    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < m && y < p && z < batch_size) {
        // printf("Enters kernel\n");
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            // I simply add z * m * n to the index to access the correct batch for A,
            // while B is z * n * p	
            sum += A[z * m * n + x * n + k] * B[z * n * p + k * p + y];
        }
        // here I add z * m * p to the index to access the correct batch for C
        C[z * m * p + x * p + y] = sum;

    }
}

__global__ void matmul_kernel_semibatched(const float*__restrict__ A, const float*__restrict__ B, float*__restrict__ C, int m, int n, int p, int batch_size) {
    int x = blockIdx.y * blockDim.y + threadIdx.y;    
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < m && y < p && z < batch_size) {
        // printf("Enters kernel\n");
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            // I simply add z * m * n to the index to access the correct batch for A,
            // while B does not have batch dim
            sum += A[z * m * n + x * n + k] * B[k * p + y];
        }
        // printf("%d sum %f \n", x * p + y, sum);
        C[z * m * p + x * p + y] = sum;

    }
}

__global__ void matmul_kernel_triple_semibatched(float* input, float* Wq, float* Wk, float* Wv, float* Q, float* K, float* V, int m, int n, int p, int batch_size){
    int x = blockIdx.y * blockDim.y + threadIdx.y;    
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    int b = z % 3;
    int flag = z / 3;

    if (x < m && y < p && b < batch_size){

        float sum = 0.0f;

        if (flag == 0) {
            
            // float sum = 0.0f;
            for (int k = 0; k < n; k++) {
                // I simply add z * m * n to the index to access the correct batch for A,
                // while B does not have batch dim
                sum += input[z * m * n + x * n + k] * Wq[k * p + y];
            }
            // printf("%d sum %f \n", x * p + y, sum);
            Q[z * m * p + x * p + y] = sum;
        }
        else if (flag == 1)
        {
                 
            for (int k = 0; k < n; k++) {
                // I simply add z * m * n to the index to access the correct batch for A,
                // while B does not have batch dim
                sum += input[z * m * n + x * n + k] * Wk[k * p + y];
            }
            // printf("%d sum %f \n", x * p + y, sum);
            K[z * m * p + x * p + y] = sum;

        }
        else if (flag == 2)
        {
            // printf("Enters kernel\n");
            
            for (int k = 0; k < n; k++) {
                // I simply add z * m * n to the index to access the correct batch for A,
                // while B does not have batch dim
                sum += input[z * m * n + x * n + k] * Wv[k * p + y];
            }
            // printf("%d sum %f \n", x * p + y, sum);
            V[z * m * p + x * p + y] = sum;
        }
    }        


}



// __global__ void matmul_kernel_semibatched_debug(float* A, float* B, float* C, int m, int n, int p, int batch_size) {
//     int y = blockIdx.y * blockDim.y + threadIdx.y;    
//     int x = blockIdx.x * blockDim.x + threadIdx.x;
//     int z = blockIdx.z * blockDim.z + threadIdx.z;

//     if (x < m && y < p && z < batch_size) {
//         // printf("Enters kernel\n");
//         float sum = 0.0f;
//         for (int k = 0; k < n; k++) {
//             // I simply add z * m * n to the index to access the correct batch for A,
//             // while B does not have batch dim
//             // printf("Batch %d, pos %d, A %f, B %f \n", z, x * p + y, A[z * m * n + x * n + k], B[k * p + y]);
//             // print content of A
//             // printf("Batch %d, pos %d, A %f \n", z, z * m * n + x * n + k, A[z * m * n + x * n + k]);
//             sum += A[z * m * n + x * n + k] * B[k * p + y];
//         }
//         // printf("Batch %d, pos %d, sum %f \n", z, x * p + y, sum);
//         C[z * m * p + x * p + y] = sum;

//     }
// }

// kernel matmul for transposed B matrix
__global__ void matmul_kernel_transposed_batched(const float* A, const float* B, float* C, int m, int n, int p, int batch_size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (row < m && col < p && z < batch_size) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            sum += A[z * m * n + row * n + k] * B[z * p * n + col * n + k];
        }
        C[z * m * p + row * p + col] = sum;
    }
}

__global__ void matmul_kernel_transposed_MH(const float*__restrict__ A,const float*__restrict__ B, float*__restrict__ C, int m, int n, int n_heads, int batch_size, int blocks_per_attn) {

    // inputs per block is the number of inputs per block, which is the number of rows of the matrix
    // inputs_per_block = 1024 / m

    int x = threadIdx.x;
    int y = threadIdx.y + blockIdx.z * blockDim.y;
    int b = blockIdx.x;
    int h = blockIdx.y;

    if (x < m && y < m && b < batch_size && h < n_heads) {

        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            sum += A[b * m * n * n_heads + h * n + y * n * n_heads + k] * B[b * m * n * n_heads + h * n + x * n * n_heads + k];
        }

        C[b * m * m * n_heads + h * m * m + m * y + x] = sum;
    }

}

__global__ void matmul_kernel_semibatched_MH(float* A, float* B, float* C, int m, int n, int p, int batch_size) {
    // TODO
}

__global__ void matmul_kernel_MH(const float* A, const float* B, float* C, int m, int n, int n_heads, int batch_size, int head_size) {
    
    // inputs per block is the number of inputs per block, which is the number of rows of the matrix
    // inputs_per_block = 1024 / m

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int b = blockIdx.z;

    // if (x == 0)    printf("x %d, y %d, b %d\n", x, y, b);

    int head = x / head_size;

    
    // m = n_tok
    // n = dk
    // int dk = n_heads * head_size;
    if (x < n && y < m) {

        float sum = 0.0f;
        for (int k = 0; k < m ; k++) {
            
            // A is Attn = [B, n_heads, T, T]
            // B is V = [B, T, dk]

            float a = A[b * m * m * n_heads + y * m + k + head * m * m];
            // float bb = B[b * m * n + x + k * n_heads * head_size];
            float bb = B[b * m * n + x + k * n_heads * head_size];
            sum += a * bb;
            
            // if (x == 1024 && k < 10) {
            //     printf("A %f, B %f , pos A %d pos B %d\n", a, bb, b * m * m * n_heads + y * m + k + head * m * m, b * m * n + x + k * n_heads * head_size);
            // }
        }

        // [ B , T, dk]

        C[b * m * n + n * y + x] = sum;
        // if (blockIdx.x == 1) {
        //     printf("value %f, pos %d\n", C[b * m * n + n * y + x], b * m * n + n * y + x);
            
        //     printf("sum %f \n", sum);
        // }
    }
}


// kernel which performs divide by sqrt(dk)
__global__ void normalize(float* A, int m, int n, const int dk) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (row < m && col < n ) {
        A[z * m * n + row * n + col] /= sqrt((float) dk);
    }

}

__global__ void normalize_atomic(float* A, int m, int n, const float sqrt_dk, int batch_size, int n_heads) {
    int x = threadIdx.x;
    int y = threadIdx.y + blockIdx.z * blockDim.y;
    int b = blockIdx.x;
    int h = blockIdx.y;

    if (y < n) {
    atomicDivide(&A[b * m * n * n_heads + h * m * n + y * m + x], sqrt_dk);
    }
}

// kernel to fill the upper triangular matrix with -inf
// __global__ void masked_fill(float* A, int m, int n, int batch_size, int head_size) {
//     int row = blockIdx.y * blockDim.y + threadIdx.y;
//     int col = blockIdx.x * blockDim.x + threadIdx.x;
//     int b = blockIdx.z;
//     int h = blockIdx.z;

//     if (row < m && col < n && b < batch_size && h < n_heads) {
//         if (row < col) {
//             A[z * m * n + row * n + col] = -CUDART_INF_F;
//         }
//     }
// }

// kernel to fill the upper triangular matrix with -inf
__global__ void masked_fill_atomic(float* A, int m, int n, int batch_size, int n_heads) {
    int x = threadIdx.x;
    int y = threadIdx.y + blockIdx.z * blockDim.y;
    int b =  blockIdx.x;
    int h = blockIdx.y;


    // if (x < m && y < n) {
        if (y < x) {
            atomicSet(&A[b * m * n * n_heads + h * m * n + y * n + x], -CUDART_INF_F);
        }
    // }
}



__global__ void kernel_exp(float * z, int num_inputs, int batch_size){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int zz = blockIdx.z * blockDim.z + threadIdx.z;
    int idx = zz * num_inputs * num_inputs + row * num_inputs + col;

    if (row < num_inputs && col < num_inputs && zz < batch_size) {
        atomicExp(&z[idx]);
    }
}

__global__ void kernel_accumulate(float * z, float * sum, int num_inputs, int batch_size){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int zz = blockIdx.z * blockDim.z + threadIdx.z;
    int idx = zz * num_inputs * num_inputs + row * num_inputs + col;
    int idx_sum = idx/num_inputs;

    if (row < num_inputs && col < num_inputs && zz < batch_size) {
        atomicAdd(&sum[idx_sum], z[idx]);
    }
}

__global__ void kernel_divide(float * z, float * sum, int num_inputs, int batch_size){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int zz = blockIdx.z * blockDim.z + threadIdx.z;
    int idx = zz * num_inputs * num_inputs + row * num_inputs + col;
    int idx_sum = idx/num_inputs;

    if (row < num_inputs && col < num_inputs && zz < batch_size) {
        atomicDivide(&z[idx], sum[idx_sum]);
    }
}

__global__ void kernel_substract(float * z, float val, int num_inputs, int batch_size) {

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int zz = blockIdx.z * blockDim.z + threadIdx.z;
    int idx = zz * num_inputs * num_inputs + row * num_inputs + col;

    if (row < num_inputs && col < num_inputs && zz < batch_size) {
        atomicAdd(&z[idx], -val);
    }

}

void softmax_mig(float *input, int num_inputs, int batch_size, dim3 ks_exp_grid, dim3 ks_exp_block) {

    // added for numerical stability, substract 100
    kernel_substract<<<ks_exp_grid, ks_exp_block>>>(input, (float) 1000, num_inputs, batch_size);

    // Print input array
    // print_from_GPU(input, num_inputs, num_inputs, batch_size);
    
    // write values of input into a file called ./data/degub_softmax.txt 
    FILE* file = fopen("./data/debug_softmax.txt", "w");
    if (file != NULL) {
        float* input_host = (float*)malloc(num_inputs * num_inputs * batch_size * sizeof(float));
        cudaMemcpy(input_host, input, num_inputs * num_inputs * batch_size * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < num_inputs * num_inputs * batch_size; i++) {
            fprintf(file, "%f\n", input_host[i]);
        }
        fclose(file);
        free(input_host);
    } else {
        printf("Failed to open file for writing.\n");
    }

    // exponentiate all elements
    kernel_exp<<<ks_exp_grid, ks_exp_block>>>(input, num_inputs, batch_size);



    // // Print input array
    // print_from_GPU(input, num_inputs, num_inputs, batch_size);

    // sum all elements from a row
    float *sum;
    cudaMalloc(&sum, batch_size * num_inputs * sizeof(float));
    kernel_accumulate<<<ks_exp_grid, ks_exp_block>>>(input, sum, num_inputs, batch_size);

    // Copy sum vector from device to host
    // print_from_GPU(sum, num_inputs, 1, batch_size);

    // divide each element by the sum
    kernel_divide<<<ks_exp_grid, ks_exp_block>>>(input, sum, num_inputs, batch_size);

    // printf("\n OUTPUT SOFTMAX \n");
    // print_from_GPU(input, num_inputs, num_inputs, batch_size);

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

float attention(float*** input, int num_inputs, int dk, int batch_size, int n_heads, float*** output, float** Wq, float** Wk, float** Wv, float **W_cproj) {    

    cudaEvent_t start_cuda, start_global, stop_cuda;
    CHECK(cudaEventCreate(&start_cuda));
    CHECK(cudaEventCreate(&start_global));
    CHECK(cudaEventCreate(&stop_cuda));

    // allocate device memory for inputs, outputs, weights, and intermediate results attn
    float* d_input, *d_output, *d_output_2, *d_Wq, *d_Wq_r, *d_Wk, *d_Wv, *d_W_cproj, *attn, *Q, *K, *V;
    float ** p_d_input = &d_input;
    float ** p_d_Wq = &d_Wq;
    float ** p_d_Wq_r = &d_Wq_r;
    float ** p_Q = &Q;
    float ** p_d_Wk = &d_Wk;
    float ** p_d_Wv = &d_Wv;
    float ** p_d_W_cproj = &d_W_cproj;

    int head_size = dk/n_heads;

    printf("WTF\n");

    loadMatrixToGPU_batched(input, p_d_input, num_inputs, dk, batch_size);
    loadMatrixToGPU(Wq, p_d_Wq, dk, dk);
    loadMatrixToGPU_b_repeated(Wq, p_d_Wq_r, dk, dk, batch_size);
    loadMatrixToGPU(Wk, p_d_Wk, dk, dk);
    loadMatrixToGPU(Wv, p_d_Wv, dk, dk);
    loadMatrixToGPU(W_cproj, p_d_W_cproj, dk, dk);
    
    cudaMalloc(&Q, batch_size * num_inputs * dk * sizeof(float));
    cudaMalloc(&K, batch_size * num_inputs * dk * sizeof(float));
    cudaMalloc(&V, batch_size * num_inputs * dk * sizeof(float));
    cudaMalloc(&d_output, batch_size * num_inputs * dk * sizeof(float));
    cudaMalloc(&d_output_2, batch_size * num_inputs * dk * sizeof(float));

    // CAREFUL: This is MHA, then size is batch x n_heads x num_inputs x num_inputs
    cudaMalloc(&attn, batch_size * n_heads * num_inputs * num_inputs * sizeof(float));

    CHECK(cudaEventRecord(start_global, 0));
    
    int total_block_n_inputs = 1;
    int dk_per_block = num_inputs;
    int total_blocks_dk = 1;
    int first_dim = dk;
    if (dk * num_inputs > 1024) {
         
        if (dk > 1024) {
            dk_per_block = 1;
            total_block_n_inputs = num_inputs;
            total_blocks_dk = (dk - 1)/1024 + 1;
            first_dim = 1024;
        }   
        else{
            dk_per_block = 1024/dk;
            total_block_n_inputs = (num_inputs - 1)/dk_per_block + 1;
        }
    }

    dim3 kernel_MH_bloqDim(first_dim, dk_per_block, 1);
    dim3 kernel_MH_gridDim(total_blocks_dk, total_block_n_inputs, batch_size);

    cublasHandle_t handle;
    // cublasCreate(&handle);

    float* A_array[batch_size];
    float* B_array[batch_size];
    float* C_array[batch_size];
    for (int p = 0; p < batch_size; ++p) {
        A_array[p] = d_input + dk * num_inputs;
        B_array[p] = d_Wq_r + dk * dk; // + p*strideB;
        C_array[p] = Q + + dk * num_inputs;
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;

    CHECK(cudaEventRecord(start_cuda, 0));
    matmul_kernel_semibatched<<<kernel_MH_gridDim, kernel_MH_bloqDim>>>(d_input, d_Wq, Q, num_inputs, dk, dk, batch_size);
    matmul_kernel_semibatched<<<kernel_MH_gridDim, kernel_MH_bloqDim>>>(d_input, d_Wk, K, num_inputs, dk, dk, batch_size);
    matmul_kernel_semibatched<<<kernel_MH_gridDim, kernel_MH_bloqDim>>>(d_input, d_Wv, V, num_inputs, dk, dk, batch_size);

    printf("Trying cublas");

    cublasSgemmBatched(     handle,
                            CUBLAS_OP_T,
                            CUBLAS_OP_T,
                            num_inputs, dk, dk,
                            &alpha,
                            A_array, dk,
                            B_array, dk,
                            &beta,
                            C_array, num_inputs,
                            batch_size);

    CHECK_KERNELCALL();

    CHECK(cudaEventRecord(stop_cuda, 0));
    CHECK(cudaEventSynchronize(stop_cuda));
    float milliseconds = 0;
    CHECK(cudaEventElapsedTime(&milliseconds, start_cuda, stop_cuda));
    printf("Kernel matmul x3 time: %f ms\n", milliseconds);

    int d = 32;
    if( num_inputs < d) {
        d = num_inputs;
    }

    dim3 attn_block_dim(d, d, 1);

    int dg = 1;
    if (d == 32) {
        dg = ((int) num_inputs/(32 + 1)) + 1;
    }

    dim3 attn_gridDim(dg, dg, batch_size);


    int blocks_per_attn = 1;
    if (num_inputs * num_inputs > 1024) {
        blocks_per_attn = (num_inputs - 1) / (1024/num_inputs) + 1;
    }

    // launch the kernel to perform Q * K^T
    dim3 matmul_kernel_blockDim(num_inputs, 1024/num_inputs, 1);
    dim3 matmul_kernel_gridDim(batch_size,n_heads, blocks_per_attn);

    CHECK(cudaEventRecord(start_cuda, 0));
    matmul_kernel_transposed_MH<<<matmul_kernel_gridDim, matmul_kernel_blockDim>>>(Q, K, attn, num_inputs, head_size, n_heads, batch_size, blocks_per_attn);
    CHECK_KERNELCALL();

    CHECK(cudaEventRecord(stop_cuda, 0));
    CHECK(cudaEventSynchronize(stop_cuda));
    milliseconds = 0;
    CHECK(cudaEventElapsedTime(&milliseconds, start_cuda, stop_cuda));
    printf("Kernel Attn (QxK^t) time: %f ms\n", milliseconds);


    float sqrt_dk = sqrt((float) dk);
    CHECK(cudaEventRecord(start_cuda, 0));
    normalize_atomic<<<matmul_kernel_gridDim, matmul_kernel_blockDim>>>(attn, num_inputs, num_inputs, sqrt_dk, batch_size, n_heads);
    CHECK_KERNELCALL();
    CHECK(cudaEventRecord(stop_cuda, 0));
    CHECK(cudaEventSynchronize(stop_cuda));
    milliseconds = 0;
    CHECK(cudaEventElapsedTime(&milliseconds, start_cuda, stop_cuda));
    printf("Kernel normalize time: %f ms\n", milliseconds);

    CHECK(cudaEventRecord(start_cuda, 0));
    masked_fill_atomic<<<matmul_kernel_gridDim, matmul_kernel_blockDim>>>(attn, num_inputs, num_inputs, batch_size, n_heads);
    CHECK_KERNELCALL();
    CHECK(cudaEventRecord(stop_cuda, 0));
    CHECK(cudaEventSynchronize(stop_cuda));
    milliseconds = 0;
    CHECK(cudaEventElapsedTime(&milliseconds, start_cuda, stop_cuda));
    printf("Kernel masked_fill time: %f ms\n", milliseconds);


    int vecs_per_block = num_inputs;
    if (num_inputs * num_inputs > 1024) {
        vecs_per_block = 1024/num_inputs;
    }

    dim3 block_dims(num_inputs, vecs_per_block);
    dim3 grid_dims(batch_size, n_heads, (num_inputs-1)/vecs_per_block + 1);

    int tokens_power_2_half = next_power_of_2(num_inputs)/2;
    
    CHECK(cudaEventRecord(start_cuda, 0));
    softmax_kernel_v2<<<grid_dims, block_dims, vecs_per_block*num_inputs*sizeof(float)>>>(attn, num_inputs, batch_size, n_heads, vecs_per_block, tokens_power_2_half);
    CHECK_KERNELCALL();
    CHECK(cudaEventRecord(stop_cuda, 0));
    CHECK(cudaEventSynchronize(stop_cuda));
    milliseconds = 0;
    CHECK(cudaEventElapsedTime(&milliseconds, start_cuda, stop_cuda));
    printf("Kernel softmax time: %f ms\n", milliseconds);
    
    CHECK(cudaEventRecord(start_cuda, 0));    
    matmul_kernel_MH<<<kernel_MH_gridDim, kernel_MH_bloqDim>>>(attn, V, d_output, num_inputs, dk, n_heads, batch_size, head_size);
    CHECK_KERNELCALL();
    CHECK(cudaEventRecord(stop_cuda, 0));
    CHECK(cudaEventSynchronize(stop_cuda));
    milliseconds = 0;
    CHECK(cudaEventElapsedTime(&milliseconds, start_cuda, stop_cuda));
    printf("Kernel Attn * V time: %f ms\n", milliseconds);

    
    CHECK(cudaEventRecord(start_cuda, 0));
    matmul_kernel_semibatched<<<kernel_MH_gridDim, kernel_MH_bloqDim>>>(d_output, d_W_cproj, d_output_2, num_inputs, dk, dk, batch_size);
    CHECK_KERNELCALL();
    CHECK(cudaEventRecord(stop_cuda, 0));
    CHECK(cudaEventSynchronize(stop_cuda));
    milliseconds = 0;
    CHECK(cudaEventElapsedTime(&milliseconds, start_cuda, stop_cuda));
    printf("Kernel Output Proj time: %f ms\n", milliseconds);

    // print d_output
    // print_from_GPU(d_output_2, num_inputs, dk, batch_size);

    CHECK(cudaEventRecord(stop_cuda, 0));
    CHECK(cudaEventSynchronize(stop_cuda));
    milliseconds = 0;
    CHECK(cudaEventElapsedTime(&milliseconds, start_global, stop_cuda));
    printf("Kernel Total time: %f ms\n", milliseconds);


    // copy the result matrix from device to host
    float * output_array = (float *) malloc(batch_size * num_inputs * dk * sizeof(float));
    cudaMemcpy(output_array, d_output_2, batch_size * num_inputs * dk * sizeof(float), cudaMemcpyDeviceToHost);
    unflattenMatrix(output_array, num_inputs, dk, batch_size, output);


    // printf("\n\n");
    // //print output_array
    // for (int i = 0; i < batch_size; i++) {
    //     printf("Batch %d\n", i);
    //     for (int j = 0; j < num_inputs; j++) {
    //         for (int k = 0; k < dk; k++) {
    //             printf("%f ", output[i][j][k]);
    //         }
    //         printf("\n");
    //     }
    // }

    
    write_from_GPU_to_file(d_output_2, dk, num_inputs, batch_size, "./data/debug_output.txt");
    
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

    return milliseconds;
}