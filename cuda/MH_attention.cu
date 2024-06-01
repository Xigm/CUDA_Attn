#include <cuda.h>
#include <cuda_runtime.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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

__device__ inline void atomicSet(float *address, float value) {
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

__global__ void matmul_kernel_batched(float* A, float* B, float* C, int m, int n, int p, int batch_size) {
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

__global__ void matmul_kernel_semibatched(float* A, float* B, float* C, int m, int n, int p, int batch_size) {
    int y = blockIdx.y * blockDim.y + threadIdx.y;    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
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
__global__ void matmul_kernel_transposed_batched(float* A, float* B, float* C, int m, int n, int p, int batch_size) {
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

__global__ void matmul_kernel_transposed_MH(float* A, float* B, float* C, int m, int n, int n_heads, int batch_size) {

    // inputs per block is the number of inputs per block, which is the number of rows of the matrix
    // inputs_per_block = 1024 / m

    int x = threadIdx.x;
    int y = threadIdx.y;
    int b = blockIdx.x;
    int h = blockIdx.y;

    if (x < m && y < m && b < batch_size && h < n_heads) {

        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            sum += A[b * m * n * n_heads + h * n + y * n * n_heads + k] * B[b * m * n * n_heads + h * n + x * n * n_heads + k];
        }

        C[b * m * m * n_heads + h * m * m + m * y + x] = sum;
        // printf("Sum %f, in position %d to the matrix value %f\n",sum, b * m * m * n_heads + h * m * m + m * y + x, C[b * m * m * n_heads + h * m * m + m * y + x]);
    }

}

__global__ void matmul_kernel_semibatched_MH(float* A, float* B, float* C, int m, int n, int p, int batch_size) {
    // TODO
}

__global__ void matmul_kernel_MH(float* A, float* B, float* C, int m, int n, int n_heads, int batch_size, int head_size) {
    
    // inputs per block is the number of inputs per block, which is the number of rows of the matrix
    // inputs_per_block = 1024 / m

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int b = blockIdx.z;

    int head = x / head_size;

    // int dk = n_heads * head_size;
    if (x < n_heads*head_size && y < m && b < batch_size) {

        float sum = 0.0f;
        for (int k = 0; k < m ; k++) {
            

            float a = A[b * m * m * n_heads + y * m + k + head * m * m];
            float bb =  B[b * m * n + x + k * n_heads * head_size];
            sum += a * bb;


            // if (x == 0 && y == 1 && b == 0) {
            //     printf("A %f, B %f \n", a, bb);
            // }
        }

        C[b * m * n + n * y + x] = sum;
        // printf("Sum %f, in position %d the matrix value %f\n",sum, b * m * n + n * y + x, C[b * m * n + n * y + x]);
    }
}


// kernel which performs divide by sqrt(dk)
__global__ void normalize(float* A, int m, int n, int dk) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (row < m && col < n ) {
        A[z * m * n + row * n + col] /= sqrt((float) dk);
    }

}

__global__ void normalize_atomic(float* A, int m, int n, float sqrt_dk, int batch_size, int n_heads) {
    int y = threadIdx.y;
    int x = threadIdx.x;
    int b = blockIdx.x;
    int h = blockIdx.y;

    atomicDivide(&A[b * m * n * n_heads + h * m * n + y * n + x], sqrt_dk);
    
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
    int y = threadIdx.y;
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
        atomicSubstract(&z[idx], val);
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
    print_from_GPU(input, num_inputs, num_inputs, batch_size);

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

double attention(float*** input, int num_inputs, int dk, int batch_size, int n_heads, float*** output, float** Wq, float** Wk, float** Wv, float **W_cproj) {    
    
    clock_t start, end;
    double gpu_time_used;

    // allocate device memory for inputs, outputs, weights, and intermediate results attn
    float* d_input, *d_output, *d_output_2, *d_Wq, *d_Wk, *d_Wv, *d_W_cproj, *attn, *Q, *K, *V;
    float ** p_d_input = &d_input;
    float ** p_d_Wq = &d_Wq;
    float ** p_d_Wk = &d_Wk;
    float ** p_d_Wv = &d_Wv;
    float ** p_d_W_cproj = &d_W_cproj;

    int head_size = dk/n_heads;

    loadMatrixToGPU_batched(input, p_d_input, num_inputs, dk, batch_size);
    loadMatrixToGPU(Wq, p_d_Wq, dk, dk);
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

    // if (err6 != cudaSuccess) {
    //     fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err6));
    //     exit(EXIT_FAILURE);
    // }
    // define grid and block dimensions

    start = clock();


    dim3 blockDim_d_q(dk, num_inputs, batch_size);
    int d1 = 32;
    int d2 = 32;
    int d3 = 32;
    if( num_inputs < d1) {
        d1 = num_inputs;
    }
    if (dk < d2) {
        d2 = dk;
    }
    if (batch_size < d3) {
        d3 = 1;
    }

    dim3 blockDim_d_q_mod(d1, d2, 1);

    int d1g = 1;
    int d2g = 1;
    int d3g = 1;
    if (d1 == 32) {
        d1g = ((int) num_inputs/(32 + 1)) + 1;
    }
    if (d2 == 32) {
        d2g = ((int) dk/(32 + 1)) + 1;
    }
    if (d3 == 32) {
        // d3g = ((int) batch_size/(32 + 1)) + 1;
        d3g = batch_size;
    }

    d3g = batch_size;
    dim3 gridDim(d1g, d2g, d3g);

    // print blockdim and grid dim
    // printf("blockDim for matmuls: (%d, %d, %d)\n", blockDim_d_q_mod.x, blockDim_d_q_mod.y, blockDim_d_q_mod.z);
    // printf("gridDim for matmuls: (%d, %d, %d)\n", gridDim.x, gridDim.y, gridDim.z);

    // launch the matrix multiplication kernel for Wq

    // print using print_from_GPU(float* deviceMatrix, int m, int n, int b)
    // print_from_GPU(d_input, num_inputs, dk, batch_size);

    matmul_kernel_semibatched<<<gridDim, blockDim_d_q_mod>>>(d_input, d_Wq, Q, num_inputs, dk, dk, batch_size);
    matmul_kernel_semibatched<<<gridDim, blockDim_d_q_mod>>>(d_input, d_Wk, K, num_inputs, dk, dk, batch_size);
    matmul_kernel_semibatched<<<gridDim, blockDim_d_q_mod>>>(d_input, d_Wv, V, num_inputs, dk, dk, batch_size);
    CHECK_KERNELCALL();

    // print Q
    // print_from_GPU(Q, num_inputs, dk, batch_size);


    // launch the matrix multiplication kernel Q * K^T
    // dim3 attn_grid_dim(num_inputs,num_inputs);
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

    dim3 attn_dim(num_inputs, num_inputs, d3g);

    printf("blockDim for attn: (%d, %d, %d)\n", attn_block_dim.x, attn_block_dim.y, attn_block_dim.z);
    printf("gridDim for attn: (%d, %d, %d)\n", attn_gridDim.x, attn_gridDim.y, attn_gridDim.z);

    dim3 matmul_kernel_blockDim(num_inputs, num_inputs,1);
    dim3 matmul_kernel_gridDim(batch_size,n_heads,1);
    matmul_kernel_transposed_MH<<<matmul_kernel_gridDim, matmul_kernel_blockDim>>>(Q, K, attn, num_inputs, head_size, n_heads, batch_size);
    CHECK_KERNELCALL();

    // print attn matrix
    // print_from_GPU_sm(attn, num_inputs, num_inputs, batch_size, n_heads);

    // launch the kernel to perform normalization by sqrt(dk)
    float sqrt_dk = sqrt((float) dk);
    normalize_atomic<<<matmul_kernel_gridDim, matmul_kernel_blockDim>>>(attn, num_inputs, num_inputs, sqrt_dk, batch_size, n_heads);
    CHECK_KERNELCALL();

    // print attn matrix
    // print_from_GPU_sm(attn, num_inputs, num_inputs, batch_size, n_heads);

    // launch the kernel to fill the triangular upper matrix with -inf
    // fix the grid and block dimensions

    // int sq_n_tokens = num_inputs * num_inputs;
    // int heads_per_block = 1024/(sq_n_tokens+1) + 1;
    // int head_splited_to_n_blocks = 1;

    // if (sq_n_tokens > 1024) {
    //     head_splited_to_n_blocks = (sq_n_tokens-1)/1024 + 1;
    // }

    // dim3 masked_fill_blockDim(sq_n_tokens*heads_per_block, sq_n_tokens*heads_per_block, 1);
    // dim3 masked_fill_gridDim(batch_size, n_heads/heads_per_block, head_splited_to_n_blocks);

    dim3 masked_fill_blockDim(num_inputs, num_inputs, 1);
    dim3 masked_fill_gridDim(batch_size, n_heads, 1);

    masked_fill_atomic<<<masked_fill_gridDim, masked_fill_blockDim>>>(attn, num_inputs, num_inputs, batch_size, n_heads);
    CHECK_KERNELCALL();

    // print attn matrix
    // print_from_GPU_sm(attn, num_inputs, num_inputs, batch_size, n_heads);

    int vecs_per_block = num_inputs;
    if (num_inputs * num_inputs > 1024) {
        vecs_per_block = 1024/num_inputs;
    }

    dim3 block_dims(num_inputs, vecs_per_block);
    dim3 grid_dims(batch_size, n_heads, (num_inputs-1)/vecs_per_block + 1);

    int tokens_power_2_half = next_power_of_2(num_inputs)/2;

    // launch the kernel for softmaxing the attn matrix
    // softmax<<<num_inputs, num_inputs>>>(attn, num_inputs, num_inputs);
    // softmax_mig(attn, num_inputs, batch_size, attn_gridDim, attn_block_dim);
    softmax_kernel<<<grid_dims, block_dims, vecs_per_block*num_inputs*sizeof(float)>>>(attn, num_inputs, batch_size, n_heads, vecs_per_block, tokens_power_2_half);
    CHECK_KERNELCALL();

    // print attn matrix
    // print_from_GPU_sm(attn, num_inputs, num_inputs, batch_size, n_heads);

    dim3 kernel_MH_blocDim(dk, num_inputs, 1);
    dim3 kernel_MH_gridDim(1, 1, batch_size);
    // launch the matrix multiplication kernel for attn * V
    matmul_kernel_MH<<<kernel_MH_gridDim, kernel_MH_blocDim>>>(attn, V, d_output, num_inputs, dk, n_heads, batch_size, head_size);
    CHECK_KERNELCALL();
    
    // print d_output here
    // print_from_GPU(d_output, num_inputs, dk, batch_size);

    // launch the matrix multiplication kernel for W_cproj
    matmul_kernel_semibatched<<<gridDim, blockDim_d_q_mod>>>(d_output, d_W_cproj, d_output_2, num_inputs, dk, dk, batch_size);
    CHECK_KERNELCALL();

    // print d_output
    // print_from_GPU(d_output_2, num_inputs, dk, batch_size);


    end = clock();
    gpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

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

    return gpu_time_used;
}