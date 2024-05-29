#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "utils.cu"

/* Atomic operations for floating point numbers 

    Definitions at the beginning of the file, if not the compiler explodes

*/

__device__ inline void atomicDivide(float *address, float val) {
    int *address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_i, assumed,
                        __float_as_int(__int_as_float(assumed) / val));
    } while (assumed != old);
}

__device__ inline void atomicSubstract(float *address, float val) {
    int *address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_i, assumed,
                        __float_as_int(__int_as_float(assumed) - val));
    } while (assumed != old);
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


__global__ void softmax_kernel(float* input, int n_tokens, int batch_size, int n_heads, int vecs_per_block, int next_token_2_half) {

    extern __shared__ float shared_data[];

    int tid = threadIdx.x + threadIdx.y * n_tokens; // current thread index
    int shared_mem_stride = threadIdx.y * n_tokens;
    int b = blockIdx.x; // current batch index
    int h = blockIdx.y; // current head index
    int row = blockIdx.z; // current row index
    int stride = n_tokens * n_tokens * n_heads;
    int size_block = n_tokens * vecs_per_block;

    float max_val;

    // Reduce to find global max within block
    // printf("%f from tid %d row %d batch %d and head %d sum %d\n", input[b * stride + h * n_tokens * n_tokens + row * n_tokens + tid], tid, row, b, h, b * stride + h * n_tokens * n_tokens + row * n_tokens + tid);
    shared_data[tid] = input[b * stride + h * n_tokens * n_tokens + row * size_block + tid];
    // printf("Value: %f\n", shared_data[tid]);
    __syncthreads();
    for (int i = next_token_2_half; i > 0; i >>= 1) {
        if (tid < i + threadIdx.y * n_tokens  && tid + i - shared_mem_stride < n_tokens) {
            // illegal memory access if not splitting the if u know
            if (shared_data[tid + i] > shared_data[tid]) shared_data[tid] = shared_data[tid + i];
            
        }
        __syncthreads();
    }
   
    max_val = shared_data[shared_mem_stride];

    // Step 2: Subtract max and exponentiate
    float sum_exp = 0.0;
    atomicSubstract(&input[b * stride + h * n_tokens * size_block + row * size_block + tid], max_val);

    atomicExp(&input[b * stride + h * n_tokens * size_block + row * size_block + tid]);

    // Reduce to find the sum of exps
    shared_data[tid] = input[b * stride + h * n_tokens * size_block + row * size_block + tid];
    __syncthreads();
    for (int i = next_token_2_half; i > 0; i >>= 1) {
        if (tid < i + shared_mem_stride && tid + i - shared_mem_stride < n_tokens) {
            shared_data[tid] += shared_data[tid + i];
        }
        __syncthreads();
    }

    // Broadcast the sum to all threads
    sum_exp = shared_data[shared_mem_stride];

    // Step 3: Divide by sum of exps
    atomicDivide(&input[b * stride + h * n_tokens * size_block + row * size_block + tid], sum_exp);

}

int next_power_of_2(int n) {
    if (n <= 1) return 1;
    if (n > 1024) return 1024;  // Adding this check to adhere to the max constraint
    int power = 1;
    while (power < n) {
        power <<= 1;
    }
    return power;
}

double softmax(float**** input, int num_inputs, int batch_size, int n_heads, float**** output) {
    float *d_input;
    float ** p_d_input = &d_input;

    printf("Loading matrix to GPU\n");

    loadMatrixToGPU_batched_sm(input, p_d_input, num_inputs, num_inputs, batch_size, n_heads);

    // print_from_GPU_sm(d_input, num_inputs, num_inputs, batch_size, n_heads);

    // int threads_per_block = 1024;
    // thing is, if num_inputs is low we are losing performance but if it is big...
    // this is better than the previous implementation. Also it is easier to understand

    int vecs_per_block = num_inputs;
    if (num_inputs * num_inputs > 1024) {
        vecs_per_block = 1024/num_inputs;
    }

    dim3 block_dims(num_inputs, vecs_per_block);
    dim3 grid_dims(batch_size, n_heads, (num_inputs-1)/vecs_per_block + 1);

    int tokens_power_2_half = next_power_of_2(num_inputs)/2;


    // time the kernel execution
    clock_t start, end;
    double cpu_time_used;
    start = clock();
    softmax_kernel<<<grid_dims, block_dims, vecs_per_block*num_inputs*sizeof(float)>>>(d_input, num_inputs, batch_size, n_heads, vecs_per_block, tokens_power_2_half);
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Time taken by GPU kernel: %f seconds\n", cpu_time_used);


    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n",
               cudaGetErrorString(cudaerr));

    // print from GPU
    // print_from_GPU_sm(d_input, num_inputs, num_inputs, batch_size, n_heads);

    // copy back to cpu
    float * output_array = (float *) malloc(batch_size * num_inputs * num_inputs * n_heads * sizeof(float));
    cudaMemcpy(output_array, d_input, batch_size * num_inputs * num_inputs * n_heads * sizeof(float), cudaMemcpyDeviceToHost);
    unflattenMatrix(output_array, num_inputs, num_inputs, batch_size, n_heads, output);

    return cpu_time_used;

}