#include <stdio.h>
#include "attention.cu"
#include <cuda_runtime.h>
#include <cuda.h>

// __global__ void matmul_kernel(float* A, float* B, float* C, int m, int n, int p) {
//     int row = blockIdx.y * blockDim.y + threadIdx.y;
//     int col = blockIdx.x * blockDim.x + threadIdx.x;

//     if (row < m && col < p) {
//         float sum = 0.0f;
//         for (int k = 0; k < n; k++) {
//             sum += A[row * n + k] * B[k * p + col];
//         }
//         C[row * p + col] = sum;
//     }
// }


int main() {
    // Define the input matrices
    float A[2][4] = {{1, 2, 1, 2}, {3, 4, 3, 4}};
    float B[4][4] = {{1, 2, 3, 4}, {1, 2, 3, 4},{1, 2, 3, 4},{1, 2, 3, 4}};

    float ** A2 = (float **)malloc(2 * sizeof(float *));
    float ** B2 = (float **)malloc(4 * sizeof(float *));
    for (int i = 0; i < 2; i++) {
        A2[i] = (float *)malloc(4 * sizeof(float));
    }
    for (int i = 0; i < 4; i++) {
        B2[i] = (float *)malloc(4 * sizeof(float));
    }

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 4; j++) {
            A2[i][j] = A[i][j];
        }
    }

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            B2[i][j] = B[i][j];
        }
    }


    // Define the output matrix
    float C[4][2] = {{-1, -1}, {-1, -1}, {-1, -1},{-1, -1}};
    float * C2 = (float *)malloc(2 * 2 * sizeof(float *));

    // Allocate device memory
    float* d_A, *d_B, *d_C;
    // cudaMalloc((void**)&d_A, 2 * 4 * sizeof(float));
    // cudaMalloc((void**)&d_B, 4 * 4 * sizeof(float));
    cudaMalloc((void**)&d_C, 4 * 2 * sizeof(float));

    // Copy input matrices from host to device
    // cudaMemcpy(d_A, A, 2 * 4 * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_B, B, 4 * 4 * sizeof(float), cudaMemcpyHostToDevice);

    loadMatrixToGPU(A2, d_A, 2, 4);
    loadMatrixToGPU(B2, d_B, 4, 4);

    float* flattenedMatrix = (float *)malloc(4 * 2 * sizeof(float));
    flattenMatrix(A2, 2,4, flattenedMatrix);
    size_t size = 8 * sizeof(float);
    // Allocate memory on the GPU
    cudaMalloc((void**)&d_A, size);
    // Copy the flattened matrix from host to device
    cudaMemcpy(d_A, flattenedMatrix, size, cudaMemcpyHostToDevice);

    flattenedMatrix = (float *)malloc(4 * 4 * sizeof(float));
    flattenMatrix(B2, 4,4, flattenedMatrix);
    size = 16 * sizeof(float);
    // Allocate memory on the GPU
    cudaMalloc((void**)&d_B, size);
    // Copy the flattened matrix from host to device
    cudaMemcpy(d_B, flattenedMatrix, size, cudaMemcpyHostToDevice);



    // Call the matmul function
    dim3 blockDim(4, 2);
    matmul_kernel<<<1,blockDim>>>(d_A, d_B, d_C, 2, 4, 4);
    CHECK_KERNELCALL();


    CHECK(cudaDeviceSynchronize());
    // Copy the result from device to host
    cudaMemcpy(C, d_C, 4 * 2 * sizeof(float), cudaMemcpyDeviceToHost);


    // Print the result
    printf("Result:\n");
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 2; j++) {
            printf("%.2f ", C[i][j]);
        }
        printf("\n");
    }

    // free the memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);


    return 0;
}