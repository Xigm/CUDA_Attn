#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>

#define CHECK_KERNELCALL()                                                          \
  {                                                                                 \
    const cudaError_t err = cudaGetLastError();                                     \
    if (err != cudaSuccess) {                                                       \
      printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
      exit(EXIT_FAILURE);                                                           \
    }                                                                               \
  }

#define CHECK(call)                                                                 \
  {                                                                                 \
    const cudaError_t err = call;                                                   \
    if (err != cudaSuccess) {                                                       \
      printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
      exit(EXIT_FAILURE);                                                           \
    }                                                                               \
  }

void flattenMatrix(float** matrix, int m, int n, float* flattenedMatrix) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            flattenedMatrix[i * n + j] = matrix[i][j];  // Correct the index from i * m + j to i * n + j
            // printf("%f ", flattenedMatrix[i * n + j]);
        }
    }
}


void loadMatrixToGPU(float** matrix, float* deviceMatrix, int m, int n) {
    // Flatten the matrix
    float* flattenedMatrix = (float *)malloc(m * n * sizeof(float));
    flattenMatrix(matrix, m, n, flattenedMatrix);

    size_t size = n * m * sizeof(float);

    // Allocate memory on the GPU
    cudaError_t err =  cudaMalloc((void**)&deviceMatrix, size);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device matrix: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    // Copy the flattened matrix from host to device
    cudaMemcpy(deviceMatrix, flattenedMatrix, size, cudaMemcpyHostToDevice);


}