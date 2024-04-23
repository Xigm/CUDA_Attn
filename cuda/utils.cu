#include <cuda_runtime.h>

void flattenMatrix(float** matrix, int m, int n, float* flattenedMatrix) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            flattenedMatrix[i * n + j] = matrix[i][j];
        }
    }
}

void loadMatrixToGPU(float** matrix, float* deviceMatrix, int n, int m) {
    // Flatten the matrix
    float* flattenedMatrix = (float *)malloc(m * n * sizeof(float));
    flattenMatrix(matrix, m, n, flattenedMatrix);

    size_t size = n * m * sizeof(float);

    // Allocate memory on the GPU
    cudaMalloc(&deviceMatrix, size);

    // Copy the flattened matrix from host to device
    cudaMemcpy(deviceMatrix, flattenedMatrix, size, cudaMemcpyHostToDevice);

}