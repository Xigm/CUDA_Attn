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

void unflattenMatrix(float* flattenedMatrix, int m, int n, int b, float*** matrix) {
  for (int k = 0; k < b; k++) {
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        matrix[k][i][j] = flattenedMatrix[m * n * k + i * n + j];
      }
    }
  }
}

void unflattenMatrix(float* flattenedMatrix, int m, int n, int b, int n_heads, float**** matrix) {
  for (int k = 0; k < b; k++) {
    for (int h = 0; h < n_heads; h++){
      for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
          matrix[k][h][i][j] = flattenedMatrix[m * n * n_heads * k + m * n * h + i * n + j];
        }
      }
    }
  }
}

void unflattenMatrix(float* flattenedMatrix, int m, int n, float** matrix) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      matrix[i][j] = flattenedMatrix[i * n + j];
    }
  }
}


void flattenMatrix(float*** matrix, int m, int n, int b, float* flattenedMatrix) {

    for (int k = 0; k < b; k++) {
      for (int i = 0; i < m; i++) {
          for (int j = 0; j < n; j++) {
              flattenedMatrix[m * n * k + i * n + j] = matrix[k][i][j];  // Correct the index from i * m + j to i * n + j
              // printf("%f ", flattenedMatrix[m * n * k + i * n + j]);
          }
      }
    }
}

void flattenMatrix_sm(float**** matrix, int m, int n, int b, int n_heads, float* flattenedMatrix) {

  for (int k = 0; k < b; k++) {
    for (int h = 0; h < n_heads; h++){
      for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            flattenedMatrix[m * n * n_heads * k + m * n * h + i * n + j] = matrix[k][h][i][j];  // Correct the index from i * m + j to i * n + j
            // printf("%f ", flattenedMatrix[m * n * n_heads * k + m * n * h + i * n + j]);
        }
      }
    }
  }
}

void flattenMatrix(float** matrix, int m, int n, float* flattenedMatrix) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            flattenedMatrix[i * n + j] = matrix[i][j];  // Correct the index from i * m + j to i * n + j
            // printf("%f ", flattenedMatrix[i * n + j]);
        }
    }
}

void flattenMatrix_repeated(float** matrix, int m, int n, int b, float* flattenedMatrix) {
  for (int k = 0; k < b; k++) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            flattenedMatrix[k * n * m + i * n + j] = matrix[i][j];  // Correct the index from i * m + j to i * n + j
            // printf("%f ", flattenedMatrix[i * n + j]);
        }
    }
  }
}


void loadMatrixToGPU(float** matrix, float** deviceMatrix, int m, int n) {
    // Flatten the matrix
    float* flattenedMatrix = (float *)malloc(m * n * sizeof(float));
    flattenMatrix(matrix, m, n, flattenedMatrix);

    size_t size = n * m * sizeof(float);

    // Allocate memory on the GPU
    cudaError_t err =  cudaMalloc((void**) deviceMatrix, size);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device matrix: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    // Copy the flattened matrix from host to device
    cudaMemcpy(*deviceMatrix, flattenedMatrix, size, cudaMemcpyHostToDevice);


}


void loadMatrixToGPU_b_repeated(float** matrix, float** deviceMatrix, int m, int n, int batch_size) {
    // Flatten the matrix
    float* flattenedMatrix = (float *)malloc(batch_size * m * n * sizeof(float));
    flattenMatrix_repeated(matrix, m, n, batch_size, flattenedMatrix);

    size_t size = batch_size * n * m * sizeof(float);

    // Allocate memory on the GPU
    cudaError_t err =  cudaMalloc((void**) deviceMatrix, size);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device matrix: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    // Copy the flattened matrix from host to device
    CHECK(cudaMemcpy(*deviceMatrix, flattenedMatrix, size, cudaMemcpyHostToDevice));;


}

void loadMatrixToGPU_batched(float*** matrix, float** deviceMatrix, int m, int n, int b) {
    // Flatten the matrix
    float* flattenedMatrix = (float *)malloc(m * n * b * sizeof(float));
    flattenMatrix(matrix, m, n, b, flattenedMatrix);

    size_t size = b * n * m * sizeof(float);

    // Allocate memory on the GPU
    cudaError_t err =  cudaMalloc((void**) deviceMatrix, size);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device matrix: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    // Copy the flattened matrix from host to device
    cudaMemcpy(*deviceMatrix, flattenedMatrix, size, cudaMemcpyHostToDevice);

}

void loadMatrixToGPU_batched_sm(float**** matrix, float** deviceMatrix, int m, int n, int b, int n_heads) {
    // Flatten the matrix
    float* flattenedMatrix = (float *)malloc(m * n * b * n_heads * sizeof(float));
    flattenMatrix_sm(matrix, m, n, b, n_heads, flattenedMatrix);

    size_t size = b * n_heads * n * m * sizeof(float);

    // Allocate memory on the GPU
    cudaError_t err =  cudaMalloc((void**) deviceMatrix, size);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device matrix: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    // Copy the flattened matrix from host to device
    cudaMemcpy(*deviceMatrix, flattenedMatrix, size, cudaMemcpyHostToDevice);
    
}

void print_from_GPU(float* deviceMatrix, int m, int n, int b) {

    float *d_array = (float *)malloc(m * n * b * sizeof(float));
    cudaMemcpy(d_array, deviceMatrix, m * n * b * sizeof(float), cudaMemcpyDeviceToHost);
    for (int k = 0; k < b; k++) {
      printf("Batch %d\n", k);
      for (int i = 0; i < m; i++) {
          for (int j = 0; j < n; j++) {
              printf("%f ", d_array[k * m * n + i * n + j]);
          }
          printf("\n");
      }
    }

    free(d_array);
}

void print_from_GPU_sm(float* deviceMatrix, int m, int n, int b, int n_heads) {

    float *d_array = (float *)malloc(m * n * b * n_heads * sizeof(float));
    cudaMemcpy(d_array, deviceMatrix, m * n * b * n_heads * sizeof(float), cudaMemcpyDeviceToHost);
    for (int k = 0; k < b; k++) {
      for (int h = 0; h < n_heads; h++){
        printf("Batch %d, Head %d\n", k, h);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                printf("%f ", d_array[k * m * n * n_heads + h * m * n + i * n + j]);
            }
            printf("\n");
        }
      }
     
    }

    free(d_array);
}

void write_from_GPU_to_file_sm(float* deviceMatrix, int m, int n, int b, int n_heads, const char* filename) {
    float *d_array = (float *)malloc(m * n * b * n_heads * sizeof(float));
    cudaMemcpy(d_array, deviceMatrix, m * n * b * n_heads * sizeof(float), cudaMemcpyDeviceToHost);

    FILE *fp = fopen(filename, "w");
    if (fp == NULL) {
        fprintf(stderr, "Failed to open the file for writing.\n");
        free(d_array);
        return;
    }

    for (int k = 0; k < b; k++) {
        for (int h = 0; h < n_heads; h++) {
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    fprintf(fp, "%f ", d_array[k * m * n * n_heads + h * m * n + i * n + j]);
                }
                fprintf(fp, "\n");
            }
        }
    }

    fclose(fp);
    free(d_array);
}


void write_from_GPU_to_file(float* deviceMatrix, int m, int n, int b, const char* filename) {
    float *d_array = (float *)malloc(m * n * b * sizeof(float));
    cudaMemcpy(d_array, deviceMatrix, m * n * b * sizeof(float), cudaMemcpyDeviceToHost);

    FILE *fp = fopen(filename, "w");
    if (fp == NULL) {
        fprintf(stderr, "Failed to open the file for writing.\n");
        free(d_array);
        return;
    }

    for (int k = 0; k < b; k++) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                fprintf(fp, "%f ", d_array[k * m * n + i * m + j]);
            }
            fprintf(fp, "\n");
        }
    }

    fclose(fp);
    free(d_array);
}