#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "MH_attention.cu"

#define MAX_DATA 1000 // Define the maximum number of data points

void attention(float*** input, int num_inputs, int dk, float** output, float** Wq, float** Wk, float** Wv, float** W_cproj);
__global__ void matmul_kernel(float* A, float* B, float* C, int m, int n, int p);
__global__ void matmul_kernel_transposed(float* A, float* B, float* C, int m, int n, int p); 
__global__ void normalize(float* A, int m, int n, int dk);
__global__ void masked_fill(float* A, int m, int n);
void matmul(float* A, float* B, float* C, int m, int n, int p);

void readMatrix(FILE *file, float **matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            fscanf(file, "%f", &matrix[i][j]);
        }
    }
}

void readMatrix_batched(FILE *file, float ***matrix, int rows, int cols, int b) {
    for (int k = 0; k < b; k++) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                fscanf(file, "%f", &matrix[k][i][j]);
            }
        }
    }
}

int main(int argc, char *argv[]) {
    FILE *output_file;	// File pointers
    int n_inputs, dk, batch_size, n_heads;
    float ***inputs, **Wq, **Wk, **Wv, **W_cproj, ***outputs;
    float kernel_time;

    if (argc < 5) {
        fprintf(stderr, "Usage: %s <input file> <output file> <n_inputs> <dk>\n", argv[0]);
        return -1;
    }

    
    n_inputs = atoi(argv[2]);
    dk = atoi(argv[3]);
    batch_size = atoi(argv[4]);
    n_heads = atoi(argv[5]);

    printf("n_inputs: %d, dk: %d, batch_size: %d\n", n_inputs, dk, batch_size);

    // Dynamically allocate the matrices
    inputs = (float ***)malloc(batch_size * sizeof(float **));
    Wq = (float **)malloc(dk * sizeof(float *));
    Wk = (float **)malloc(dk * sizeof(float *));
    Wv = (float **)malloc(dk * sizeof(float *));
    W_cproj = (float **)malloc(dk * sizeof(float *));
    outputs = (float ***)malloc(batch_size * sizeof(float **));

    for (int i = 0; i < batch_size; i++) {
        inputs[i] = (float **)malloc(n_inputs * sizeof(float *));
        outputs[i] = (float **)malloc(n_inputs * sizeof(float *));
    }

    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < n_inputs; j++) {
            inputs[i][j] = (float *)malloc(dk * sizeof(float));
            outputs[i][j] = (float *)malloc(dk * sizeof(float));
        }
    }

    
    for (int i = 0; i < dk; i++) {
        Wq[i] = (float *)malloc(dk * sizeof(float));
        Wk[i] = (float *)malloc(dk * sizeof(float));
        Wv[i] = (float *)malloc(dk * sizeof(float));
        W_cproj[i] = (float *)malloc(dk * sizeof(float));
    }


    clock_t start = clock();
    kernel_time = attention(inputs, n_inputs, dk, batch_size, n_heads, outputs, Wq, Wk, Wv, W_cproj);
    clock_t end = clock();
    float seconds = (float)(end - start) / CLOCKS_PER_SEC;
    // printf("Time taken by C code: %f seconds\n", seconds);

    // write kernel time on output_file
    // Open the output file for writing
    output_file = fopen(argv[1], "wb");
    if (output_file == NULL) {
        perror("Error opening output file");
        return -1;
    }

    fprintf(output_file, "%f\n", kernel_time);

    // Free the dynamically allocated memory
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < n_inputs; j++) {
            free(inputs[i][j]);
            free(outputs[i][j]);
        }
    }
    for (int i = 0; i < dk; i++) {
        free(Wq[i]);
        free(Wk[i]);
        free(Wv[i]);
    }
    free(inputs);
    free(Wq);
    free(Wk);
    free(Wv);

    return 0;

}
