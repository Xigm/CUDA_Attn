#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "attention.cu"

#define MAX_DATA 1000 // Define the maximum number of data points

void attention(float** input, int num_inputs, int dk, float** output, float** Wq, float** Wk, float** Wv, float** W_cproj);
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

int main(int argc, char *argv[]) {
    FILE *inputFile, *outputFile;	// File pointers
    int n_inputs, dk;
    float **inputs, **Wq, **Wk, **Wv, **W_cproj, **outputs;

    if (argc < 5) {
        fprintf(stderr, "Usage: %s <input file> <output file> <n_inputs> <dk>\n", argv[0]);
        return -1;
    }

    inputFile = fopen(argv[1], "r");
    if (inputFile == NULL) {
        perror("Error opening input file");
        return -1;
    }

    n_inputs = atoi(argv[3]);
    dk = atoi(argv[4]);

    // Dynamically allocate the matrices
    inputs = (float **)malloc(n_inputs * sizeof(float *));
    Wq = (float **)malloc(dk * sizeof(float *));
    Wk = (float **)malloc(dk * sizeof(float *));
    Wv = (float **)malloc(dk * sizeof(float *));
    W_cproj = (float **)malloc(dk * sizeof(float *));
    outputs = (float **)malloc(n_inputs * sizeof(float *));

    for (int i = 0; i < n_inputs; i++) {
        inputs[i] = (float *)malloc(dk * sizeof(float));
        outputs[i] = (float *)malloc(dk * sizeof(float));
    }
    
    for (int i = 0; i < dk; i++) {
        Wq[i] = (float *)malloc(dk * sizeof(float));
        Wk[i] = (float *)malloc(dk * sizeof(float));
        Wv[i] = (float *)malloc(dk * sizeof(float));
        W_cproj[i] = (float *)malloc(dk * sizeof(float));
    }

    // Load the matrices
    readMatrix(inputFile, inputs, n_inputs, dk);
    readMatrix(inputFile, Wq, dk, dk);
    readMatrix(inputFile, Wk, dk, dk);
    readMatrix(inputFile, Wv, dk, dk);
    readMatrix(inputFile, W_cproj, dk, dk);

    fclose(inputFile);

    clock_t start = clock();
    attention(inputs, n_inputs, dk, outputs, Wq, Wk, Wv, W_cproj);
    clock_t end = clock();
    float seconds = (float)(end - start) / CLOCKS_PER_SEC;
    printf("Time taken by C code: %f seconds\n", seconds);

    // Open the output file for writing
    outputFile = fopen(argv[2], "wb");
    if (outputFile == NULL) {
        perror("Error opening output file");
        fclose(inputFile);
        return -1;
    }

    // Process and write the content of inputs in the output file
    int count = 0;
    printf("%d %d\n", n_inputs, dk);
    for (int i = 0; i < n_inputs; i++) {
        for (int j = 0; j < dk; j++) {
            fprintf(outputFile, "%f ", outputs[i][j]);
            count++;
            fprintf(outputFile, "\n");
        }
        
    }

    // append the time taken to the output file
    fprintf(outputFile, "%f\n", seconds);

    // Close the file
    fclose(outputFile);

    // Free the dynamically allocated memory
    for (int i = 0; i < n_inputs; i++) {
        free(inputs[i]);
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
