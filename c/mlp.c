#include <stdio.h>

// Function to perform matrix multiplication
void matmul(float **A, float **B, float **C, int m, int n, int p);
void gelu(float **x, int rows, int cols);

// Function to implement a multi-layer perceptron
void mlp(int m, int n, int p, float ** A, float ** B, float ** C) {
    // Perform matrix multiplication
    matmul(A, B, A, m, n, n*4);

    // Apply Gelu activation function
    gelu(A, m, p);

    // Perform matrix multiplication
    matmul(A, C, A, m, n*4, n);
}