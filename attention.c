#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void attention(float** input, int num_inputs, int dk, float** output, float** Wq, float** Wk, float** Wv);
void matmul(float** A, float** B, float** C, int m, int n, int p);
void transpose(float** input, float** output, int m, int n);
void softmax(float* input, int size);

// Function to calculate attention
void attention(float** input, int num_inputs, int dk, float** output, float** Wq, float** Wk, float** Wv) {

    // define query, key, value matrices
    float** Q = (float**)malloc(num_inputs * sizeof(float*));
    float** K = (float**)malloc(num_inputs * sizeof(float*));
    float** V = (float**)malloc(num_inputs * sizeof(float*));
    float** attn_scores = (float**)malloc(num_inputs * sizeof(float*));
    float** transposed_K = (float**)malloc(dk * sizeof(float*));

    for (int i = 0; i < num_inputs; i++) {
        Q[i] = (float*)malloc(dk * sizeof(float));
        K[i] = (float*)malloc(dk * sizeof(float));
        V[i] = (float*)malloc(dk * sizeof(float));
        attn_scores[i] = (float*)malloc(num_inputs * sizeof(float));
    }

    for (int i = 0; i < dk; i++) {
        transposed_K[i] = (float*)malloc(num_inputs * sizeof(float));
    }

    matmul(input, Wq, Q, num_inputs, dk, dk);
    matmul(input, Wk, K, num_inputs, dk, dk);
    matmul(input, Wv, V, num_inputs, dk, dk);

    // Perform attention operation
    // K is dims: num_inputs x dk
    transpose(K, transposed_K, num_inputs, dk);

    matmul(Q, transposed_K, attn_scores, num_inputs, dk, num_inputs);

    // Normalize the attention scores by internal dimension squared
    float dk_sqrt = sqrt(dk);
    for (int i = 0; i < num_inputs; i++) {
        for (int j = 0; j < num_inputs; j++) {
            attn_scores[i][j] = attn_scores[i][j] / dk_sqrt;
        }
    }

    // Softmax the attention scores for each 
    for (int i = 0; i < num_inputs; i++) {
        softmax(attn_scores[i], num_inputs);
    }

    // Multiply the attention scores by the value matrix
    matmul(attn_scores, V, output, num_inputs, num_inputs, dk);


}