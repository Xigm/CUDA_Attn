#include <stdio.h>

// Attention function
void attention(float** input, int num_inputs, int dk, float** output, float** Wq, float** Wk, float** Wv, float **W_cproj);

// MLP function
void mlp(int m, int n, float ** A, float ** B, float ** C);

// Layer norm
void layer_norm(float ** matrix, int rows, int cols);

// Transformer block function
void transformerBlock(float** input, int num_inputs, int dk, float** output, float** Wq, float** Wk, float** Wv, float **W_cproj, float ** W_li, float ** W_lo) {

    // Perform attention
    attention(input, num_inputs, dk, output, Wq, Wk, Wv, W_cproj);

    // Layer norm
    // layer_norm(output, num_inputs, dk);

    // Perform MLP
    mlp(num_inputs, dk, output, W_li, W_lo);

}
