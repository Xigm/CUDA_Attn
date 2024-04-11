#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Function to calculate attention
void attention(float** input, int num_inputs, int input_size, float** output, float** Wq, float** Wk, float** Wv) {

    // Perform attention operation
    for (int i = 0; i < num_inputs; i++) {
        for (int j = 0; j < input_size; j++) {
            float sum_q = 0.0;
            float sum_k = 0.0;
            float sum_v = 0.0;
            for (int k = 0; k < num_inputs; k++) {
                sum_q += input[k][j] * Wq[k][j];
                sum_k += input[k][j] * Wk[k][j];
                sum_v += input[k][j] * Wv[k][j];
            }
            output[i][j] = (sum_q * sum_k) / num_inputs;
        }
    }
}