#include <math.h>

void softmax(float* input, int size) {

    float max_val = input[0];
    float sum = 0.0;

    // Find the maximum value in the input array
    for (int i = 1; i < size; i++) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }

    // Compute the exponential of each element and sum them
    for (int i = 0; i < size; i++) {
        input[i] = exp(input[i] - max_val);
        sum += input[i];
    }

    // Normalize the values by dividing each element by the sum
    for (int i = 0; i < size; i++) {
        input[i] /= sum;
    }
}