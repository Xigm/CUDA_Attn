#include <stdio.h>
#include <math.h>

#define PI 3.14159265358979323846

// Function to compute GELU for a 2D matrix
void gelu(float **x, int rows, int cols) {
    printf("-> Activation::gelu\n");

    float a = sqrt(2.0 / PI);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float xCubed = x[i][j] * x[i][j] * x[i][j];
            x[i][j] = 0.5 * x[i][j] * (1 + tanh(a * (x[i][j] + 0.044715 * xCubed))); // A is often 0.044715 in GELU implementations
        }
    }

    printf("<- Activation::gelu\n");
}
