#include <stdio.h>
#include <math.h>

void layer_norm(float ** matrix, int rows, int cols) {

    // Calculate mean for each column
    for (int j = 0; j < cols; j++) {
        float sum = 0.0;
        for (int i = 0; i < rows; i++) {
            sum += matrix[i][j];
        }
        float mean = sum / rows;

        // Calculate variance for each column
        float variance = 0.0;
        for (int i = 0; i < rows; i++) {
            variance += (matrix[i][j] - mean) * (matrix[i][j] - mean);
        }
        variance /= rows;

        // Calculate standard deviation for each column
        float std_dev = sqrt(variance);

        // Normalize each element in the column
        for (int i = 0; i < rows; i++) {
            matrix[i][j] = (matrix[i][j] - mean) / std_dev;
        }
    }
}
