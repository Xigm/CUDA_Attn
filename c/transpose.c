#include <stdio.h>

void transpose(float** input, float** output, int m, int n) {

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            output[j][i] = input[i][j];
        }
    }
}
