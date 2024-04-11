#include <stdio.h>
#include <float.h>

void casual_mask(float** matrix, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (j > i) {
                matrix[i][j] = -FLT_MAX;
            }
        }
    }
}
