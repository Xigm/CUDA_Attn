#include <stdio.h>

void matmul(int *matrix1, int *matrix2, int *result, int size) {
    int i, j, k;

    for (i = 0; i < size; i++) {
        for (j = 0; j < size; j++) {
            *(result + i*size + j) = 0;
            for (k = 0; k < size; k++) {
                *(result + i*size + j) += *(matrix1 + i*size + k) * *(matrix2 + k*size + j);
            }
        }
    }
}

