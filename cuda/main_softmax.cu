#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "softmax.cu"

void readMatrix(FILE *file, float **matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            fscanf(file, "%f", &matrix[i][j]);
        }
    }
}

void readMatrix_batched(FILE *file, float ***matrix, int rows, int cols, int b) {
    for (int k = 0; k < b; k++) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                fscanf(file, "%f", &matrix[k][i][j]);
            }
        }
    }
}

void readMatrix_batched_sm(FILE *file, float ****matrix, int rows, int cols, int b, int n_heads) {
    for (int k = 0; k < b; k++) {
        for (int h = 0; h < n_heads; h++){
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    fscanf(file, "%f", &matrix[k][h][i][j]);
                }
            }
        }
    }
}

int main(int argc, char *argv[]) {
    FILE *inputFile, *outputFile;	// File pointers
    int n_inputs, n_heads, batch_size;
    float ****inputs, ****outputs;

    if (argc < 5) {
        fprintf(stderr, "Usage: %s <input file> <output file> <n_inputs> <dk>\n", argv[0]);
        return -1;
    }

    inputFile = fopen(argv[1], "r");
    if (inputFile == NULL) {
        perror("Error opening input file");
        return -1;
    }

    n_inputs = atoi(argv[3]);
    n_heads = atoi(argv[4]);
    batch_size = atoi(argv[5]);

    printf("n_inputs: %d, n_heads: %d, batch_size: %d\n", n_inputs, n_heads, batch_size);

    // Dynamically allocate the matrices
    inputs = (float ****)malloc(batch_size * sizeof(float ***));
    outputs = (float ****)malloc(batch_size * sizeof(float ***));

    for (int i = 0; i < batch_size; i++) {
        inputs[i] = (float ***)malloc(n_heads * sizeof(float **));
        outputs[i] = (float ***)malloc(n_heads * sizeof(float **));
    }

    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < n_heads; j++) {
            inputs[i][j] = (float **)malloc(n_inputs * sizeof(float *));
            outputs[i][j] = (float **)malloc(n_inputs * sizeof(float *));
        }
    }

    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < n_heads; j++) {
            for (int k = 0; k < n_inputs; k++) {
                inputs[i][j][k] = (float *)malloc(n_inputs * sizeof(float));
                outputs[i][j][k] = (float *)malloc(n_inputs * sizeof(float));
            }
        }
    }


    // Load the matrices
    printf("Reading input file\n");
    readMatrix_batched_sm(inputFile, inputs, n_inputs, n_inputs, batch_size, n_heads);
    fclose(inputFile);

    printf("Computing softmax\n");

    clock_t start = clock();
    double cpu_time_used = softmax(inputs, n_inputs, batch_size, n_heads, outputs);
    clock_t end = clock();
    float seconds = (float)(end - start) / CLOCKS_PER_SEC;
    printf("Time taken by C code: %f seconds\n", seconds);


    // Open the output file for writing
    outputFile = fopen(argv[2], "wb");
    if (outputFile == NULL) {
        perror("Error opening output file");
        fclose(inputFile);
        return -1;
    }

    // Process and write the content of inputs in the output file
    int count = 0;
    printf("%d %d\n", n_inputs, n_heads);
    for (int k = 0; k < batch_size; k++) {
        for (int h = 0; h < n_heads; h++) {
            for (int i = 0; i < n_inputs; i++) {
                for (int j = 0; j < n_inputs; j++) {
                    fprintf(outputFile, "%f ", outputs[k][h][i][j]);
                    count++;
                    fprintf(outputFile, "\n");
                }
            }     
        }   
    }

    // append the time taken to the output file
    fprintf(outputFile, "%f\n", seconds);
    fprintf(outputFile, "%f\n", (float) cpu_time_used);

    // Close the file
    fclose(outputFile);

    // Free the dynamically allocated memory
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < n_heads; j++) {
            for (int k = 0; k < n_inputs; k++) {
                free(inputs[i][j][k]);
                free(outputs[i][j][k]);
            }
            free(inputs[i][j]);
            free(outputs[i][j]);
        }
        free(inputs[i]);
        free(outputs[i]);
    }    

    free(inputs);
    free(outputs);

    return 0;

}
