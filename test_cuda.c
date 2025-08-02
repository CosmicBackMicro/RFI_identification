#include <stdio.h>
#include <stdlib.h>
#include "include/cuda_acceleration.h"

int main() {
    printf("Testing CUDA functionality...\n");
    
    if (!cuda_isAvailable()) {
        printf("CUDA not available\n");
        return 1;
    }
    
    if (cuda_init() != 0) {
        printf("CUDA initialization failed\n");
        return 1;
    }
    
    // Test matrix transpose
    int rows = 4, cols = 3;
    float input[] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f,
        10.0f, 11.0f, 12.0f
    };
    
    float *output = (float*)malloc(rows * cols * sizeof(float));
    
    printf("Original matrix (%dx%d):\n", rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%6.1f ", input[i * cols + j]);
        }
        printf("\n");
    }
    
    // Test CUDA matrix transpose
    cuda_transpose(input, output, rows, cols);
    printf("Transposed matrix (%dx%d):\n", cols, rows);
    for (int i = 0; i < cols; i++) {
        for (int j = 0; j < rows; j++) {
            printf("%6.1f ", output[i * rows + j]);
        }
        printf("\n");
    }
    
    free(output);
    cuda_cleanup();
    printf("CUDA test completed successfully!\n");
    
    return 0;
}
