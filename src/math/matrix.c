#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "ml_math.h"

Matrix* matrix_create(int rows, int cols) {
    if (rows <= 0 || cols <= 0) {return NULL;}

    Matrix* matrix = malloc(sizeof(Matrix));
    if (matrix == NULL) {return NULL;}
    
    matrix->data = calloc(rows * cols, sizeof(double));
    if (matrix->data == NULL) {
        free(matrix);
        return NULL;
    }
        
    matrix->rows = rows;
    matrix->cols = cols;

    return matrix;
}

Matrix* matrix_create_identity(int n) {
    Matrix* matrix = matrix_create(n, n);
    if (matrix == NULL) {
        return NULL;
    }

    int i; 
    for (i = 0; i < n; i++) {
        matrix->data[(i * n) + i] = 1.0;
    }

    return matrix;
}

void matrix_free(Matrix* matrix){
    if (matrix == NULL) {return;}

    free(matrix->data);
    free(matrix);
}

double matrix_get(const Matrix* matrix, int row, int col) {
    if (matrix == NULL || matrix->data == NULL) {return 0.0;}
    
    if (
        row < 0 || 
        col < 0 ||
        row >= matrix->rows || 
        col >= matrix->cols
    ){
        return 0.0;
    }
    
    return matrix->data[row * matrix->cols + col];
}

void matrix_set(Matrix* matrix, int row, int col, double value) {
    if (matrix == NULL || matrix->data == NULL) {return;}
    
    if (
        row < 0 || 
        col < 0 ||
        row >= matrix->rows || 
        col >= matrix->cols
    ){
        return;
    }

    matrix->data[row * matrix->cols + col] = value;
}

void matrix_print(const Matrix* matrix) {
    if (matrix == NULL) {
        printf("This matrix has not been initialized!\n");
        return;
    }
    if (matrix->data == NULL) {
        printf("This matrix is empty!\n");
        return;
    }

    int row, col;
    for (row = 0; row < matrix->rows; row++) {
        printf("Row %d: [", row);
        for (col = 0; col < matrix->cols; col++) {
            printf("%.4f", matrix->data[row * matrix->cols + col]);
            if (col < matrix->cols - 1) {
                printf(", ");
            }
        }
        printf("]\n");
    }
}

Matrix* matrix_add(const Matrix* a, const Matrix* b) {
	if (
		a == NULL ||
		b == NULL ||
		a->data == NULL ||
		b->data == NULL
	) {return NULL;}
		
	if (
		a->cols != b->cols ||
		a->rows != b->rows
	) {return NULL;}
	
	int row, col;
	int index;

	Matrix* matrix = matrix_create(a->rows, a->cols);
	if (matrix == NULL) {return NULL;}
	
	for (index = 0; index < matrix->rows * matrix->cols; index++) {
		matrix->data[index] = a->data[index] + b->data[index];
	}

	return matrix;
}

Matrix* matrix_sub(const Matrix* a, const Matrix* b) {
	if (
		a == NULL ||
		b == NULL ||
		a->data == NULL ||
		b->data == NULL
	) {return NULL;}
		
	if (
		a->cols != b->cols ||
		a->rows != b->rows
	) {return NULL;}
	
	int row, col;
	int index;

	Matrix* matrix = matrix_create(a->rows, a->cols);
	if (matrix == NULL) {return NULL;}
	
	for (index = 0; index < matrix->rows * matrix->cols; index++) {
		matrix->data[index] = a->data[index] - b->data[index];
	}

	return matrix;
}

void matrix_scale(Matrix* matrix, double scalar) {
	if (matrix == NULL || matrix->data == NULL) {return;}
	
	int i;
	for (i = 0; i < matrix->cols * matrix->rows; i++) {
		matrix->data[i] *= scalar;
	}
}

Matrix* matrix_transpose(const Matrix* matrix){
	if (matrix == NULL || matrix->data == NULL) {return;}
	
	Matrix* new_matrix = matrix_create(matrix->rows, matrix->cols);
	
	int row, col;
	int source_index, dest_index;
	for (row = 0; row < matrix->rows; row++) {
		for (col = 0; col < matrix->cols; col++) {
			source_index = row * matrix->cols + col;
			dest_index = col * matrix->rows + row;

			new_matrix->data[dest_index] = matrix->data[source_index];
		}
	}

	return new_matrix;
}

Matrix* matrix_mul(const Matrix* a, const Matrix* b) {
	if (
		a == NULL ||
		b == NULL ||
		a->data == NULL ||
		b->data == NULL
	) {return NULL;}
		
	if (a->cols != b->rows) {return NULL;}
	
	int row, col;
	int index;

	Matrix* matrix = matrix_create(a->rows, a->cols);
	if (matrix == NULL) {return NULL;}
	
	for (index = 0; index < matrix->rows * matrix->cols; index++) {
		matrix->data[index] = a->data[index] - b->data[index];
	}

	return matrix;
}
