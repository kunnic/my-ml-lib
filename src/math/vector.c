#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ml_math.h"

Vector* create_vector(int size) {
	// Initialize a vector type Vector, return NULL if fails
	
	// Cannot allocate a vector of size 0 or lower
	if (size <= 0) {return NULL;}
	
	// Allocating the vector:
	// 	- a pointer of type double (an array - vector)
	//	- an integer as size
	Vector *vector = malloc(sizeof(Vector));
	if (vector == NULL) {return NULL;}
	
	// Allocates double data array
	vector->data = calloc(size, sizeof(double));
	if (vector->data == NULL) {
		free(vector);
		return NULL
	}
	vector->size = size;

	return vector;
}

Vector* create_vector_from(const double* values, int size) {
	if (size <= 0 || values == NULL) {return NULL;}

	Vector* vector = malloc(sizeof(Vector));
	if (vector == NULL) {return NULL;}

	vector->data = malloc(size * sizeof(double));
	if (vector->data == NULL) {
		free(vector);
		return NULL;
	}
	
	memcpy(vector->data, values, size * sizeof(double));
	vector->size = size;

	return vector;
}

void free_vector(Vector* vector) {
	// Scenario:
	// 	- vector have data
	// 	- vector have null
	// 	- vector is null
	//
	// 	!!! DANGEROUS IF THE VECTOR IS ALREADY FREED !!!
	if (vector == NULL) {return;}
	
	free(vector->data);
	free(vector);
}

void print_vector(const Vector* vector) {
	if (vector == NULL) {return;}
	if (vector->data == NULL) {
		printf("Vector size %d with no values", vector->size);
		return;
	}
	int i;
	printf("Vector size %d, content:\n[", vector->size);
	for (i = 0; i < vector->size; i++) {
		//printf("%d,", *(vector->data + i));
		printf("%.4f,", vector->data[i]);
	}
}
