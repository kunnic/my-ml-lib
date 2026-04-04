#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ml_math.h"

Vector* vector_create(int size) {
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
		return NULL;
	}
	vector->size = size;

	return vector;
}

Vector* vector_create_from(const double* values, int size) {
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

void vector_free(Vector* vector) {
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

void vector_print(const Vector* vector) {
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

Vector* vector_add(const Vector* a, const Vector* b) {
	if (
		a == NULL ||
		b == NULL ||
		a->data == NULL ||
		b->data == NULL
	) {return NULL;}

	if (a->size <=0 || a->size != b->size) {return NULL;}
	
	Vector *vector = vector_create(a->size);
	if (vector == NULL) {
		return NULL;
	}

	int i;
	for (i = 0; i < a->size; i++) {
		vector->data[i] = a->data[i] + b->data[i];
	}

	return vector;
}

Vector* vector_sub(const Vector* a, const Vector* b) {
	if (
		a == NULL ||
		b == NULL ||
		a->data == NULL ||
		b->data == NULL
	) {return NULL;}

	if (a->size <=0 || a->size != b->size) {return NULL;}
	
	Vector *vector = vector_create(a->size);
	if (vector == NULL) {
		return NULL;
	}

	int i;
	for (i = 0; i < a->size; i++) {
		vector->data[i] = a->data[i] - b->data[i];
	}

	return vector;
}

Vector* vector_scale(Vector* vector, double scalar) {
	if (
		vector == NULL ||
		vector->size <= 0 ||
		vector->data == NULL
	) {return NULL;}

	int i;
	for (i = 0; i < vector->size; i++) {
		vector->data[i] *= scalar;
	}

	return vector;
}

double vector_dot(const Vector* a, const Vector* b) {
	if (
		a == NULL ||
		b == NULL ||
		a->data == NULL ||
		b->data == NULL
	) {return 0.0;}

	if (a->size <=0 || a->size != b->size) {return 0.0;}
	
	int i;
	double sum = 0.0;
	for (i = 0; i < a->size; i++) {
		sum += a->data[i] * b->data[i];
	}

	return sum;
}

double vector_norm(const Vector* vector) {
	return sqrt(vector_dot(vector, vector));
}

double vector_L2(const Vector* a, const Vector* b) {
	return sqrt(
		vector_dot(a, a) + vector_dot(b, b) - 2 * vector_dot(a, b)
	);
}

double vector_normalize(const Vector* vector) {
	if (vector == NULL) {return NULL;}

	double norm = vector_norm(vector);

	if (norm < 1e-12) {return NULL;}

	Vector* vector_result = vector_create(vector->size);
	if (vector_result == NULL) {
		return NULL;
	}

	int i;
	for (i = 0; i < vector_result->size; i++) {
		vector_result->data[i] = vector->data[i] / norm; 
	}

	return vector_result;
}
