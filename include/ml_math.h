#ifndef ML_MATH_H
#define ML_MATH_H

typedef struct {
	double *data;
	int size;
} Vector;

Vector* create_vector(int size);

Vector* create_vector_from(const double *values, int size);

void 	free_vector(Vector *vector);

void 	print_vector(const Vector *vector);

#endif
