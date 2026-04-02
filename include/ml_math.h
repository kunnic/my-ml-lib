#ifndef ML_MATH_H
#define ML_MATH_H

typedef struct {
	double *data;
	int size;
} Vector;

// Vector init
Vector* vector_create(int size);
Vector* vector_create_from(const double *values, int size);
void 	vector_free(Vector *vector);
void 	vector_print(const Vector *vector);

// Vector aritmethic
Vector* vector_add(const Vector* a, const Vector* b);
Vector* vector_sub(const Vector* a, const Vector* b);
Vector*	vector_scale(Vector* vector, double scalar);
double vector_dot(const Vector* a, const Vector* b);
#endif
