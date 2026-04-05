#ifndef ML_MATH_H
#define ML_MATH_H

typedef struct {
	double *data;
	int size;
} Vector;

typedef struct {
	double *data;
	int rows;
	int cols;
} Matrix;

// Vector init
Vector* vector_create(int size);
Vector* vector_create_from(const double *values, int size);
void 	vector_free(Vector *vector);
void 	vector_print(const Vector *vector);

// Vector aritmethic
Vector* vector_add(const Vector* a, const Vector* b);
Vector* vector_sub(const Vector* a, const Vector* b);
Vector*	vector_scale(Vector* vector, double scalar);
double 	vector_dot(const Vector* a, const Vector* b);
double 	vector_norm(const Vector* vector);
double 	vector_L2(const Vector* a, const Vector* b);
Vector* vector_normalize(const Vector* vector);

// Matrix init
Matrix* matrix_create(int rows, int cols);
Matrix* matrix_create_identity(int n);

void 	matrix_free(Matrix* matrix);
double 	matrix_get(const Matrix* matrix, int row, int col);
void 	matrix_set(Matrix* matrix, int row, int col, double value);
void 	matrix_print(const Matrix* matrix);

// Matrix aritmethic
Matrix* matrix_add(const Matrix* a, const Matrix* b);
Matrix* matrix_sub(const Matrix* a, const Matrix* b);
Matrix* matrix_transpose(const Matrix* matrix);
void 	matrix_scale(Matrix *matrix, double scalar);
Matrix* matrix_mul(const Matrix* a, const Matrix* b);
#endif
