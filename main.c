#include <stdio.h>
#include "ml_math.h"

int main(void) {
	// --- vector_create: zero-initialized vector ---
	printf("=== vector_create ===\n");
	Vector *v_zeros = vector_create(4);
	vector_print(v_zeros);
	printf("]\n\n");

	// --- vector_create_from: copy from array ---
	printf("=== vector_create_from ===\n");
	double vals_a[] = {1.0, 2.0, 3.0};
	double vals_b[] = {4.0, 5.0, 6.0};
	Vector *a = vector_create_from(vals_a, 3);
	Vector *b = vector_create_from(vals_b, 3);
	printf("a = "); vector_print(a); printf("]\n");
	printf("b = "); vector_print(b); printf("]\n\n");

	// --- vector_add: a + b ---
	printf("=== vector_add ===\n");
	Vector *add_result = vector_add(a, b);
	printf("a + b = "); vector_print(add_result); printf("]\n\n");

	// --- vector_sub: a - b ---
	printf("=== vector_sub ===\n");
	Vector *sub_result = vector_sub(a, b);
	printf("a - b = "); vector_print(sub_result); printf("]\n\n");

	// --- vector_scale: in-place scale ---
	printf("=== vector_scale ===\n");
	double vals_c[] = {1.0, 2.0, 3.0};
	Vector *c = vector_create_from(vals_c, 3);
	printf("before scale: "); vector_print(c); printf("]\n");
	vector_scale(c, 2.0);
	printf("after *2.0:   "); vector_print(c); printf("]\n\n");

	// --- vector_free: NULL safety ---
	printf("=== vector_free ===\n");
	vector_free(NULL); // should not crash
	printf("vector_free(NULL) OK\n");

	// --- cleanup ---
	vector_free(v_zeros);
	vector_free(a);
	vector_free(b);
	vector_free(add_result);
	vector_free(sub_result);
	vector_free(c);

	printf("\nDone.\n");
	return 0;
}
