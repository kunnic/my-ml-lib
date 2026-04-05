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
	printf("vector_free(NULL) OK\n\n");

	// --- vector_dot ---
	printf("=== vector_dot ===\n");
	printf("[1,2,3]·[4,5,6] = %.4f  (expect 32.0000)\n", vector_dot(a, b));
	double perp_x[] = {1.0, 0.0};
	double perp_y[] = {0.0, 1.0};
	Vector *vx = vector_create_from(perp_x, 2);
	Vector *vy = vector_create_from(perp_y, 2);
	printf("[1,0]·[0,1]     = %.4f  (expect 0.0000, perpendicular)\n\n", vector_dot(vx, vy));

	// --- vector_norm ---
	printf("=== vector_norm ===\n");
	double vals_34[] = {3.0, 4.0};
	Vector *v34 = vector_create_from(vals_34, 2);
	printf("norm([3,4]) = %.4f  (expect 5.0000)\n\n", vector_norm(v34));

	// --- vector_L2 ---
	printf("=== vector_L2 (Euclidean distance) ===\n");
	printf("L2([1,2,3], [4,5,6]) = %.4f\n\n", vector_L2(a, b));

	// --- vector_normalize ---
	printf("=== vector_normalize ===\n");
	Vector *v_unit = vector_normalize(v34);
	printf("normalize([3,4]) = "); vector_print(v_unit); printf("]\n");
	printf("norm of result   = %.4f  (expect 1.0000)\n\n", vector_norm(v_unit));

	// --- cleanup ---
	vector_free(v_zeros);
	vector_free(a);
	vector_free(b);
	vector_free(add_result);
	vector_free(sub_result);
	vector_free(c);
	vector_free(vx);
	vector_free(vy);
	vector_free(v34);
	vector_free(v_unit);

	printf("Done.\n");
	return 0;
}
