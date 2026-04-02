#include <stdio.h>
#include <stdlib.h>

int main() {
	int n = 5;

	/*
	int *p = malloc(5 * sizeof(int));

	int i;
	for (i = 0; i < 5; i++) {
		p[i] = 0;
	}

	for (i = 0; i < 5; i++) {
		printf("index %d, value %d", i, p[i]);
	}

	free(p);

	return; */

	int n = 5;
	int i;

	int* p = calloc(5 * sizeof(int));

	if (p == NULL) {
		printf("No mem to alloc!");
		return 1;	
	}
	
	for (i = 0; i < n; i++) {
		printf("index %d, value %d\n", i, p[i]);
	}

	free(p);

	return 0;
}
