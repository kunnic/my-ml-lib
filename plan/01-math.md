# Phase 1: Math Primitives — Vectors, Matrices, Linear Algebra

## Why This Phase Matters
Everything in ML is built on linear algebra. A neural network is just matrix
multiplications + activation functions. Linear regression is solving a linear
system. You can't build ANY model without this foundation.

## Dependencies: None (this is the base layer)

---

## Step 1.0: C Memory Model

### KNOW (study these BEFORE writing any code)
- **Stack vs. Heap:**
  - Stack: automatic, fast, limited size (~8MB), dies when function returns.
    Local variables like `int x = 5;` or `double arr[10];` live here.
  - Heap: manual, unlimited (until RAM runs out), persists until you free it.
    Anything from malloc/calloc lives here. YOU must free it.
  - Rule: if the size is known at compile time and small, use stack.
    If unknown or large, use heap.

- **malloc(size):** Allocates `size` bytes on the heap. Returns void*.
  Does NOT initialize — memory contains garbage. Returns NULL if out of memory.
  Example: `double *arr = malloc(5 * sizeof(double));`

- **calloc(count, size):** Allocates `count * size` bytes, initialized to zero.
  Example: `double *arr = calloc(5, sizeof(double));` — all 5 doubles are 0.0.

- **free(ptr):** Releases heap memory. After free, the pointer is dangling —
  NEVER use it again. Set it to NULL after free as good practice.

- **Pointer arithmetic:**
  - `ptr + i` doesn't add `i` bytes — it adds `i * sizeof(*ptr)` bytes.
  - `double *p = arr; p[3]` is the same as `*(p + 3)` — both access the 4th element.
  - This is how you treat a flat malloc'd block as an array.

- **Structs:**
  - `typedef struct { double *data; int n; } Vec;` — defines a new type.
  - Access: `Vec v; v.n = 5;` (dot for stack structs)
  - Access: `Vec *v = malloc(...); v->n = 5;` (arrow for heap pointers)

- **Memory bugs:**
  - Leak: malloc without free. Memory grows until program crashes.
  - Double-free: free(ptr) twice. Corrupts heap, crashes unpredictably.
  - Use-after-free: access memory after free. Reads garbage or crashes.
  - Buffer overflow: write past end of allocated block. Corrupts other data.

- **-fsanitize=address:** Compiler flag that detects all the above at runtime.
  ALWAYS compile with this during development. Slight speed cost but catches
  bugs that would otherwise be silent and deadly.

### CODE
Write a program that:
1. `double *arr = malloc(5 * sizeof(double));`
2. Fill: `for (int i = 0; i < 5; i++) *(arr + i) = (double)(i + 1);`
3. Print: `for (int i = 0; i < 5; i++) printf("%.1f ", arr[i]);`
4. `free(arr); arr = NULL;`
Compile: `gcc -Wall -Wextra -fsanitize=address -g practice.c -o practice`

### MASTER — pass ALL before moving on
- [ ] Can explain stack vs heap: when each is used, pros/cons, without notes
- [ ] Can write malloc → fill → print → free from memory, compiles clean
- [ ] Can explain: what happens if you forget free? (leak)
- [ ] Can explain: what happens if you free twice? (corruption/crash)
- [ ] Can explain: what happens if you read after free? (garbage/crash)
- [ ] Can explain what -fsanitize=address does in one sentence
- [ ] Practice program: zero errors under sanitizer

---

## Step 1.1: Project Scaffolding + Makefile

### KNOW
- **Makefile anatomy:**
  ```
  target: prerequisites
  [TAB]recipe
  ```
  - Target: what to build (e.g., `libml.a`, `test_vector`)
  - Prerequisites: files it depends on (e.g., `vector.o matrix.o`)
  - Recipe: shell command(s) to run (MUST be indented with TAB, not spaces)

- **Automatic variables:**
  - `$@` = the target name
  - `$<` = the first prerequisite
  - `$^` = all prerequisites

- **Pattern rules:**
  - `%.o: %.c` — compile any .c to .o
  - Recipe: `$(CC) $(CFLAGS) -c $< -o $@`

- **.PHONY:** Tells make these targets aren't files:
  `.PHONY: all test clean`

- **Static library (.a file):**
  - It's an archive of .o files bundled together
  - Create: `ar rcs libml.a vector.o matrix.o linalg.o`
  - `ar` = archiver, `r` = replace, `c` = create, `s` = index
  - Link: `gcc test_vector.c -L. -lml -lm -o test_vector`

- **Header guards:** Prevent double-include errors
  ```c
  #ifndef ML_MATH_H
  #define ML_MATH_H
  // ... declarations ...
  #endif
  ```

- **-I flag:** `gcc -Iinclude` means "also look in include/ for #include files"

### CODE
Create these directories (if not existing): src/math/, tests/, examples/
Create Makefile:
- CC = gcc
- CFLAGS = -Wall -Wextra -std=c99 -Iinclude -fsanitize=address -g
- MATH_SRCS = $(wildcard src/math/*.c)
- MATH_OBJS = $(MATH_SRCS:.c=.o)
- all: libml.a
- libml.a: $(MATH_OBJS) → ar rcs $@ $^
- Pattern rule: %.o: %.c → $(CC) $(CFLAGS) -c $< -o $@
- test: libml.a → compile and run each tests/test_*.c
- clean: rm -f src/**/*.o libml.a bin/*
- .PHONY: all test clean

Create include/ml_math.h with header guards (empty body for now).

### MASTER
- [ ] `make clean && make` runs without errors (even with empty/stub .c files)
- [ ] Can explain: target, prerequisite, recipe — with an example
- [ ] Can explain: what `$@`, `$<`, `$^` mean
- [ ] Can explain: what `ar rcs libml.a *.o` does
- [ ] Can explain: why header guards exist (what goes wrong without them)
- [ ] Adding a new .c file to src/math/ — make picks it up automatically (wildcard)

---

## Step 1.2: Vec Struct + Lifecycle (create / free / print)

### KNOW
- **Vector:** An ordered list of n real numbers: v = [v₁, v₂, ..., vₙ] ∈ ℝⁿ
  - n is called the "dimension" or "length"
  - In code: a heap-allocated array of doubles + its length
  - Example: position in 3D space = [x, y, z], a vector in ℝ³

- **Ownership semantics:**
  - Whoever creates (mallocs) is NOT necessarily who frees
  - Convention in this library: functions that RETURN a pointer → caller owns it, caller frees
  - vec_create_from(vals, n) MUST copy the data (memcpy), not store the pointer
  - WHY: if you store the pointer, and the caller frees or modifies their array,
    your Vec's data becomes garbage. This is called a "dangling pointer."

- **const correctness:**
  - `const Vec *v` = "I promise this function won't modify the Vec"
  - Use for inputs: `double vec_dot(const Vec *a, const Vec *b);`
  - Helps catch bugs at compile time, documents intent

- **memcpy(dest, src, n_bytes):** Copies n_bytes from src to dest.
  Must not overlap. Use for: `memcpy(v->data, vals, n * sizeof(double));`

### CODE

File: include/ml_math.h
```c
#ifndef ML_MATH_H
#define ML_MATH_H

typedef struct {
    double *data;   // heap-allocated array of doubles
    int n;          // number of elements
} Vec;

// Lifecycle
Vec*  vec_create(int n);                          // all zeros
Vec*  vec_create_from(const double *vals, int n); // copy from array
void  vec_free(Vec *v);                           // safe free (handles NULL)
void  vec_print(const Vec *v);                    // [1.00, 2.00, 3.00]

#endif
```

File: src/math/vector.c
- vec_create: malloc the Vec struct, calloc the data array. Return NULL if either fails.
- vec_create_from: malloc Vec, malloc data, memcpy from vals. Return NULL on failure.
- vec_free: if (v == NULL) return; free(v->data); free(v);
- vec_print: printf("["); loop with "%.4f" and commas; printf("]\n");

### MASTER
- [ ] vec_create(3): all 3 elements are 0.0 — verified in test
- [ ] vec_create_from({1.0, 2.0, 3.0}, 3): values match original array
- [ ] Modify the original array after create_from → Vec's data unchanged (proves copy, not pointer share)
- [ ] vec_free(NULL) doesn't crash (test this explicitly)
- [ ] Can explain in own words: why memcpy, not `v->data = vals`
- [ ] Can explain: what const Vec* means and when to use it
- [ ] Zero sanitizer errors in create → use → free cycle

---

## Step 1.3: Vector Arithmetic (add / sub / scale)

### KNOW
- **Vector addition:** a + b = [a₁+b₁, a₂+b₂, ..., aₙ+bₙ]
  - Element-wise operation
  - REQUIRES same length (if a.n != b.n, it's undefined → return NULL)
  - Commutative: a + b = b + a
  - Associative: (a + b) + c = a + (b + c)

- **Scalar multiplication:** c·v = [c·v₁, c·v₂, ..., c·vₙ]
  - Stretches or shrinks the vector
  - c = 2: doubles each element
  - c = -1: reverses direction (negation)
  - c = 0: gives the zero vector

- **Subtraction:** a - b = a + (-1)·b = [a₁-b₁, a₂-b₂, ...]

- **Two memory patterns in C:**
  1. Allocate-and-return (vec_add, vec_sub):
     - Creates a NEW Vec inside the function
     - Returns pointer to caller
     - Caller MUST free when done
     - Original vectors are unchanged
  2. In-place (vec_scale):
     - Modifies the Vec you pass in
     - No new allocation
     - Returns void (or the same pointer)
     - Original vector IS changed

  Know when to use which:
  - Allocate-and-return: when you need both originals AND the result
  - In-place: when you just want to modify and don't need the old value

### CODE
Add to ml_math.h:
```c
Vec*  vec_add(const Vec *a, const Vec *b);   // returns NEW Vec = a + b
Vec*  vec_sub(const Vec *a, const Vec *b);   // returns NEW Vec = a - b
void  vec_scale(Vec *v, double scalar);      // IN-PLACE: v *= scalar
```

In src/math/vector.c:
- vec_add: check a->n == b->n (else NULL). vec_create(a->n). Loop: result->data[i] = a->data[i] + b->data[i]. Return result.
- vec_sub: same but subtraction
- vec_scale: loop: v->data[i] *= scalar. No allocation.

### MASTER
- [ ] [1,2,3] + [4,5,6] = [5,7,9]
- [ ] [1,2,3] - [4,5,6] = [-3,-3,-3]
- [ ] [1,2,3] scaled by 2.0 = [2,4,6]
- [ ] [1,2,3] scaled by 0.0 = [0,0,0]
- [ ] [1,2,3] scaled by -1.0 = [-1,-2,-3]
- [ ] Add vectors of different lengths → returns NULL
- [ ] Can trace vec_add and list every malloc that needs a matching free
- [ ] Run 100 iterations of: create two vecs → add → free all three → zero leaks
- [ ] Can explain the difference between allocate-and-return vs in-place patterns

---

## Step 1.4: Dot Product & Norm

### KNOW
- **Dot product (inner product):** a·b = Σᵢ aᵢ·bᵢ = a₁b₁ + a₂b₂ + ... + aₙbₙ
  - Takes two vectors of same length → returns ONE scalar (number)
  - NOT a vector! It collapses two vectors into a single number.
  - Properties:
    - Commutative: a·b = b·a
    - Distributive: a·(b+c) = a·b + a·c
    - Scalar associative: (ca)·b = c(a·b)

- **Geometric meaning of dot product:**
  - a·b = |a| × |b| × cos(θ), where θ is the angle between them
  - If perpendicular (90°): cos(90°) = 0, so dot = 0
  - If same direction (0°): cos(0°) = 1, so dot = |a|×|b| (maximum)
  - If opposite (180°): cos(180°) = -1, so dot = -|a|×|b| (most negative)
  - Positive dot → same general direction
  - Negative dot → opposite general direction
  - Zero dot → perpendicular (orthogonal)

- **Euclidean norm (L2 norm):** ‖v‖ = √(Σᵢ vᵢ²) = √(v·v)
  - The "length" or "magnitude" of the vector
  - Always ≥ 0. Only zero for the zero vector.
  - Example: ‖[3,4]‖ = √(9+16) = √25 = 5 (Pythagorean theorem!)
  - Derived from dot product: ‖v‖ = √(v·v)

- **Unit vector (normalization):** v̂ = v / ‖v‖
  - Has length/norm = 1
  - Points in the same direction as v, just scaled to length 1
  - Undefined for the zero vector (division by zero)

- **WHY THIS IS THE MOST IMPORTANT OPERATION IN ML:**
  - Linear regression prediction: ŷ = w·x + b (dot product of weights and input)
  - Every single neuron: output = activation(w·x + b)
  - Cosine similarity: sim(a,b) = (a·b) / (‖a‖·‖b‖) — used in NLP, search, recommendations
  - Loss functions use norms: MSE ∝ ‖y - ŷ‖²
  - Gradient descent: update direction is based on dot products

- **C math library:** #include <math.h>, compile with -lm flag
  - sqrt() for square root
  - fabs() for absolute value of doubles

### CODE
Add to ml_math.h:
```c
double vec_dot(const Vec *a, const Vec *b);   // a·b scalar
double vec_norm(const Vec *v);                // ‖v‖ (L2 norm)
Vec*   vec_normalize(const Vec *v);           // returns NEW unit vector v/‖v‖
```

In src/math/vector.c:
- vec_dot: check a->n == b->n. sum = 0; loop: sum += a->data[i] * b->data[i]; return sum.
- vec_norm: return sqrt(vec_dot(v, v)); — reuse your own function!
- vec_normalize: double n = vec_norm(v); if (n < 1e-12) return NULL; create new vec; loop: result->data[i] = v->data[i] / n;

### MASTER
- [ ] dot([1,0], [0,1]) = 0.0 — can explain WHY (perpendicular, cos90°=0)
- [ ] dot([1,2,3], [4,5,6]) = 32.0 — can hand-compute: 4+10+18
- [ ] dot([1,1], [1,1]) = 2.0
- [ ] norm([3,4]) = 5.0 — can explain: √(9+16) = √25, Pythagorean theorem
- [ ] norm([1,0,0]) = 1.0
- [ ] norm([0,0,0]) = 0.0
- [ ] normalize([3,4]) = [0.6, 0.8] — and norm(result) ≈ 1.0
- [ ] normalize zero vector → returns NULL (can't divide by zero)
- [ ] Can derive on paper: ‖v‖ = √(v·v) starting from the definition
- [ ] Can explain in 2 sentences: "Why is dot product the foundation of ML?"
- [ ] Can explain geometric meaning: positive/negative/zero dot product
- [ ] Knows to use -lm flag when linking

---

## Step 1.5: Mat Struct + Lifecycle

### KNOW
- **Matrix:** A 2D grid of real numbers. A ∈ ℝᵐˣⁿ = m rows × n columns.
  - Element Aᵢⱼ = value at row i, column j
  - Rows are horizontal, columns are vertical
  - m×n: m is ROWS (height), n is COLS (width)
  - Example: a 2×3 matrix has 2 rows and 3 columns = 6 elements

- **Row-major storage (CRITICAL to understand):**
  - A 2D matrix is stored as a FLAT 1D array in memory
  - Row after row: first row 0, then row 1, then row 2, ...
  - Index formula: Aᵢⱼ = data[i * cols + j]
  - Example for 3×4 matrix:
    ```
    Matrix view:        Memory (flat array):
    [ a b c d ]         [a, b, c, d, e, f, g, h, i, j, k, l]
    [ e f g h ]          ↑row0↑      ↑row1↑      ↑row2↑
    [ i j k l ]

    A[0][0]=a → data[0*4+0] = data[0]
    A[0][3]=d → data[0*4+3] = data[3]
    A[1][0]=e → data[1*4+0] = data[4]
    A[2][3]=l → data[2*4+3] = data[11]
    ```
  - WHY row-major: when you access a whole row sequentially,
    you're reading contiguous memory → CPU cache is happy → FAST.
    When you access a whole column, you jump by `cols` each time → cache misses → SLOW.

- **Identity matrix (I):**
  - Square matrix (n×n) with 1s on diagonal, 0s elsewhere
  - Iᵢⱼ = 1 if i==j, 0 if i≠j
  - Property: A×I = I×A = A for any compatible A (neutral element for multiplication)
  - Example 3×3: [[1,0,0], [0,1,0], [0,0,1]]

- **Square vs rectangular:**
  - Square: m == n (same rows and cols). Can have inverse, determinant.
  - Rectangular: m ≠ n. Cannot be inverted. Used for data matrices often.

### CODE
Add to ml_math.h:
```c
typedef struct {
    double *data;   // heap-allocated, row-major
    int rows;
    int cols;
} Mat;

Mat*   mat_create(int rows, int cols);           // all zeros (calloc)
Mat*   mat_create_identity(int n);               // n×n identity matrix
void   mat_free(Mat *m);                         // safe free (handles NULL)
double mat_get(const Mat *m, int i, int j);      // read Aᵢⱼ
void   mat_set(Mat *m, int i, int j, double v);  // write Aᵢⱼ
void   mat_print(const Mat *m);                  // pretty-print
```

In src/math/matrix.c:
- mat_create: malloc Mat, calloc data (rows*cols doubles). Return NULL on failure.
- mat_create_identity: mat_create(n,n), then loop: mat_set(m, i, i, 1.0) for i=0..n-1.
- mat_free: if(m==NULL) return; free(m->data); free(m);
- mat_get: return m->data[i * m->cols + j]; (optionally assert bounds)
- mat_set: m->data[i * m->cols + j] = v;
- mat_print: double loop, printf with alignment

### MASTER
- [ ] mat_create(3,4): all 12 elements are 0.0
- [ ] mat_create_identity(3): diagonal = 1.0, off-diagonal = 0.0
- [ ] mat_get/mat_set round-trip for all 4 corners: (0,0), (0,cols-1), (rows-1,0), (rows-1,cols-1)
- [ ] Can compute data[i*cols+j] in your head instantly for any (i,j,cols)
- [ ] Can draw flat memory layout of a 3×4 matrix on paper, labeling every index
- [ ] Can explain: why row access is fast and column access is slow in row-major
- [ ] mat_free(NULL) doesn't crash
- [ ] Zero memory leaks

---

## Step 1.6: Matrix Add & Transpose

### KNOW
- **Matrix addition:** C = A + B, where Cᵢⱼ = Aᵢⱼ + Bᵢⱼ
  - Element-wise (unlike matrix multiplication!)
  - REQUIRES same dimensions: both m×n
  - Properties: commutative (A+B = B+A), associative ((A+B)+C = A+(B+C))

- **Transpose:** B = Aᵀ, where Bᵢⱼ = Aⱼᵢ
  - Flip rows and columns. Row i becomes column i.
  - Shape changes: m×n → n×m
  - Example: [[1,2,3], [4,5,6]]ᵀ = [[1,4], [2,5], [3,6]]
  - Index mapping: transposed(i,j) reads from original(j,i)
  - In row-major: original data[j*n + i] becomes transposed data[i*m + j]

- **Properties to know BY HEART:**
  - (Aᵀ)ᵀ = A (transpose twice = original)
  - (A+B)ᵀ = Aᵀ + Bᵀ
  - (cA)ᵀ = c(Aᵀ)
  - A row vector [1, 2, 3] transposed = column vector [[1],[2],[3]]
  - For symmetric matrices: A = Aᵀ (it's its own transpose)

- **Why transpose matters in ML:**
  - (XᵀX) appears in linear regression normal equation
  - Backpropagation uses Wᵀ (transpose of weight matrix)
  - Switching between row and column representations of data

### CODE
Add to ml_math.h:
```c
Mat* mat_add(const Mat *a, const Mat *b);      // returns NEW Mat = A + B
Mat* mat_sub(const Mat *a, const Mat *b);      // returns NEW Mat = A - B
Mat* mat_transpose(const Mat *a);              // returns NEW Mat = Aᵀ
void mat_scale(Mat *m, double scalar);         // IN-PLACE: M *= scalar
```

### MASTER
- [ ] [[1,2],[3,4]] + [[5,6],[7,8]] = [[6,8],[10,12]] — by hand and code
- [ ] [[1,2],[3,4]]ᵀ = [[1,3],[2,4]] — by hand and code
- [ ] [[1,2,3],[4,5,6]]ᵀ shape is 3×2, not 2×3
- [ ] (Aᵀ)ᵀ = A — test passes (element-wise equality with tolerance)
- [ ] Can explain how row-major index mapping changes for transpose
- [ ] Dimension mismatch in add → returns NULL
- [ ] Can state (Aᵀ)ᵀ = A and (A+B)ᵀ = Aᵀ+Bᵀ from memory

---

## Step 1.7: Matrix Multiplication (THE critical step)

### KNOW
- **Definition:** C = A × B
  - A is m×k, B is k×n → C is m×n
  - Cᵢⱼ = Σₚ Aᵢₚ · Bₚⱼ (p goes from 0 to k-1)
  - Each element of C = dot product of ROW i of A with COLUMN j of B
  - Inner dimensions MUST match: (m×**k**) times (**k**×n)
  - Outer dimensions give the result shape: (**m**×k) times (k×**n**) → m×n

- **STEP BY STEP EXAMPLE:**
  ```
  A = [[1, 2],    B = [[5, 6],
       [3, 4]]         [7, 8]]

  C[0][0] = row0(A) · col0(B) = 1×5 + 2×7 = 5 + 14 = 19
  C[0][1] = row0(A) · col1(B) = 1×6 + 2×8 = 6 + 16 = 22
  C[1][0] = row1(A) · col0(B) = 3×5 + 4×7 = 15 + 28 = 43
  C[1][1] = row1(A) · col1(B) = 3×6 + 4×8 = 18 + 32 = 50

  C = [[19, 22],
       [43, 50]]
  ```

- **THIS IS NOT ELEMENT-WISE.** A_{ij} * B_{ij} is called Hadamard product.
  Matrix multiplication is completely different.

- **NOT commutative:** AB ≠ BA in general.
  - They might not even be the same shape! (2×3)(3×4) is 2×4, but (3×4)(2×3) is invalid.
  - Even for square matrices: AB ≠ BA usually.
  - Concrete counter-example:
    A = [[1,2],[3,4]], B = [[0,1],[1,0]]
    AB = [[2,1],[4,3]], BA = [[3,4],[1,2]] — different!

- **Properties to know BY HEART:**
  - AI = IA = A (identity is neutral)
  - (AB)C = A(BC) (associative)
  - A(B+C) = AB + AC (distributive)
  - (AB)ᵀ = BᵀAᵀ (transpose REVERSES the order!)
  - NOT commutative: AB ≠ BA

- **Time complexity:** O(m × k × n) — three nested loops.
  For two n×n matrices: O(n³). This is expensive for large matrices.

- **Why it's THE backbone of ML:**
  - Neural net forward pass: output = W × input + bias
  - Batch prediction: Y = X × W (all predictions at once)
  - Linear regression: (XᵀX)⁻¹Xᵀy uses two matrix multiplies
  - Backpropagation: gradient flows through transposed weight matrices

### CODE
Add to ml_math.h:
```c
Mat* mat_mul(const Mat *a, const Mat *b);  // returns NEW m×n Mat, or NULL
```

In src/math/matrix.c:
```
mat_mul(a, b):
  if (a->cols != b->rows) return NULL;  // inner dimensions must match
  Mat *c = mat_create(a->rows, b->cols);
  for (int i = 0; i < a->rows; i++)
    for (int j = 0; j < b->cols; j++)
      for (int p = 0; p < a->cols; p++)
        c->data[i * c->cols + j] += a->data[i * a->cols + p] * b->data[p * b->cols + j];
  return c;
```

### MASTER
- [ ] [[1,2],[3,4]] × [[5,6],[7,8]] = [[19,22],[43,50]] — computed BY HAND showing all 4 dot products
- [ ] (2×3)(3×4) = 2×4 — can explain why it works (inner dims match: 3==3)
- [ ] (2×3)(2×3) fails — can explain why (inner dims: 3 ≠ 2)
- [ ] A × I = A — test passes for 3×3 and 4×4
- [ ] Mismatched dims → returns NULL
- [ ] Can give a concrete pair A,B where AB ≠ BA
- [ ] Can state: (AB)ᵀ = BᵀAᵀ — and explain why the order reverses
- [ ] Can state time complexity: O(m·k·n), and explain the triple loop
- [ ] Can explain why matmul is the backbone of ML in 2-3 sentences

---

## Step 1.8: Matrix-Vector Multiply

### KNOW
- **y = Ax** where A is m×n, x has length n, y has length m
  - yᵢ = Σⱼ Aᵢⱼ · xⱼ = dot(row_i(A), x)
  - Each output element = dot product of one row of A with the input vector
  - This is a special case of mat_mul where B is an n×1 matrix

- **Why a separate function:**
  - Avoid creating/freeing a 1-column matrix wrapper
  - Cleaner API: takes Vec, returns Vec
  - Slightly more efficient (no extra allocation for column matrix)

- **This is what a neural network layer does:**
  - Input: vector x (neuron activations from previous layer)
  - Weights: matrix W (one row per output neuron)
  - Output: y = Wx + b (mat-vec multiply + bias addition)
  - Each output neuron i: yᵢ = Σⱼ Wᵢⱼ·xⱼ + bᵢ

- **WORKED EXAMPLE:**
  ```
  A = [[1, 2],     x = [1, 1]
       [3, 4]]

  y[0] = dot([1,2], [1,1]) = 1 + 2 = 3
  y[1] = dot([3,4], [1,1]) = 3 + 4 = 7

  y = [3, 7]
  ```

### CODE
Add to ml_math.h:
```c
Vec* mat_vec_mul(const Mat *m, const Vec *v);  // NEW Vec of length m->rows
```

Implementation: check m->cols == v->n. Create vec of length m->rows.
For each row i: result[i] = dot of row i with v.

### MASTER
- [ ] [[1,2],[3,4]] · [1,1] = [3,7] by hand AND in code
- [ ] [[1,0,0],[0,1,0],[0,0,1]] · [5,6,7] = [5,6,7] (identity does nothing)
- [ ] Can state: "each output = one row dotted with input" without thinking
- [ ] Dimension check: m->cols != v->n → returns NULL
- [ ] Can relate to neural networks: "this is one layer's forward pass without activation"

---

## Step 1.9: Gaussian Elimination — Solve Ax = b

### KNOW
- **System of linear equations:**
  - n equations, n unknowns
  - Example (2 equations, 2 unknowns):
    ```
    2x + 1y = 11
    5x + 3y = 27
    ```
  - In matrix form: Ax = b where A=[[2,1],[5,3]], x=[x,y], b=[11,27]

- **Three row operations (these DON'T change the solution):**
  1. Swap two rows
  2. Multiply a row by a nonzero scalar
  3. Add a multiple of one row to another

- **Forward elimination (step by step for the example):**
  ```
  Augmented matrix [A|b]:
  [ 2  1 | 11 ]
  [ 5  3 | 27 ]

  Goal: eliminate the 5 below the 2 (make lower-left = 0)
  Row2 = Row2 - (5/2)*Row1:
  multiplier = 5/2 = 2.5
  [ 2    1   | 11   ]
  [ 0   0.5  | -0.5 ]

  Now it's upper-triangular!
  ```

- **Back substitution (solve from bottom up):**
  ```
  From row 2: 0.5·y = -0.5  →  y = -1
  From row 1: 2·x + 1·(-1) = 11  →  2x = 12  →  x = 6

  Solution: x = 6, y = -1
  Check: 2(6) + 1(-1) = 11 ✓, 5(6) + 3(-1) = 27 ✓
  ```

- **Partial pivoting (DO THIS or your code will fail on real data):**
  - At step k, look at column k from row k downward
  - Find the row with the LARGEST |value| in that column
  - Swap that row into position k
  - WHY: dividing by a tiny number amplifies floating-point errors massively
  - Example: if pivot is 0.0000001, dividing by it multiplies errors by 10,000,000
  - With pivoting, you always divide by the largest available value → smallest error

- **Singular matrix:**
  - If, after pivoting, the best pivot is 0 (or < 1e-12) → system has no unique solution
  - Return NULL
  - Means the rows are "linearly dependent" — one row is a combination of others

- **CRITICAL RULES:**
  - NEVER modify the input A or b. Create an augmented copy [A|b].
  - Use fabs() for absolute value (not abs() — that's for ints!)
  - Use tolerance 1e-12 for "is this zero?" checks, not == 0.0

- **Why it matters for ML:**
  - Linear regression closed form: w = (XᵀX)⁻¹Xᵀy
  - This requires solving (XᵀX)w = Xᵀy, which is exactly Ax=b
  - Also used in: solving normal equations, least squares, ridge regression

### CODE
File: include/ml_linalg.h
```c
#ifndef ML_LINALG_H
#define ML_LINALG_H
#include "ml_math.h"

Vec* linalg_solve(const Mat *A, const Vec *b);  // solve Ax=b, NULL if singular

#endif
```

File: src/math/linalg.c

Implementation steps:
1. Validate: A must be square, A->rows == b->n
2. Build augmented matrix: n rows × (n+1) cols. Copy A into left, b into rightmost column.
3. Forward elimination with partial pivoting:
   - For each column k (0 to n-1):
     a. Find max |aug[i][k]| for i = k to n-1 → that's the pivot row
     b. If max < 1e-12 → singular, free augmented, return NULL
     c. Swap row k with pivot row (swap all n+1 elements)
     d. For each row i below (i = k+1 to n-1):
        - multiplier = aug[i][k] / aug[k][k]
        - aug[i][j] -= multiplier * aug[k][j] for all j
4. Back substitution:
   - For i = n-1 down to 0:
     - x[i] = aug[i][n] (the b column)
     - Subtract known terms: x[i] -= aug[i][j] * x[j] for j = i+1 to n-1
     - Divide: x[i] /= aug[i][i]
5. Return x as a Vec

### MASTER
- [ ] Can solve 2×2 system by hand showing every step of elimination
- [ ] Can trace 3×3 elimination step by step on paper
- [ ] Can explain partial pivoting: what, why, what goes wrong without it
- [ ] Concrete example of why pivoting matters:
      Without pivoting, if A[k][k] = 0.00001, multiplier = 100000 → error explodes.
      With pivoting, you'd swap in a row where that column is, say, 5.0 → much better.
- [ ] Test: [[2,1],[5,3]]x = [11,27] → x = [6,-1]
- [ ] Test: singular matrix (e.g., [[1,2],[2,4]]x = [3,6]) → returns NULL
- [ ] Test: Ix = b → x = b
- [ ] Test: A and b are UNMODIFIED after the call (print before and after)
- [ ] Knows: use fabs() not abs() for doubles
- [ ] Knows: check < 1e-12, never == 0.0

---

## Step 1.10: Matrix Inverse & Determinant

### KNOW
- **Inverse A⁻¹:**
  - A × A⁻¹ = A⁻¹ × A = I
  - Only exists for SQUARE, NON-SINGULAR matrices (det ≠ 0)
  - If you know A⁻¹, you can solve Ax=b as x = A⁻¹b

- **Gauss-Jordan method for inverse:**
  1. Build augmented matrix [A | I] — A on the left, identity on the right
  2. Row-reduce the LEFT side to become I (full elimination — above AND below each pivot)
  3. The RIGHT side becomes A⁻¹
  - This is like solving n systems Ax=eᵢ simultaneously (eᵢ = each column of identity)

- **2×2 shortcut (memorize this):**
  ```
  [[a, b], [c, d]]⁻¹ = (1 / (ad - bc)) × [[d, -b], [-c, a]]
  ```
  - ad - bc is the 2×2 determinant
  - Swap a↔d, negate b and c, divide by determinant

- **Determinant:**
  - A scalar that encodes whether a matrix is invertible
  - det(A) = 0 ↔ A is singular ↔ no inverse exists ↔ rows are linearly dependent
  - Geometric meaning: det = signed volume of the parallelepiped formed by row vectors.
    det = 0 means the vectors are "flat" (linearly dependent, collapsed dimension)

- **Computing determinant via elimination:**
  - Do Gaussian elimination (same as in Step 1.9)
  - det = (product of diagonal pivots) × (-1)^(number of row swaps)
  - Each row swap flips the sign of the determinant

- **Properties to know BY HEART:**
  - det(I) = 1
  - det(AB) = det(A) × det(B)
  - det(Aᵀ) = det(A)
  - det(cA) = cⁿ × det(A) for n×n matrix
  - det(A⁻¹) = 1/det(A)
  - det = 0 ↔ singular ↔ no inverse

- **CRITICAL: Floating-point tolerance:**
  - A × A⁻¹ will NOT be exactly I. Off-diagonal elements might be 1e-16 instead of 0.
  - ALWAYS compare with tolerance: fabs(actual - expected) < 1e-9
  - NEVER write: `if (mat_get(result, i, j) == 0.0)` — this WILL fail
  - Instead: `if (fabs(mat_get(result, i, j)) < 1e-9)`

### CODE
Add to ml_linalg.h:
```c
Mat*   linalg_inverse(const Mat *A);   // A⁻¹, NULL if singular
double linalg_det(const Mat *A);       // determinant
```

Implementation of linalg_inverse:
1. Validate: square matrix
2. Build augmented [A | I]: n rows × 2n cols
3. Forward elimination with partial pivoting (same as Step 1.9 but wider matrix)
4. ALSO do backward elimination (reduce ABOVE each pivot, not just below)
5. Scale each row so diagonal = 1.0
6. Extract right half as the result

Implementation of linalg_det:
1. Copy A (don't modify original)
2. Gaussian elimination with pivoting, count swaps
3. det = product of diagonal × (-1)^swaps
4. If any pivot < 1e-12 → det = 0.0

### MASTER
- [ ] 2×2 inverse by hand: [[4,7],[2,6]]⁻¹ = (1/10)[[6,-7],[-2,4]]
- [ ] 3×3 determinant by hand (pick one method: cofactor or elimination, master it)
- [ ] A × A⁻¹ ≈ I — test uses fabs() < 1e-9, NOT ==
- [ ] det([[1,2],[3,4]]) = 1×4 - 2×3 = -2
- [ ] det(I) = 1.0
- [ ] Singular: [[1,2],[2,4]] → inverse returns NULL, det = 0
- [ ] Can explain the full chain: det=0 ↔ singular ↔ no inverse ↔ linearly dependent rows
- [ ] Can explain why == fails for doubles (IEEE 754 rounding)
- [ ] Inverse of inverse ≈ original: (A⁻¹)⁻¹ ≈ A

---

## Step 1.11: Test Runner + All Tests

### KNOW
- **IEEE 754 floating-point:**
  - Computers store decimals in binary. Most decimal fractions are infinite in binary.
  - 0.1 in binary ≈ 0.0001100110011... (repeating). Gets truncated → tiny error.
  - 0.1 + 0.2 = 0.30000000000000004 (not 0.3)
  - This is why == NEVER works for computed doubles
  - Solution: compare with tolerance: fabs(a - b) < epsilon
  - Typical epsilon: 1e-9 for most tests, 1e-6 for accumulated operations

- **C preprocessor macros for test framework:**
  - __FILE__: expands to current filename string
  - __LINE__: expands to current line number
  - do { ... } while(0): wraps multi-statement macros safely
  - This lets ASSERT print exactly WHERE the failure occurred

- **Test framework design (keep it simple — ~25 lines):**
  ```c
  // tests/test_runner.h
  #include <stdio.h>
  #include <math.h>

  static int _tests_passed = 0;
  static int _tests_failed = 0;

  #define ASSERT(cond) do { \
      if (cond) { _tests_passed++; } \
      else { _tests_failed++; printf("FAIL: %s:%d: %s\n", __FILE__, __LINE__, #cond); } \
  } while(0)

  #define ASSERT_NEAR(a, b, eps) do { \
      if (fabs((double)(a) - (double)(b)) < (eps)) { _tests_passed++; } \
      else { _tests_failed++; printf("FAIL: %s:%d: %.10f != %.10f (eps=%.1e)\n", \
             __FILE__, __LINE__, (double)(a), (double)(b), (double)(eps)); } \
  } while(0)

  #define TEST_SUMMARY() printf("\n%d passed, %d failed\n", _tests_passed, _tests_failed); \
      return _tests_failed > 0 ? 1 : 0;
  ```

- **Test file structure:**
  ```c
  #include "test_runner.h"
  #include "ml_math.h"

  void test_vec_create() {
      Vec *v = vec_create(3);
      ASSERT(v != NULL);
      ASSERT_NEAR(v->data[0], 0.0, 1e-12);
      ASSERT(v->n == 3);
      vec_free(v);
  }

  int main() {
      test_vec_create();
      // ... more tests ...
      TEST_SUMMARY();
  }
  ```

### CODE
- tests/test_runner.h — the macro-based framework above
- tests/test_vector.c — ALL tests from Steps 1.2, 1.3, 1.4
- tests/test_matrix.c — ALL tests from Steps 1.5, 1.6, 1.7, 1.8
- tests/test_linalg.c — ALL tests from Steps 1.9, 1.10

Makefile test target:
```makefile
TEST_SRCS = $(wildcard tests/test_*.c)
TEST_BINS = $(TEST_SRCS:.c=)

test: libml.a $(TEST_BINS)
	@for t in $(TEST_BINS); do echo "=== $$t ===" && ./$$t; done

tests/test_%: tests/test_%.c libml.a
	$(CC) $(CFLAGS) $< -L. -lml -lm -o $@
```

### MASTER
- [ ] `make test` compiles and runs all 3 test files
- [ ] ALL assertions pass (zero failures)
- [ ] Zero memory errors from -fsanitize=address
- [ ] Can add a new test function + assertions without looking at examples
- [ ] Can explain: why ASSERT_NEAR and not ASSERT(a == b) for doubles
- [ ] Can explain: what the do{...}while(0) idiom accomplishes in macros

---

## PHASE 1 GRADUATION — All must be true before Phase 2:

### Knowledge checks (can do without ANY notes):
- [ ] Explain stack vs heap, malloc/calloc/free
- [ ] Hand-compute: dot product of two 3D vectors
- [ ] Hand-compute: 2×2 matrix multiplication, showing each dot product
- [ ] Hand-compute: transpose of a 2×3 matrix
- [ ] Hand-compute: 2×2 inverse using ad-bc formula
- [ ] Hand-compute: 2×2 determinant
- [ ] Solve a 2×2 linear system on paper using Gaussian elimination
- [ ] Explain row-major layout: draw flat memory for a 3×4 matrix
- [ ] Explain partial pivoting: what, why, what goes wrong without it
- [ ] Explain why dot product is the core operation in ML (2-3 sentences)
- [ ] Explain ownership: which functions allocate, who frees, why

### Code checks:
- [ ] `make test` — all green
- [ ] `make test` under -fsanitize=address — zero errors
- [ ] Every function that returns a pointer has proper NULL checks in tests
- [ ] All float comparisons use ASSERT_NEAR, never ==
