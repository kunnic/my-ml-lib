# ML Library from Scratch — Overview & Architecture

## Goal
Build a complete educational machine learning library in pure C, implementing
everything from basic math (vectors, matrices) through optimization to ML models
(linear regression, neural networks). No external dependencies except libc/libm.

## Architecture

```
my-ml-lib/
├── Makefile
├── include/
│   ├── ml_math.h        # Vec and Mat types + all math operations
│   ├── ml_linalg.h      # Linear algebra (solve, inverse, determinant)
│   ├── ml_stats.h       # Statistics (mean, variance, correlation)
│   ├── ml_functions.h   # Activation functions + loss functions + derivatives
│   ├── ml_random.h      # RNG, Gaussian random, shuffling
│   ├── ml_data.h        # CSV loading, preprocessing, train/test split
│   ├── ml_optim.h       # Gradient descent, SGD, mini-batch, Adam
│   ├── ml_linear.h      # Linear regression, logistic regression
│   ├── ml_knn.h         # k-Nearest Neighbors
│   ├── ml_tree.h        # Decision tree
│   ├── ml_nn.h          # Neural network (MLP)
│   └── ml_eval.h        # Evaluation metrics
├── src/
│   ├── math/
│   │   ├── vector.c
│   │   ├── matrix.c
│   │   ├── linalg.c
│   │   ├── stats.c
│   │   ├── functions.c
│   │   └── random.c
│   ├── data/
│   │   ├── csv.c
│   │   └── preprocess.c
│   ├── optim/
│   │   └── gradient_descent.c
│   ├── models/
│   │   ├── linear_regression.c
│   │   ├── logistic_regression.c
│   │   ├── knn.c
│   │   ├── decision_tree.c
│   │   └── nn.c
│   └── eval/
│       └── metrics.c
├── tests/
│   ├── test_runner.h
│   ├── test_vector.c
│   ├── test_matrix.c
│   ├── test_linalg.c
│   ├── test_stats.c
│   ├── test_functions.c
│   ├── test_data.c
│   ├── test_optim.c
│   ├── test_linear.c
│   ├── test_knn.c
│   ├── test_tree.c
│   ├── test_nn.c
│   └── test_eval.c
└── examples/
    ├── linear_reg_demo.c
    ├── logistic_reg_demo.c
    ├── knn_demo.c
    ├── tree_demo.c
    └── nn_xor.c
```

## Conventions
- Language: C99, compiled with gcc
- CFLAGS: -Wall -Wextra -std=c99 -Iinclude -fsanitize=address -g
- Link: -lm (for math.h functions)
- All floating-point: double (not float)
- All matrices: row-major storage
- Memory ownership: functions that return pointers allocate — caller must free
- Error handling: return NULL on failure (bad dimensions, singular matrix, malloc fail)
- Float comparison: NEVER use ==. Always use fabs(a-b) < epsilon
- Naming: module_action (vec_create, mat_mul, linalg_solve, nn_forward)

## Phase Dependency Chain
Phase 1 (Math) → Phase 2 (Stats/Functions) → Phase 3 (Data) → Phase 4 (Optim) → Phase 5 (Models) → Phase 6 (Eval)

Each phase depends on ALL previous phases. Master each before moving on.

## Makefile Targets
- make           — builds libml.a from all src/**/*.c
- make test      — builds + runs all tests/*.c, links against libml.a
- make examples  — builds all examples/*.c
- make clean     — removes *.o, *.a, bin/*
