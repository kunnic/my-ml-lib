# Phase 3: Data Loading & Preprocessing

## Why This Phase Matters
Real ML works on real data. You need to load it, clean it, normalize it, and
split it into training/testing sets. Without this, you can only test on
hardcoded toy examples.

## Dependencies: Phase 1 (Vec, Mat), Phase 2 (stats for normalization, RNG for shuffling)

---

## Step 3.0: CSV Loader

### KNOW
- **CSV format:** Comma-Separated Values. Each line = one data row. Commas separate fields.
  ```
  sepal_length,sepal_width,petal_length,petal_width,species
  5.1,3.5,1.4,0.2,0
  4.9,3.0,1.4,0.2,0
  7.0,3.2,4.7,1.4,1
  ```
  - First row is usually a header (column names) — skip it
  - Each subsequent row = one data sample
  - Last column is often the label/target

- **Parsing strategy:**
  1. First pass: count lines to know how many rows
  2. Second pass (or rewind with rewind(file)): parse each line
  3. Store features in a Mat (n_samples × n_features)
  4. Store labels in a Vec (n_samples)

- **String parsing in C:**
  - fopen(path, "r"): open file for reading. Returns NULL if not found.
  - fgets(buffer, size, file): read one line into buffer. Returns NULL at EOF.
  - strtok(str, ","): split string by comma delimiter.
    First call: strtok(line, ",") — starts tokenizing.
    Subsequent calls: strtok(NULL, ",") — continues from last position.
  - atof(str): convert string to double. Returns 0.0 for non-numeric strings.
  - fclose(file): always close when done.
  - WARNING: strtok modifies the input string! Use on a buffer, not the original.

- **Counting lines:**
  ```c
  int count = 0;
  char buf[4096];
  while (fgets(buf, sizeof(buf), file)) count++;
  rewind(file);  // go back to start for second pass
  ```

- **Error handling:**
  - File doesn't exist → fopen returns NULL → return NULL
  - Malformed line → skip or handle gracefully
  - Buffer size: 4096 chars per line is usually plenty

### CODE
File: include/ml_data.h, src/data/csv.c
```c
#ifndef ML_DATA_H
#define ML_DATA_H

#include "ml_math.h"

typedef struct {
    Mat *X;        // features matrix (n_samples × n_features)
    Vec *y;        // labels vector (n_samples)
    int n_samples;
    int n_features;
} Dataset;

Dataset* data_load_csv(const char *filepath, int has_header, int label_col);
void     data_free(Dataset *d);

#endif
```

Implementation of data_load_csv:
1. fopen the file. If NULL → return NULL.
2. If has_header, read and discard first line.
3. First pass: count lines (= n_samples). Also count commas in first data line to get n_features.
4. rewind(file), skip header again if needed.
5. Allocate Mat(n_samples, n_features) and Vec(n_samples).
6. Second pass: for each line, tokenize by comma, fill Mat row and Vec label.
7. fclose. Return Dataset.

### MASTER
- [ ] Can load a simple CSV with 4 feature columns and 1 label column
- [ ] has_header=1 skips first line, has_header=0 doesn't
- [ ] X has correct shape: n_samples × n_features
- [ ] y has correct length: n_samples
- [ ] Values match what's in the file (spot-check a few rows)
- [ ] File not found → returns NULL (no crash, no leak)
- [ ] data_free properly frees X, y, and the Dataset struct
- [ ] No buffer overflows (tested with sanitizer)
- [ ] Can explain the two-pass parsing strategy

---

## Step 3.1: Data Preprocessing

### KNOW
- **Why normalize/standardize data:**
  - Features have wildly different scales: age (0-100), salary (20000-500000), height (1.5-2.0)
  - Gradient descent treats all features equally in terms of step size
  - Large-scale features dominate the gradient → model converges slowly or poorly
  - Normalization puts all features on similar scales → faster, more stable convergence
  - Example: without normalization, gradient is huge for salary, tiny for height.
    GD overshoots on salary, barely moves on height.

- **Min-max normalization:** x' = (x - min) / (max - min)
  - Maps every feature to [0, 1] range
  - min → 0, max → 1, everything else proportional
  - Preserves relative relationships within each feature
  - Sensitive to outliers: one extreme value compresses all others into a tiny range
  - Apply PER COLUMN (each feature independently)

- **Z-score standardization:** x' = (x - mean) / std
  - Maps to mean=0, std=1 (standard normal scale)
  - More robust to outliers than min-max
  - Common default choice in practice
  - After standardization: ~68% of values in [-1, 1], ~95% in [-2, 2]
  - Apply PER COLUMN (each feature independently)

- **When to use which:**
  - Min-max: when you need bounded output (e.g., image pixels 0-1)
  - Z-score: general default, especially with gradient-based methods
  - Neither: tree-based models (decision trees) don't need normalization

- **Train/test split:**
  - Divide data into training set (learn from) and test set (evaluate on)
  - NEVER evaluate on training data — you'll think the model is better than it is (overfitting)
  - Typical ratios: 80/20, 70/30
  - MUST be random: if data is sorted by class, taking first 80% = biased split
  - Implementation:
    1. Create array of indices [0, 1, 2, ..., n-1]
    2. Fisher-Yates shuffle (from Phase 2)
    3. First 80% of shuffled indices → train, last 20% → test
    4. Copy corresponding rows into train/test datasets

- **One-hot encoding:**
  - Convert class labels (integers) into binary vectors
  - If n_classes = 3: label 0 → [1,0,0], label 1 → [0,1,0], label 2 → [0,0,1]
  - Creates a Mat of shape (n_samples × n_classes)
  - Needed for: softmax output layer, multi-class cross-entropy loss
  - Each row has exactly one 1, rest are 0

### CODE
Add to ml_data.h, implement in src/data/preprocess.c:
```c
// Normalization (IN-PLACE, column by column)
void data_normalize_minmax(Mat *X);
void data_normalize_zscore(Mat *X);

// Train/test split
typedef struct {
    Dataset *train;
    Dataset *test;
} TrainTestSplit;

TrainTestSplit* data_train_test_split(const Dataset *d, double test_ratio, unsigned int seed);
void            data_split_free(TrainTestSplit *s);

// One-hot encoding
Mat* data_one_hot(const Vec *labels, int n_classes);
```

Implementation of data_normalize_minmax:
- For each column j (0 to X->cols-1):
  - Find min and max of that column
  - If max == min (constant feature): set all to 0.0 (avoid division by zero)
  - Else: X[i][j] = (X[i][j] - min) / (max - min) for each row i

Implementation of data_train_test_split:
1. Create index array [0, 1, ..., n-1]
2. rng_seed(seed) then rng_shuffle_indices
3. n_test = (int)(n_samples * test_ratio)
4. n_train = n_samples - n_test
5. Create train Dataset (n_train rows) and test Dataset (n_test rows)
6. Copy rows according to shuffled indices

### MASTER
- [ ] After min-max: each column has min ≈ 0.0, max ≈ 1.0
- [ ] After z-score: each column has mean ≈ 0.0, std ≈ 1.0
- [ ] Constant column (all same value): doesn't crash, becomes all 0.0
- [ ] Train/test split with 0.2 ratio: ~20% in test, ~80% in train
- [ ] train.n_samples + test.n_samples == original n_samples (no data lost)
- [ ] No duplicates: every original sample appears in exactly one split
- [ ] Same seed → same split (reproducible)
- [ ] Different seed → different split
- [ ] One-hot: labels [0,1,2,0] → [[1,0,0],[0,1,0],[0,0,1],[1,0,0]]
- [ ] One-hot: each row sums to 1.0
- [ ] Can explain why normalization helps gradient descent (2-3 sentences)
- [ ] Can explain why train/test split must be random

---

## PHASE 3 GRADUATION:
- [ ] Can load any well-formed CSV with numeric data + labels
- [ ] Can normalize by min-max and z-score, and explain when each is appropriate
- [ ] Can perform reproducible train/test splits
- [ ] Can explain why splitting is necessary (overfitting prevention)
- [ ] Can explain one-hot encoding purpose and usage
- [ ] All tests pass, zero sanitizer errors, zero memory leaks
