# Phase 6: Evaluation & Putting It All Together

## Why This Phase Matters
A model is useless if you can't measure how good it is. Evaluation tells you
whether your model actually learned something, and metrics help you compare
models and diagnose problems.

## Dependencies: ALL previous phases

---

## Step 6.0: Regression Metrics

### KNOW
- **Mean Squared Error (MSE):**
  MSE = (1/n) Σ(y_pred - y_true)²
  - Most common regression metric
  - Penalizes large errors heavily (squared)
  - Units: target variable squared (e.g., dollars²)
  - Always ≥ 0. Lower = better. Perfect = 0.

- **Root Mean Squared Error (RMSE):**
  RMSE = √MSE
  - Same as MSE but in original units (e.g., dollars)
  - More interpretable: "on average, predictions are off by RMSE units"
  - Always ≥ 0. Lower = better.

- **Mean Absolute Error (MAE):**
  MAE = (1/n) Σ|y_pred - y_true|
  - Less sensitive to outliers than MSE (no squaring)
  - In original units
  - Use when outliers should not dominate the metric

- **R² (Coefficient of Determination):**
  R² = 1 - (SS_res / SS_tot)
  SS_res = Σ(y_true - y_pred)²  (residual sum of squares)
  SS_tot = Σ(y_true - y_mean)²  (total sum of squares)
  - Measures: how much better is your model than just predicting the mean?
  - R² = 1: perfect predictions
  - R² = 0: model is as good as predicting the mean (useless)
  - R² < 0: model is WORSE than predicting the mean (broken)
  - R² = 0.85 means "model explains 85% of the variance in the data"

### CODE
File: include/ml_eval.h, src/eval/metrics.c
```c
#ifndef ML_EVAL_H
#define ML_EVAL_H

#include "ml_math.h"

// Regression metrics
double eval_mse(const Vec *y_true, const Vec *y_pred);
double eval_rmse(const Vec *y_true, const Vec *y_pred);
double eval_mae(const Vec *y_true, const Vec *y_pred);
double eval_r2(const Vec *y_true, const Vec *y_pred);

#endif
```

### MASTER
- [ ] MSE of identical vectors = 0.0
- [ ] RMSE = sqrt(MSE) always
- [ ] MAE ≤ RMSE always (mathematical property)
- [ ] R² = 1.0 for perfect predictions
- [ ] R² ≈ 0.0 when predicting the mean
- [ ] Can hand-compute: y_true=[1,2,3], y_pred=[1.1,2.2,2.8]
      MSE = (0.01 + 0.04 + 0.04)/3 = 0.03
      RMSE = √0.03 ≈ 0.173
      MAE = (0.1 + 0.2 + 0.2)/3 = 0.167

---

## Step 6.1: Classification Metrics

### KNOW
- **Accuracy:** (correct predictions) / (total predictions)
  - Simple and intuitive
  - MISLEADING for imbalanced data: 99% accuracy can be terrible
  - Example: 99 negative, 1 positive. Predict all negative → 99% accuracy. But you missed the one positive case entirely.

- **Confusion Matrix:**
  For binary classification:
  ```
                Predicted
                0         1
  Actual  0    TN        FP
          1    FN        TP
  ```
  TN (True Negative): correctly predicted 0
  FP (False Positive): predicted 1, actually 0 (Type I error)
  FN (False Negative): predicted 0, actually 1 (Type II error)
  TP (True Positive): correctly predicted 1

- **Precision:** TP / (TP + FP)
  "Of all the things I predicted as positive, how many actually were?"
  High precision = few false alarms.
  Example: spam filter. High precision = rarely marks good email as spam.

- **Recall (Sensitivity):** TP / (TP + FN)
  "Of all actual positives, how many did I catch?"
  High recall = rarely misses a positive case.
  Example: disease detection. High recall = rarely misses a sick patient.

- **F1 Score:** 2 * (precision * recall) / (precision + recall)
  Harmonic mean of precision and recall.
  Balances precision and recall — high only when BOTH are high.
  F1 = 1.0 is perfect. F1 = 0.0 is worst.

- **Precision vs Recall tradeoff:**
  By adjusting the classification threshold (default 0.5):
  - Higher threshold (0.9): more conservative. Higher precision, lower recall.
  - Lower threshold (0.1): more aggressive. Lower precision, higher recall.
  - There's no free lunch: you can't maximize both simultaneously (usually).

- **Multi-class extension:**
  For K classes, confusion matrix is K × K.
  Precision/recall computed per class, then averaged:
  - Macro average: compute metric per class, then average (treats all classes equally)
  - Can be computed but not critical for now — start with binary.

### CODE
Add to ml_eval.h, implement in src/eval/metrics.c:
```c
// Classification metrics
double eval_accuracy(const Vec *y_true, const Vec *y_pred);
double eval_precision(const Vec *y_true, const Vec *y_pred, int positive_class);
double eval_recall(const Vec *y_true, const Vec *y_pred, int positive_class);
double eval_f1(const Vec *y_true, const Vec *y_pred, int positive_class);

// Confusion matrix
typedef struct {
    int **matrix;     // matrix[actual][predicted]
    int  n_classes;
} ConfusionMatrix;

ConfusionMatrix* eval_confusion_matrix(const Vec *y_true, const Vec *y_pred, int n_classes);
void             eval_cm_print(const ConfusionMatrix *cm);
void             eval_cm_free(ConfusionMatrix *cm);
```

Implementation of eval_confusion_matrix:
1. Allocate n_classes × n_classes integer matrix (all zeros)
2. For each sample: matrix[(int)y_true[i]][(int)y_pred[i]]++
3. Return

### MASTER
- [ ] Accuracy of 100% correct predictions = 1.0
- [ ] Can hand-compute confusion matrix:
      y_true=[0,0,1,1,1], y_pred=[0,1,1,1,0]
      TN=1, FP=1, FN=1, TP=2
      Accuracy = 3/5 = 0.6
      Precision = 2/3 ≈ 0.667
      Recall = 2/3 ≈ 0.667
      F1 = 2*(0.667*0.667)/(0.667+0.667) ≈ 0.667
- [ ] Can explain precision/recall tradeoff with a real-world example
- [ ] Can explain why accuracy is misleading for imbalanced data
- [ ] Confusion matrix diagonal = correct predictions
- [ ] cm_print shows a readable table

---

## Step 6.2: End-to-End Example Programs

### KNOW
This is where you put it all together in complete programs that demonstrate
the full pipeline: load data → preprocess → train → evaluate → report.

### CODE
File: examples/linear_regression_demo.c
```c
// Full pipeline:
// 1. Load CSV (maybe a synthetic y = 3x₁ + 2x₂ + 1 + noise)
// 2. Normalize features
// 3. Train/test split (80/20)
// 4. Train LinReg with normal equation
// 5. Train LinReg with GD
// 6. Predict on test set
// 7. Print MSE, RMSE, MAE, R² for both approaches
// 8. Compare: are the weights similar?
```

File: examples/classification_demo.c
```c
// Full pipeline:
// 1. Load Iris dataset (or similar)
// 2. Normalize features
// 3. Train/test split
// 4. Train LogReg on binary subset (e.g., class 0 vs class 1)
// 5. Train KNN (k=5)
// 6. Predict
// 7. Print accuracy, precision, recall, F1 for both
// 8. Print confusion matrices
// 9. Train decision tree, compare
```

File: examples/neural_network_demo.c
```c
// Full pipeline:
// 1. Load dataset (Iris for multi-class, or XOR for sanity)
// 2. One-hot encode labels
// 3. Normalize features
// 4. Train/test split
// 5. Create MLP: [n_features, 16, n_classes]
// 6. Train with mini-batch SGD
// 7. Print loss curve (every 100 epochs)
// 8. Predict on test set
// 9. Print accuracy and confusion matrix
```

### MASTER
- [ ] Each demo runs end-to-end without crashing
- [ ] Each demo prints meaningful metrics
- [ ] Linear regression: R² > 0.9 on synthetic linear data
- [ ] Classification: accuracy > 80% on separable data
- [ ] Neural network: loss consistently decreases, accuracy > 70%
- [ ] All demos clean up memory (no leaks reported by sanitizer)

---

## Step 6.3: Documentation

### KNOW
- Write a README.md for the project with:
  - What is this library
  - How to build (make)
  - How to run tests (make test)
  - How to run examples (make examples)
  - Brief description of each module
  - What you learned

- Optionally: add Doxygen-style comments to public header functions
  ```c
  /**
   * @brief Create a new vector of given size.
   * @param n Number of elements.
   * @return Pointer to new Vec, or NULL on failure.
   */
  Vec* vec_create(int n);
  ```

### MASTER
- [ ] README exists and is accurate
- [ ] Someone else could clone the repo, build, run tests, and run examples
- [ ] Each header file has brief comments explaining the module's purpose

---

## PHASE 6 GRADUATION (and PROJECT GRADUATION):
- [ ] All metrics implemented and tested
- [ ] At least 3 end-to-end examples work correctly
- [ ] README documents the entire library
- [ ] make builds the library
- [ ] make test runs all tests and they pass
- [ ] make examples builds all demos
- [ ] Zero memory leaks on all tests and examples
- [ ] You can explain every model, every metric, and every component
- [ ] You built a machine learning library from scratch in C. 🎓
