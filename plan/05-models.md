# Phase 5: Machine Learning Models

## Why This Phase Matters
This is the culmination. Everything you built — math, stats, data loading,
optimization — comes together here. Each model teaches fundamentally different
approaches to learning from data.

## Dependencies: ALL previous phases

---

## Step 5.0: Linear Regression

### KNOW
- **What it does:** Predicts a continuous output (e.g., house price, temperature)
  by fitting a linear function: y_pred = Xw + b (or y_pred = X_aug @ w with bias column)

- **Two approaches:**
  1. **Closed-form (Normal Equation):** w = (XᵀX)⁻¹Xᵀy
     - Exact solution, no iteration needed
     - O(n³) for matrix inversion — impractical for large n_features
     - Works perfectly for small to medium problems
     - You already have all the math for this from Phase 1!
  2. **Gradient Descent:**
     - Uses the GD code from Phase 4
     - Loss: MSE = (1/n) Σ(y_pred - y_true)²
     - Gradient: ∇L = (2/n) Xᵀ(Xw - y)
     - Scales to very large datasets

- **Bias term:**
  Instead of separate weight vector w and scalar b, prepend a column of 1s to X.
  X_augmented = [1 | X]  (first column is all ones)
  Then w includes the bias as w[0], and y_pred = X_aug @ w handles everything.

- **Assumptions of linear regression:**
  - Relationship between X and y is approximately linear
  - Errors (residuals) are normally distributed
  - Features are not perfectly correlated with each other (multicollinearity)
  - Not required to understand deeply now, but good to know

### CODE
File: include/ml_models.h, src/models/linear_regression.c
```c
#ifndef ML_MODELS_H
#define ML_MODELS_H

#include "ml_math.h"
#include "ml_data.h"
#include "ml_optim.h"

// ── Linear Regression ──────────────────────
typedef struct {
    Vec *weights;       // includes bias as weights[0]
    int  n_features;    // original feature count (without bias column)
    int  fitted;        // 0 = not trained, 1 = trained
} LinReg;

LinReg* linreg_create(int n_features);
void    linreg_free(LinReg *m);

// Closed-form solution
int     linreg_fit_normal(LinReg *m, const Mat *X, const Vec *y);

// Gradient descent solution
int     linreg_fit_gd(LinReg *m, const Mat *X, const Vec *y, const GDConfig *cfg);

// Predict
Vec*    linreg_predict(const LinReg *m, const Mat *X);
```

Implementation of linreg_fit_normal:
1. Augment X: prepend column of 1s → X_aug (n × (d+1))
2. Compute XᵀX → result is (d+1) × (d+1)
3. Compute Xᵀy → result is (d+1) vector
4. Solve: w = inverse(XᵀX) * Xᵀy (or use your mat_solve)
5. Store weights in model, set fitted=1
6. Free temporaries

Implementation of linreg_predict:
1. Assert model is fitted
2. Augment X with 1s column
3. Compute X_aug @ weights → predictions vector
4. Return predictions

### MASTER
- [ ] Normal equation gives exact solution for small problems
- [ ] GD gives approximately the same weights as normal equation (close, not exact)
- [ ] predict returns sensible values (not NaN, not wildly off)
- [ ] y = 2x + 3 dataset → weights ≈ [3.0, 2.0] (bias=3, slope=2)
- [ ] Predictions on training data match actual y values closely (low MSE)
- [ ] Can explain why normal equation is O(n³) and when GD is preferred
- [ ] Can write the MSE gradient formula from memory
- [ ] Model refuses to predict if not fitted

---

## Step 5.1: Logistic Regression

### KNOW
- **What it does:** Binary classification (predict 0 or 1).
  Examples: spam/not-spam, sick/healthy, cat/dog.

- **Key insight:** Use linear regression + sigmoid to get a probability.
  z = Xw + b (linear combination, same as linear regression)
  p = sigmoid(z) = 1/(1+e^(-z))  → outputs probability in [0, 1]
  Predict class 1 if p ≥ 0.5, else class 0.

- **Why not just use linear regression for classification?**
  Linear regression can output values < 0 or > 1 — not probabilities.
  Sigmoid squashes everything to [0, 1] — valid probability.

- **Loss function: Binary Cross-Entropy (BCE)**
  L = -(1/n) Σ [y_i * log(p_i) + (1-y_i) * log(1-p_i)]
  - When y=1: loss = -log(p). High p → low loss. Low p → high loss (penalizes wrong prediction).
  - When y=0: loss = -log(1-p). Low p → low loss. High p → high loss.
  - This is a convex function → gradient descent finds the global minimum.

- **Gradient of BCE w.r.t. weights:**
  ∇L = (1/n) Xᵀ(sigmoid(Xw) - y)
  Remarkably similar to linear regression gradient! The sigmoid is already baked in.

- **Decision boundary:**
  The set of points where Xw + b = 0 (i.e., sigmoid = 0.5).
  For 2D: w₁x₁ + w₂x₂ + b = 0 → a line dividing the two classes.

- **No closed-form solution.** Must use gradient descent.

### CODE
Add to ml_models.h, implement in src/models/logistic_regression.c:
```c
typedef struct {
    Vec *weights;
    int  n_features;
    int  fitted;
    double threshold;   // default 0.5
} LogReg;

LogReg* logreg_create(int n_features);
void    logreg_free(LogReg *m);
int     logreg_fit(LogReg *m, const Mat *X, const Vec *y, const GDConfig *cfg);
Vec*    logreg_predict_proba(const LogReg *m, const Mat *X);  // raw probabilities
Vec*    logreg_predict(const LogReg *m, const Mat *X);        // 0 or 1
```

Implementation of logreg_fit:
1. Augment X with 1s column
2. Set up grad_fn: computes (1/n) Xᵀ(sigmoid(Xw) - y)
3. Set up loss_fn: computes BCE loss
4. Call gd_batch (or gd_sgd) with these functions
5. Store resulting weights, set fitted=1

The grad_fn and loss_fn can be static functions in the .c file.

### MASTER
- [ ] Can classify a linearly separable 2D dataset with >90% accuracy
- [ ] predict_proba returns values between 0 and 1
- [ ] predict returns only 0s and 1s
- [ ] BCE loss decreases over training epochs
- [ ] Can hand-compute: sigmoid(0)=0.5, sigmoid(large)≈1, sigmoid(-large)≈0
- [ ] Can derive the BCE gradient from memory
- [ ] Can explain why sigmoid is used (maps to probability)
- [ ] Can explain what the decision boundary is
- [ ] Threshold of 0.5 is default; can be changed for precision/recall tradeoff

---

## Step 5.2: K-Nearest Neighbors (KNN)

### KNOW
- **What it does:** Classifies a point based on the K closest training points.
  No "training" phase — it just memorizes data and computes at predict time.
  This is called a "lazy learner" or "instance-based learning."

- **Algorithm:**
  1. Given a test point x
  2. Compute distance from x to EVERY training point
  3. Find the K training points with smallest distance
  4. Count class labels among those K neighbors
  5. Predict the most common class (majority vote)

- **Distance metrics:**
  - Euclidean: d = √(Σ(x_i - y_i)²)  — most common, uses vec_dist from Phase 1
  - Manhattan: d = Σ|x_i - y_i|  — useful for high dimensions
  - Others exist but start with Euclidean

- **Choosing K:**
  - K=1: nearest neighbor. Very sensitive to noise (one outlier = wrong prediction)
  - K too large: everything gets classified as the majority class
  - Good starting point: K = √n (square root of training set size)
  - K should be odd (avoids ties in binary classification)

- **Time complexity:**
  - Training: O(1) — literally just store the data
  - Prediction for one point: O(n × d) — compute distance to all n training points
  - For large datasets, this is SLOW. That's the tradeoff.

- **No training → no gradient descent needed.**
  KNN is fundamentally different from regression models!

### CODE
Add to ml_models.h, implement in src/models/knn.c:
```c
typedef struct {
    Mat *X_train;     // stored training features (reference or copy)
    Vec *y_train;     // stored training labels
    int  k;           // number of neighbors
    int  n_classes;
} KNN;

KNN*  knn_create(int k, int n_classes);
void  knn_free(KNN *m);
void  knn_fit(KNN *m, const Mat *X, const Vec *y);  // just stores data
Vec*  knn_predict(const KNN *m, const Mat *X_test);
int   knn_predict_single(const KNN *m, const Vec *x); // classify one point
```

Implementation of knn_predict_single:
1. Compute distance from x to each training point (vec_dist)
2. Find K indices with smallest distances
   - Simple approach: maintain a sorted list of (distance, index) pairs
   - Or: compute all distances, then do partial sort / selection
3. Count class votes among K nearest neighbors
4. Return the class with the most votes

Finding K smallest — simple approach:
- Allocate array of (distance, label) pairs, length n_train
- Fill with distances
- Sort by distance (simple selection sort is fine for now)
- Take first K entries, count votes

### MASTER
- [ ] K=1: prediction = label of nearest training point
- [ ] K=n: prediction = majority class in entire dataset (useless classifier)
- [ ] Accuracy on a simple 2D dataset (two well-separated clusters) is near 100%
- [ ] Can explain the time/space tradeoff of KNN vs parametric models
- [ ] Performance degrades gracefully with noise
- [ ] Can explain why K should be odd for binary classification
- [ ] knn_fit is O(1), knn_predict is O(n × d × n_test)

---

## Step 5.3: Decision Tree (Classification)

### KNOW
- **What it does:** Builds a tree of if/else questions to classify data.
  Example tree for "should I play tennis?":
  ```
  Is outlook sunny?
  ├── Yes: Is humidity > 70?
  │   ├── Yes: Don't play
  │   └── No: Play
  └── No: Is it windy?
      ├── Yes: Don't play
      └── No: Play
  ```

- **How it learns (building the tree):**
  At each node, pick the BEST feature and BEST threshold to split the data.
  "Best" = the split that most reduces impurity (makes child nodes purer).

- **Impurity measures:**
  - **Gini impurity:** Gini(S) = 1 - Σ(p_i²)
    where p_i is the proportion of class i in the set S.
    - Pure node (all same class): Gini = 0
    - Maximally mixed (50/50 binary): Gini = 0.5
    - Range: [0, 0.5] for binary, [0, 1-1/K] for K classes
  - **Information gain (entropy-based):**
    Entropy(S) = -Σ(p_i * log₂(p_i))
    IG = Entropy(parent) - weighted_avg(Entropy(children))
    More theoretically motivated but similar in practice.
  - Start with Gini — it's simpler and just as effective.

- **Splitting algorithm (greedy):**
  For each node:
  1. For each feature j:
     a. For each unique value v in that feature:
        - Split data into left (feature_j ≤ v) and right (feature_j > v)
        - Compute Gini of left and right
        - Weighted average = (n_left/n)*Gini_left + (n_right/n)*Gini_right
  2. Pick the (feature, value) pair with lowest weighted Gini
  3. Create two child nodes with the split data
  4. Recurse on children

- **Stopping criteria:**
  - Max depth reached (e.g., depth ≤ 5)
  - Too few samples in a node (e.g., < 5 samples)
  - Node is pure (Gini = 0, all same class)
  When stopping: create a leaf node, predict majority class.

- **Prediction:** Walk the tree from root, follow the appropriate branch at each node,
  return the label at the leaf.

### CODE
Add to ml_models.h, implement in src/models/decision_tree.c:
```c
typedef struct TreeNode {
    int    is_leaf;
    int    predicted_class;    // only for leaf nodes
    int    split_feature;      // which feature to split on
    double split_value;        // threshold value
    struct TreeNode *left;     // feature ≤ value
    struct TreeNode *right;    // feature > value
    int    depth;
} TreeNode;

typedef struct {
    TreeNode *root;
    int max_depth;
    int min_samples;
    int n_classes;
} DecTree;

DecTree*  dtree_create(int max_depth, int min_samples, int n_classes);
void      dtree_free(DecTree *m);
void      dtree_fit(DecTree *m, const Mat *X, const Vec *y);
Vec*      dtree_predict(const DecTree *m, const Mat *X_test);
int       dtree_predict_single(const DecTree *m, const Vec *x);
```

Implementation of find_best_split:
1. Initialize best_gini = infinity
2. For each feature j in 0..n_features-1:
   a. For each sample i (use its value as threshold):
      - Split: left = rows where X[row][j] <= X[i][j], right = rest
      - Compute weighted Gini
      - If < best_gini: update best (feature=j, value=X[i][j])
3. Return best split (feature, value)

Implementation of build_tree (recursive):
1. If depth >= max_depth OR n_samples < min_samples OR node is pure → create leaf
2. Find best split
3. Partition data into left and right subsets
4. Recursively build left and right subtrees
5. Return node

### MASTER
- [ ] Can compute Gini impurity by hand for a simple set
      Example: [A,A,A,B,B] → p_A=3/5, p_B=2/5, Gini=1-(9/25+4/25)=12/25=0.48
- [ ] Tree correctly classifies linearly separable data
- [ ] Tree correctly classifies non-linear patterns (e.g., XOR pattern)
- [ ] max_depth=1 → stump (one split), limited accuracy
- [ ] max_depth=very large → overfits training data (100% train accuracy, lower test accuracy)
- [ ] Can print/display the tree structure (which feature, which threshold)
- [ ] Can explain greedy splitting and why it's not guaranteed to find the globally optimal tree
- [ ] Free correctly deallocates all nodes (recursive free, test with sanitizer)

---

## Step 5.4: Neural Network / Multi-Layer Perceptron (MLP)

### KNOW
- **What it does:** Universal function approximator. Can learn ANY mapping from inputs
  to outputs, given enough neurons and data. This is the foundation of deep learning.

- **Architecture:**
  ```
  Input layer → Hidden layer(s) → Output layer
  [x₁, x₂, x₃] → [h₁, h₂, h₃, h₄] → [y₁, y₂]
  ```
  Each arrow = weighted connection. Each node = weighted sum + activation function.

- **Forward pass (computing output):**
  For a 2-layer network (1 hidden layer):
  ```
  z₁ = X @ W₁ + b₁            // linear transform
  a₁ = relu(z₁)                // activation (non-linearity)
  z₂ = a₁ @ W₂ + b₂           // another linear transform
  output = softmax(z₂)         // final activation (for classification)
  ```
  Where:
  - W₁ shape: (n_features × n_hidden)
  - b₁ shape: (n_hidden)
  - W₂ shape: (n_hidden × n_classes)
  - b₂ shape: (n_classes)

- **Why non-linearity (activation function) is essential:**
  Without it: hidden = X @ W₁, output = hidden @ W₂ = X @ (W₁ @ W₂) = X @ W_combined
  It collapses to a single linear transform! Activation functions are what give NNs power.

- **Backpropagation:**
  Chain rule applied backwards through the network to compute gradients.
  For a 2-layer network:
  ```
  // Output layer errors
  dz₂ = output - y_onehot                   // for softmax + cross-entropy
  dW₂ = (1/n) * a₁ᵀ @ dz₂
  db₂ = (1/n) * column_sums(dz₂)

  // Hidden layer errors (chain rule!)
  da₁ = dz₂ @ W₂ᵀ
  dz₁ = da₁ ⊙ relu_derivative(z₁)          // ⊙ = element-wise multiply
  dW₁ = (1/n) * Xᵀ @ dz₁
  db₁ = (1/n) * column_sums(dz₁)
  ```
  Then update: W₁ -= lr * dW₁, W₂ -= lr * dW₂, etc.

- **relu_derivative:**
  relu(x) = max(0, x)
  relu'(x) = 1 if x > 0, else 0

- **Loss: Cross-Entropy**
  L = -(1/n) Σᵢ Σⱼ y_onehot[i][j] * log(output[i][j])
  Measures how different the predicted probability distribution is from truth.

- **Weight initialization:**
  - All zeros: TERRIBLE. All neurons compute the same thing. Symmetry = they stay identical forever.
  - Random small values: standard approach.
  - Xavier/Glorot: random from N(0, sqrt(2/(fan_in+fan_out))). Best practice for sigmoid/tanh.
  - He: random from N(0, sqrt(2/fan_in)). Best for ReLU.
  - For now: random uniform in [-0.5, 0.5] is fine.

### CODE
Add to ml_models.h, implement in src/models/mlp.c:
```c
typedef struct {
    int    n_layers;       // number of weight matrices (layers - 1)
    int   *layer_sizes;    // array: [n_input, n_hidden1, ..., n_output]
    Mat  **W;              // weight matrices: W[0] is input→hidden1, etc.
    Vec  **b;              // bias vectors
    // Cache for backprop
    Mat  **z;              // pre-activation values (before activation)
    Mat  **a;              // post-activation values (after activation)
} MLP;

typedef struct {
    double lr;
    int    max_epochs;
    int    batch_size;
    unsigned int seed;
} MLPConfig;

MLP*  mlp_create(const int *layer_sizes, int n_layers, unsigned int seed);
void  mlp_free(MLP *m);
Mat*  mlp_forward(MLP *m, const Mat *X);            // returns output probabilities
void  mlp_backward(MLP *m, const Mat *X, const Mat *Y_onehot, double lr);
void  mlp_fit(MLP *m, const Mat *X, const Vec *y, int n_classes, const MLPConfig *cfg);
Vec*  mlp_predict(const MLP *m, const Mat *X);
```

Implementation of mlp_forward:
1. a[0] = X (input is the first "activation")
2. For each layer l:
   z[l] = a[l-1] @ W[l] + b[l]  (broadcast bias across rows)
   a[l] = relu(z[l])             // for hidden layers
   a[last] = softmax(z[last])    // for output layer
3. Return a[last]

Implementation of mlp_backward:
1. dz = a[last] - Y_onehot  (softmax + cross-entropy derivative)
2. For l from last layer down to 0:
   dW[l] = (1/n) * a[l]ᵀ @ dz
   db[l] = (1/n) * column_sums(dz)
   W[l] -= lr * dW[l]
   b[l] -= lr * db[l]
   if not first layer:
     da = dz @ W[l]ᵀ
     dz = da ⊙ relu_derivative(z[l])

### MASTER
- [ ] Forward pass produces valid probabilities (sum to 1 per row, all ≥ 0)
- [ ] Simple problem: XOR pattern with 1 hidden layer (4 neurons) converges
- [ ] Loss decreases consistently over training
- [ ] Network with 0 hidden layers ≈ logistic regression (sanity check)
- [ ] Can hand-trace forward pass for a 2-input, 2-hidden, 1-output network
- [ ] Can explain why non-linearity is essential (without it = just linear)
- [ ] Can explain backpropagation as chain rule applied backwards
- [ ] Can explain why zero initialization breaks (symmetry problem)
- [ ] Different initializations produce different training curves
- [ ] Can predict on test data after training, accuracy is reasonable
- [ ] MLPs can learn non-linear decision boundaries (unlike logistic regression)

---

## PHASE 5 GRADUATION:
- [ ] Can implement and use all 5 models from scratch
- [ ] Can explain when to use each model (problem type, data size, interpretability)
- [ ] Can train on one dataset and evaluate on another (no data leakage)
- [ ] Can compare models: LinReg for regression, LogReg for binary, KNN for small data,
      DecTree for interpretability, MLP for complex patterns
- [ ] Understand overfitting vs underfitting and how it manifests in each model
- [ ] All models tested on at least one real dataset (e.g., Iris, simple synthetic data)
- [ ] Zero memory leaks, zero sanitizer errors across all models
