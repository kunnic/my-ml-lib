# Phase 4: Optimization (Gradient Descent)

## Why This Phase Matters
Optimization is the ENGINE of machine learning. Models learn by minimizing a
loss function, and gradient descent is how they do it. Without understanding
GD deeply, you'll never understand why models work, fail, or misbehave.

## Dependencies: Phase 1 (Vec, Mat for parameter storage), Phase 2 (RNG for SGD shuffling)

---

## Step 4.0: Batch Gradient Descent

### KNOW
- **What is optimization?**
  The process of finding parameters (weights) that minimize a loss function.
  Loss function = measure of how wrong the model is. Lower = better.

- **Gradient:**
  A vector of partial derivatives. Points in the direction of steepest INCREASE.
  To minimize, go in the OPPOSITE direction (negative gradient).
  For a loss function L(w₁, w₂):
    ∇L = [∂L/∂w₁, ∂L/∂w₂]
  Update: w_new = w_old - lr * ∇L

- **Learning rate (lr):**
  Controls step size. Too large → overshoots, diverges. Too small → painfully slow.
  Typical starting values: 0.01, 0.001, 0.0001
  You MUST tune this for each problem — there is no universal best value.

- **Batch gradient descent:**
  Computes gradient using ALL training samples, then takes one step.
  One pass through all data = one "epoch."
  Properties:
  - Gradient estimate is accurate (averaged over all data)
  - Each step is expensive (process all samples)
  - Smooth convergence curve
  - For n samples, n_features dimensions: gradient computation is O(n × n_features)

- **Convergence:**
  How do you know when to stop?
  1. Max epochs: stop after N iterations (e.g., 1000)
  2. Loss threshold: stop when loss < some small value
  3. Loss change: stop when |loss_new - loss_old| < epsilon (loss barely changing)
  In practice, use a combination: stop when converged OR max epochs reached.

- **The gradient descent loop:**
  ```
  for epoch in 1..max_epochs:
      grad = compute_gradient(X, y, weights)  // using ALL data
      weights = weights - lr * grad
      loss = compute_loss(X, y, weights)
      if |loss - prev_loss| < tolerance:
          break  // converged
  ```

- **Gradient for MSE loss (linear regression):**
  Loss: L = (1/n) * Σ(y_pred - y_true)²
  Gradient w.r.t. weights: ∇L = (2/n) * Xᵀ(Xw - y)
  This uses matrix math from Phase 1!

### CODE
File: include/ml_optim.h, src/optim/gd.c
```c
#ifndef ML_OPTIM_H
#define ML_OPTIM_H

#include "ml_math.h"

typedef struct {
    double lr;           // learning rate
    int    max_epochs;   // maximum iterations
    double tol;          // convergence tolerance
} GDConfig;

typedef struct {
    Vec *weights;        // final weights
    double final_loss;   // loss at convergence
    int    epochs_run;   // how many epochs it actually ran
    double *loss_history; // loss at each epoch (for plotting)
} GDResult;

// Gradient function type: given X, y, weights → compute gradient vector
typedef Vec* (*GradFn)(const Mat *X, const Vec *y, const Vec *w);
// Loss function type: given X, y, weights → compute scalar loss
typedef double (*LossFn)(const Mat *X, const Vec *y, const Vec *w);

GDResult* gd_batch(const Mat *X, const Vec *y, GradFn grad_fn, LossFn loss_fn,
                   const GDConfig *cfg);
void gd_result_free(GDResult *r);

#endif
```

Implementation of gd_batch:
1. Initialize weights to zeros (Vec of length X->cols)
2. Allocate loss_history array (max_epochs doubles)
3. For each epoch:
   a. Compute gradient: grad = grad_fn(X, y, weights)
   b. Update: weights[i] -= lr * grad[i] for all i
   c. Free the gradient vector
   d. Compute loss: loss = loss_fn(X, y, weights)
   e. Store in loss_history
   f. Check convergence: |loss - prev_loss| < tol → break
4. Package into GDResult and return

### MASTER
- [ ] Can explain gradient descent in your own words (what, why, how)
- [ ] Can explain what the learning rate does, and what happens when it's too big/small
- [ ] Hand-compute: 2D function f(x)=x², gradient = 2x, start at x=5, lr=0.1, three steps
      Step 1: x = 5 - 0.1*10 = 4.0
      Step 2: x = 4 - 0.1*8 = 3.2
      Step 3: x = 3.2 - 0.1*6.4 = 2.56
- [ ] GD on a simple MSE problem converges to reasonable weights
- [ ] loss_history is monotonically decreasing (for convex problems with good lr)
- [ ] Convergence: GD stops early when loss stops changing
- [ ] Different learning rates produce different convergence speeds
- [ ] Can write the MSE gradient formula from memory

---

## Step 4.1: Stochastic & Mini-Batch Gradient Descent

### KNOW
- **Stochastic Gradient Descent (SGD):**
  Instead of using ALL samples to compute gradient, use just ONE random sample.
  Much cheaper per step, but gradient estimate is noisy.
  Properties:
  - Extremely fast per iteration (O(n_features) instead of O(n × n_features))
  - Noisy gradient: sometimes steps in wrong direction
  - The noise actually helps escape local minima (a feature, not a bug!)
  - Converges faster in wall-clock time for large datasets
  - For one sample (x_i, y_i): grad = x_i * (x_i · w - y_i)

- **Mini-batch gradient descent:**
  Use a small batch of B samples (e.g., B = 32, 64, 128).
  Compromise between batch (all samples) and stochastic (one sample).
  Properties:
  - More stable gradient estimate than SGD (less noise)
  - Cheaper than full batch (B << n)
  - The standard choice in modern ML/DL
  - Common batch sizes: 32, 64, 128. Powers of 2 (historical hardware reasons).

- **Epochs in SGD/Mini-batch:**
  One epoch = one full pass through the training data.
  Within each epoch, shuffle the data, then process batches sequentially.
  ```
  for epoch in 1..max_epochs:
      shuffle training data
      for each batch in data:
          grad = compute_gradient(batch_X, batch_y, weights)
          weights = weights - lr * grad
  ```

- **Key differences:**
  | Property       | Batch GD      | SGD           | Mini-batch    |
  |----------------|---------------|---------------|---------------|
  | Samples/step   | All n         | 1             | B (e.g., 32)  |
  | Step cost      | O(n·d)        | O(d)          | O(B·d)        |
  | Gradient noise | None          | High          | Medium        |
  | Convergence    | Smooth        | Noisy/zigzag  | Medium        |
  | Standard?      | Small data    | Rarely alone  | Most common   |

### CODE
Add to ml_optim.h, implement in src/optim/sgd.c:
```c
typedef struct {
    double lr;
    int    max_epochs;
    int    batch_size;    // 1 = pure SGD, n = batch GD, else mini-batch
    unsigned int seed;    // for reproducible shuffling
} SGDConfig;

GDResult* gd_sgd(const Mat *X, const Vec *y, GradFn grad_fn, LossFn loss_fn,
                 const SGDConfig *cfg);
```

Implementation:
1. Create index array [0, 1, ..., n-1]
2. For each epoch:
   a. Shuffle indices using rng_shuffle with seed
   b. For each batch (chunk of batch_size indices):
      - Extract batch rows from X and y
      - Compute gradient on batch
      - Update weights
      - Free batch data
   c. Compute loss on full dataset (for tracking)
   d. Check convergence
3. Return GDResult

### MASTER
- [ ] Can explain the difference between batch, stochastic, and mini-batch GD
- [ ] SGD converges to similar weights as batch GD (maybe slightly different)
- [ ] SGD loss history is noisier than batch GD loss history
- [ ] Mini-batch with batch_size=n gives same result as batch GD
- [ ] Shuffling data each epoch matters (test ordered vs shuffled)
- [ ] Can explain why mini-batch is the practical default
- [ ] Code handles edge case: n not divisible by batch_size (last batch is smaller)

---

## Step 4.2: Learning Rate Scheduling (Optional Enhancement)

### KNOW
- **Problem:** Fixed learning rate is suboptimal.
  - Start with large lr → take big steps → get close quickly
  - Later, large lr overshoots the minimum → bounces around
  - Solution: start high, decay over time

- **Common schedules:**
  1. **Step decay:** Cut lr by factor every K epochs
     lr = initial_lr * decay_rate^(epoch / step_size)
     Example: lr=0.1, decay every 10 epochs by 0.5:
     epoch 0-9: lr=0.1, epoch 10-19: lr=0.05, epoch 20-29: lr=0.025

  2. **Exponential decay:** lr = initial_lr * e^(-decay * epoch)
     Smooth decrease. decay controls how fast.

  3. **1/t decay:** lr = initial_lr / (1 + decay * epoch)
     Classic. Slows down gradually. Has theoretical convergence guarantees.

- **In practice:** Just pick one and tune it. It's more art than science.

### CODE
```c
typedef enum {
    LR_CONSTANT,
    LR_STEP_DECAY,
    LR_EXP_DECAY,
    LR_INV_DECAY
} LRSchedule;

typedef struct {
    LRSchedule type;
    double initial_lr;
    double decay_rate;
    int    step_size;     // for step decay
} LRConfig;

double lr_get(const LRConfig *cfg, int epoch);
```

### MASTER
- [ ] lr_get returns correct values for each schedule type
- [ ] Step decay: lr drops by factor at expected intervals
- [ ] All schedules: lr is always positive
- [ ] Can explain why decaying lr helps convergence

---

## Step 4.3: Momentum & Adam (Optional, Advanced)

### KNOW
- **Momentum:**
  Instead of just using the current gradient, accumulate a "velocity" that smooths out
  the gradient updates. Like a ball rolling downhill with inertia.
  ```
  v = beta * v + (1 - beta) * gradient     // beta typically 0.9
  weights = weights - lr * v
  ```
  Benefits: accelerates in consistent gradient directions, dampens oscillations.

- **Adam (Adaptive Moment Estimation):**
  Combines momentum with per-parameter adaptive learning rates.
  Maintains two running averages:
  - m: first moment (mean of gradients) → momentum
  - v: second moment (mean of squared gradients) → adaptive lr
  ```
  m = beta1 * m + (1-beta1) * grad          // beta1 = 0.9
  v = beta2 * v + (1-beta2) * grad²         // beta2 = 0.999
  m_hat = m / (1 - beta1^t)                 // bias correction
  v_hat = v / (1 - beta2^t)
  w = w - lr * m_hat / (sqrt(v_hat) + eps)  // eps = 1e-8
  ```
  Adam is the default optimizer in most deep learning. It just works.

- **When to use what:**
  - Plain SGD: simple, educational, good baseline
  - SGD + momentum: better convergence, still simple
  - Adam: modern default, best if you don't want to tune

### CODE
```c
GDResult* gd_sgd_momentum(const Mat *X, const Vec *y, GradFn grad_fn,
                          LossFn loss_fn, const SGDConfig *cfg, double beta);

GDResult* gd_adam(const Mat *X, const Vec *y, GradFn grad_fn, LossFn loss_fn,
                  const SGDConfig *cfg, double beta1, double beta2, double eps);
```

### MASTER
- [ ] Momentum converges faster than plain SGD on same problem
- [ ] Adam converges with less lr tuning than plain SGD
- [ ] Can write the Adam update equations from memory
- [ ] Can explain what the bias correction step does (prevents early steps from being too small)

---

## PHASE 4 GRADUATION:
- [ ] Can implement batch GD, SGD, and mini-batch GD from scratch
- [ ] Can explain the tradeoffs between all three methods
- [ ] GD on a simple quadratic loss produces correct minimum
- [ ] Can plot (or print) loss curves showing convergence
- [ ] Learning rate tuning: can demonstrate divergence (too high) and slow convergence (too low)
- [ ] Can explain momentum and Adam conceptually, even if implementation is optional
- [ ] All code tested, no leaks, no sanitizer errors
