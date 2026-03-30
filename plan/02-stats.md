# Phase 2: Statistics, Activation Functions & Random Numbers

## Why This Phase Matters
Statistics summarize data. Activation functions add nonlinearity to neural networks
(without them, stacking layers = one big matrix multiply = still linear, useless).
RNG is needed for weight initialization, data shuffling, and stochastic algorithms.

## Dependencies: Phase 1 (Vec, Mat, all operations)

---

## Step 2.0: Descriptive Statistics

### KNOW
- **Mean (average):** μ = (1/n) Σᵢ xᵢ
  - The "center" of the data
  - Sensitive to outliers (one huge value pulls it way up)

- **Variance:** σ² = (1/n) Σᵢ (xᵢ - μ)²
  - Measures how "spread out" the data is
  - High variance = data points far from mean
  - Low variance = data points clustered near mean
  - NOTE: population variance divides by n, sample variance by (n-1).
    For ML, population (n) is more common. Use n.

- **Standard deviation:** σ = √(variance)
  - Same unit as the data (variance is in squared units)
  - 68% of data within ±1σ of mean (for normal distributions)

- **Covariance:** cov(x,y) = (1/n) Σᵢ (xᵢ - μₓ)(yᵢ - μᵧ)
  - Measures how two variables move TOGETHER
  - Positive: x goes up → y goes up
  - Negative: x goes up → y goes down
  - Zero: no linear relationship
  - cov(x,x) = var(x) (covariance of x with itself = variance)

- **Correlation (Pearson):** r = cov(x,y) / (σₓ · σᵧ)
  - Normalized covariance: always between -1 and +1
  - +1 = perfect positive linear relationship
  - -1 = perfect negative linear relationship
  - 0 = no linear relationship (could still be nonlinear!)

- **Min, Max, Argmin, Argmax:**
  - min/max: the smallest/largest value
  - argmin/argmax: the INDEX of the smallest/largest value
  - argmax is used everywhere: predicted class = argmax of output probabilities

### CODE
File: include/ml_stats.h, src/math/stats.c
```c
double stats_mean(const Vec *v);
double stats_variance(const Vec *v);           // population variance (÷n)
double stats_std(const Vec *v);                // √variance
double stats_covariance(const Vec *a, const Vec *b);
double stats_correlation(const Vec *a, const Vec *b);
double stats_min(const Vec *v);
double stats_max(const Vec *v);
int    stats_argmin(const Vec *v);             // returns INDEX
int    stats_argmax(const Vec *v);             // returns INDEX
```

### MASTER
- [ ] mean([1,2,3,4,5]) = 3.0
- [ ] variance([1,2,3,4,5]) = 2.0 — can compute by hand
- [ ] std([1,2,3,4,5]) = √2 ≈ 1.414
- [ ] covariance of identical vectors = variance
- [ ] correlation of [1,2,3] with [2,4,6] = 1.0 (perfect positive)
- [ ] correlation of [1,2,3] with [6,4,2] = -1.0 (perfect negative)
- [ ] argmax([3,1,4,1,5]) = 4 (index of 5)
- [ ] Can explain variance vs standard deviation in own words
- [ ] Can explain what covariance measures (direction of co-movement)

---

## Step 2.1: Activation Functions & Their Derivatives

### KNOW
- **Why activation functions exist:**
  - A neural net layer does: y = Wx + b (linear transformation)
  - Stacking linear layers: y = W₂(W₁x + b₁) + b₂ = (W₂W₁)x + (W₂b₁+b₂)
  - This is STILL just one linear transformation! Multiple layers = useless without nonlinearity.
  - Activation function adds nonlinearity BETWEEN layers: y = f(Wx + b)
  - This is what gives neural networks the power to learn complex patterns.

- **Sigmoid: σ(x) = 1 / (1 + e⁻ˣ)**
  - Output range: (0, 1) — good for probabilities
  - σ(0) = 0.5, σ(large positive) ≈ 1, σ(large negative) ≈ 0
  - S-shaped curve
  - Derivative: σ'(x) = σ(x) · (1 - σ(x))
  - Problem: vanishing gradient — for very large/small x, derivative ≈ 0,
    gradients stop flowing → deep networks learn slowly
  - Used in: logistic regression output, LSTM gates

- **ReLU: f(x) = max(0, x)**
  - Output: x if x > 0, else 0 — clips negatives to zero
  - Derivative: f'(x) = 1 if x > 0, 0 if x < 0 (undefined at x=0, use 0)
  - Most popular activation in modern deep learning
  - Advantage: no vanishing gradient for positive values, very fast to compute
  - Problem: "dying ReLU" — if input is always negative, gradient is always 0, neuron never learns

- **Tanh: f(x) = (eˣ - e⁻ˣ) / (eˣ + e⁻ˣ)**
  - Output range: (-1, 1) — zero-centered (unlike sigmoid)
  - tanh(0) = 0
  - Derivative: f'(x) = 1 - tanh²(x)
  - Better than sigmoid in hidden layers (zero-centered)
  - Still has vanishing gradient problem

- **Softmax: softmax(xᵢ) = eˣⁱ / Σⱼ eˣʲ**
  - Takes a VECTOR, outputs a VECTOR of same length
  - All outputs are positive and sum to 1.0 → probability distribution
  - Used for multi-class classification output layer
  - Numerical trick: subtract max(x) before exp to avoid overflow
    softmax(x) = softmax(x - max(x)) — mathematically identical, numerically stable

- **Chain rule (THIS IS HOW BACKPROPAGATION WORKS):**
  - If y = f(g(x)), then dy/dx = f'(g(x)) · g'(x)
  - In neural nets: loss → activation → linear → previous layer
  - Each step multiplies by the local derivative
  - This is why you need derivatives of every activation function

### CODE
File: include/ml_functions.h, src/math/functions.c
```c
// Scalar activations
double fn_sigmoid(double x);
double fn_sigmoid_deriv(double x);     // takes x, not sigmoid(x)
double fn_relu(double x);
double fn_relu_deriv(double x);
double fn_tanh_act(double x);          // tanh (avoid name collision with math.h tanh)
double fn_tanh_deriv(double x);

// Vector activations
Vec* fn_softmax(const Vec *x);         // returns NEW probability vector

// Loss functions (operate on vectors)
double fn_mse_loss(const Vec *y_true, const Vec *y_pred);    // (1/n)Σ(yᵢ-ŷᵢ)²
Vec*   fn_mse_grad(const Vec *y_true, const Vec *y_pred);    // (2/n)(ŷ-y)
double fn_cross_entropy_loss(const Vec *y_true, const Vec *y_pred);  // -Σ yᵢ log(ŷᵢ)
```

### MASTER
- [ ] sigmoid(0) = 0.5
- [ ] sigmoid(100) ≈ 1.0, sigmoid(-100) ≈ 0.0
- [ ] sigmoid_deriv: verify with finite differences: (sigmoid(x+h) - sigmoid(x-h)) / (2h) ≈ sigmoid_deriv(x)
- [ ] relu(5) = 5, relu(-3) = 0
- [ ] softmax([1,2,3]): all positive, sums to 1.0
- [ ] softmax([1000, 1001, 1002]): doesn't overflow (max-subtraction trick)
- [ ] MSE of identical vectors = 0.0
- [ ] Can explain: why neural nets need activation functions (2-3 sentences)
- [ ] Can explain: the vanishing gradient problem with sigmoid
- [ ] Can explain the chain rule and how it relates to backpropagation
- [ ] Can verify any derivative numerically using finite differences

---

## Step 2.2: Random Number Utilities

### KNOW
- **Why randomness in ML:**
  - Weight initialization: neurons start with random weights (if all same → all learn the same thing)
  - Data shuffling: shuffle training data each epoch to prevent learning order-dependent patterns
  - Stochastic gradient descent: randomly sample mini-batches
  - Train/test split: randomly divide data

- **Pseudorandom number generator (PRNG):**
  - Not truly random — deterministic sequence from a seed
  - Same seed → same sequence → reproducible experiments
  - srand(seed) sets the seed, rand() generates next number
  - rand() returns int in [0, RAND_MAX]. To get double in [0,1): (double)rand() / (RAND_MAX + 1.0)

- **Gaussian (normal) random — Box-Muller transform:**
  - You need: normally distributed random numbers (bell curve, mean=0, std=1)
  - rand() only gives UNIFORM distribution
  - Box-Muller: take two uniform randoms u₁, u₂ in (0,1):
    z₀ = √(-2 ln u₁) · cos(2π u₂)
    z₁ = √(-2 ln u₁) · sin(2π u₂)
  - z₀ and z₁ are both standard normal (mean=0, std=1)
  - For mean=μ, std=σ: result = μ + σ·z

- **Fisher-Yates shuffle (unbiased random permutation):**
  - For i from n-1 down to 1:
    - j = random integer in [0, i]
    - swap array[i] and array[j]
  - Every permutation is equally likely (unbiased)
  - Used for: shuffling training data indices, random train/test split

- **Xavier initialization: W ~ N(0, 1/n_in)**
  - Draw weights from normal distribution with std = √(1/n_in)
  - n_in = number of inputs to the layer
  - Keeps signal magnitude roughly constant across layers
  - Used with sigmoid/tanh activations

- **He initialization: W ~ N(0, 2/n_in)**
  - std = √(2/n_in)
  - Designed for ReLU activations (ReLU kills half the signal)

### CODE
File: include/ml_random.h, src/math/random.c
```c
void   rng_seed(unsigned int seed);
double rng_uniform(void);              // [0, 1)
double rng_gaussian(double mean, double std);  // Box-Muller
void   rng_shuffle_indices(int *arr, int n);   // Fisher-Yates
Vec*   rng_vec_gaussian(int n, double mean, double std);  // vector of random values
Mat*   rng_mat_xavier(int rows, int cols);     // Xavier init
Mat*   rng_mat_he(int rows, int cols);         // He init
```

### MASTER
- [ ] rng_seed(42) + generate → same sequence every time (reproducible)
- [ ] 10000 uniform samples: mean ≈ 0.5 (within 0.02)
- [ ] 10000 gaussian(0,1) samples: mean ≈ 0 (within 0.05), std ≈ 1 (within 0.05)
- [ ] shuffle [0,1,2,3,4]: all elements still present, order changed
- [ ] Can explain Box-Muller in own words: "transforms uniform → gaussian"
- [ ] Can explain why Fisher-Yates is unbiased
- [ ] Can explain Xavier vs He: when to use each (sigmoid/tanh vs ReLU)
- [ ] Can explain why random weight init matters (what if all weights = 0?)

---

## PHASE 2 GRADUATION:
- [ ] Can compute mean/variance/std by hand
- [ ] Can draw sigmoid, ReLU, tanh curves from memory
- [ ] Can write sigmoid and its derivative from memory
- [ ] Can explain why activations are needed, vanishing gradient, chain rule
- [ ] Can explain Box-Muller conceptually
- [ ] All Phase 2 tests pass, zero sanitizer errors
