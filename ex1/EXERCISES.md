# Exercise 1 — Deep Learning Exercises (Python)

This folder contains the Python conversion of the Exercise 1 MATLAB files. Below is a summary of every file and what you need to complete in each.

---

## Files to Complete (YOUR CODE HERE)

These are the files where you need to fill in the `### YOUR CODE HERE ###` sections.

### 1. `linear_regression.py`
- **What to implement:** Compute the linear regression objective function and gradient **using a loop** over examples.
- **TODO 1:** Loop over examples in `X` to compute the objective value `f`.
- **TODO 2:** Loop over examples in `X` to compute the gradient `g`.

### 2. `linear_regression_vec.py`
- **What to implement:** Compute the linear regression objective function and gradient **using vectorized NumPy code** (no loops).
- **TODO:** Write vectorized expressions for `f` and `g`.

### 3. `logistic_regression.py`
- **What to implement:** Compute the logistic regression objective function and gradient **using a loop** over examples.
- **TODO 1:** Loop over the dataset to compute the objective value `f`.
- **TODO 2:** Loop over the dataset to compute the gradient `g`.

### 4. `logistic_regression_vec.py`
- **What to implement:** Compute the logistic regression objective function and gradient **using vectorized NumPy code** (no loops).
- **TODO:** Write vectorized expressions for `f` and `g`.

### 5. `softmax_regression_vec.py`
- **What to implement:** Compute the softmax regression objective function and gradient **using vectorized NumPy code**.
- **TODO:** Write vectorized expressions for `f` and `g`. Remember to flatten `g` back into a vector before returning.

---

## Driver / Runner Scripts

These scripts load data, call the functions above, run optimisation, and print results. You do **not** need to edit these (unless uncommenting the vectorized sections).

| File | Description |
|------|-------------|
| `ex1a_linreg.py` | Loads the Boston housing dataset, runs linear regression, plots predictions. Uncomment the vectorized section after completing `linear_regression_vec.py`. |
| `ex1b_logreg.py` | Loads binary MNIST (digits 0 & 1), runs logistic regression, prints accuracy. Uncomment the vectorized section after completing `logistic_regression_vec.py`. |
| `ex1c_softmax.py` | Loads full MNIST (digits 0–9), runs softmax regression, prints accuracy. |

---

## Helper / Utility Files

These files are already complete. No edits needed.

| File | Description |
|------|-------------|
| `sigmoid.py` | Implements the sigmoid function $\sigma(a) = \frac{1}{1 + e^{-a}}$. |
| `grad_check.py` | Numerically checks the gradient of a function using finite differences. Useful for verifying your implementations. |
| `binary_classifier_accuracy.py` | Computes accuracy for binary logistic regression. |
| `multi_classifier_accuracy.py` | Computes accuracy for multi-class (softmax) classification. |
| `ex1_load_mnist.py` | Loads and preprocesses the MNIST dataset using `sklearn.datasets.fetch_openml`. |

---

## Dependencies

```
numpy
scipy
matplotlib
scikit-learn
```

Install with:

```bash
pip install numpy scipy matplotlib scikit-learn
```

---

## How to Run

```bash
# Exercise 1a — Linear Regression
python ex1a_linreg.py

# Exercise 1b — Logistic Regression
python ex1b_logreg.py

# Exercise 1c — Softmax Regression
python ex1c_softmax.py
```

> **Note:** `ex1a_linreg.py` expects a `housing.data` file in the same directory (the Boston Housing dataset). Make sure it is present before running.
