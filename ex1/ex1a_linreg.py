"""
Exercise 1a: Linear Regression

This exercise uses data from the UCI repository:
  Bache, K. & Lichman, M. (2013). UCI Machine Learning Repository
  http://archive.ics.uci.edu/ml
  Irvine, CA: University of California, School of Information and Computer Science.

Data created by:
  Harrison, D. and Rubinfeld, D.L.
  "Hedonic prices and the demand for clean air"
  J. Environ. Economics & Management, vol.5, 81-102, 1978.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time

from linear_regression import linear_regression
from linear_regression_vec import linear_regression_vec


# Load housing data from CSV file.
data = np.loadtxt('housing.csv', delimiter=',', skiprows=1)
data = data.T  # put examples in columns

# Include a row of 1s as an additional intercept feature.
data = np.vstack([np.ones((1, data.shape[1])), data])

# Shuffle examples.
perm = np.random.permutation(data.shape[1])
data = data[:, perm]

# Split into train and test sets.
# The last row of 'data' is the median home price.
train_X = data[:-1, :400]
train_y = data[-1, :400]

test_X = data[:-1, 400:]
test_y = data[-1, 400:]

m = train_X.shape[1]
n = train_X.shape[0]

# Initialize the coefficient vector theta to random values.
theta = np.random.rand(n)

# Run the optimizer with linear_regression as the objective.
#
# TODO: Implement the linear regression objective and gradient computations
#       in linear_regression.py
#
start = time.time()
options = {'maxiter': 200}


def objective(t):
    f, g = linear_regression(t, train_X, train_y)
    return f, g


result = minimize(objective, theta, jac=True, method='L-BFGS-B', options=options)
theta = result.x
print(f'Optimization took {time.time() - start:.6f} seconds.')

# Run optimizer with linear_regression_vec as the objective.
#
# TODO: Implement linear regression in linear_regression_vec.py
#       using NumPy's vectorization features to speed up your code.
#       Compare the running time for your linear_regression.py and
#       linear_regression_vec.py implementations.
#
# Uncomment the lines below to run your vectorized code.
# theta = np.random.rand(n)
# start = time.time()
# def objective_vec(t):
#     f, g = linear_regression_vec(t, train_X, train_y)
#     return f, g
# result = minimize(objective_vec, theta, jac=True, method='L-BFGS-B', options=options)
# theta = result.x
# print(f'Optimization took {time.time() - start:.6f} seconds.')

# Plot predicted prices and actual prices from training set.
actual_prices = train_y
predicted_prices = theta @ train_X

# Print out root-mean-squared (RMS) training error.
train_rms = np.sqrt(np.mean((predicted_prices - actual_prices) ** 2))
print(f'RMS training error: {train_rms:.6f}')

# Print out test RMS error.
actual_prices = test_y
predicted_prices = theta @ test_X
test_rms = np.sqrt(np.mean((predicted_prices - actual_prices) ** 2))
print(f'RMS testing error: {test_rms:.6f}')

# Plot predictions on test data.
plot_prices = True
if plot_prices:
    sorted_idx = np.argsort(actual_prices)
    actual_prices = actual_prices[sorted_idx]
    predicted_prices = predicted_prices[sorted_idx]

    plt.plot(actual_prices, 'rx', label='Actual Price')
    plt.plot(predicted_prices, 'bx', label='Predicted Price')
    plt.legend()
    plt.xlabel('House #')
    plt.ylabel('House price ($1000s)')
    plt.show()
