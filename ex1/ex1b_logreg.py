"""
Exercise 1b: Logistic Regression
"""

import numpy as np
import time
from scipy.optimize import minimize

from ex1_load_mnist import load_mnist
from logistic_regression import logistic_regression
from logistic_regression_vec import logistic_regression_vec
from binary_classifier_accuracy import binary_classifier_accuracy


# Load the MNIST data for this exercise.
# train['X'] and test['X'] will contain the training and testing images.
#   Each matrix has shape (n, m) where:
#      m is the number of examples.
#      n is the number of pixels in each image.
# train['y'] and test['y'] will contain the corresponding labels (0 or 1).
binary_digits = True
train, test = load_mnist(binary_digits)

# Add row of 1s to the dataset to act as an intercept term.
train['X'] = np.vstack([np.ones((1, train['X'].shape[1])), train['X']])
test['X'] = np.vstack([np.ones((1, test['X'].shape[1])), test['X']])

# Training set dimensions
m = train['X'].shape[1]
n = train['X'].shape[0]

# Train logistic regression classifier using optimizer
options = {'maxiter': 100}

# First, we initialize theta to some small random values.
theta = np.random.rand(n) * 0.001

# Call optimizer with logistic_regression as the objective function.
#
# TODO: Implement batch logistic regression in the logistic_regression.py file!
#
start = time.time()


def objective(t):
    f, g = logistic_regression(t, train['X'], train['y'])
    return f, g


result = minimize(objective, theta, jac=True, method='L-BFGS-B', options=options)
theta = result.x
print(f'Optimization took {time.time() - start:.6f} seconds.')

# Now, call optimizer again with logistic_regression_vec as the objective.
#
# TODO: Implement batch logistic regression in logistic_regression_vec.py using
#       NumPy's vectorization features to speed up your code. Compare the running
#       time for your logistic_regression.py and logistic_regression_vec.py implementations.
#
# Uncomment the lines below to run your vectorized code.
# theta = np.random.rand(n) * 0.001
# start = time.time()
# def objective_vec(t):
#     f, g = logistic_regression_vec(t, train['X'], train['y'])
#     return f, g
# result = minimize(objective_vec, theta, jac=True, method='L-BFGS-B', options=options)
# theta = result.x
# print(f'Optimization took {time.time() - start:.6f} seconds.')

# Print out training accuracy.
accuracy = binary_classifier_accuracy(theta, train['X'], train['y'])
print(f'Training accuracy: {100 * accuracy:.1f}%')

# Print out accuracy on the test set.
accuracy = binary_classifier_accuracy(theta, test['X'], test['y'])
print(f'Test accuracy: {100 * accuracy:.1f}%')
