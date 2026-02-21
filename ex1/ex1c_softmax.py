"""
Exercise 1c: Softmax Regression
"""

import numpy as np
import time
from scipy.optimize import minimize

from ex1_load_mnist import load_mnist
from softmax_regression_vec import softmax_regression_vec
from multi_classifier_accuracy import multi_classifier_accuracy


# Load the MNIST data for this exercise.
# train['X'] and test['X'] will contain the training and testing images.
#   Each matrix has shape (n, m) where:
#      m is the number of examples.
#      n is the number of pixels in each image.
# train['y'] and test['y'] will contain the corresponding labels (0 to 9).
binary_digits = False
num_classes = 10
train, test = load_mnist(binary_digits)

# Add row of 1s to the dataset to act as an intercept term.
train['X'] = np.vstack([np.ones((1, train['X'].shape[1])), train['X']])
test['X'] = np.vstack([np.ones((1, test['X'].shape[1])), test['X']])
train['y'] = train['y'] + 1  # make labels 1-based
test['y'] = test['y'] + 1    # make labels 1-based

# Training set info
m = train['X'].shape[1]
n = train['X'].shape[0]

# Train softmax classifier using optimizer
options = {'maxiter': 200}

# Initialize theta. We use a matrix where each column corresponds to a class,
# and each row is a classifier coefficient for that class.
# Inside the optimizer, theta will be stretched out into a long vector (theta.flatten()).
# We only use num_classes-1 columns, since the last column is always assumed 0.
theta = np.random.rand(n, num_classes - 1) * 0.001

# Call optimizer with softmax_regression_vec as the objective.
#
# TODO: Implement batch softmax regression in softmax_regression_vec.py
#       using a vectorized implementation.
#
start = time.time()


def objective(t):
    f, g = softmax_regression_vec(t, train['X'], train['y'])
    return f, g


result = minimize(objective, theta.flatten(order='F'), jac=True, method='L-BFGS-B', options=options)
theta = result.x.reshape(n, num_classes - 1, order='F')
print(f'Optimization took {time.time() - start:.6f} seconds.')

# Expand theta to include the last class (all zeros).
theta = np.hstack([theta, np.zeros((n, 1))])

# Print out training accuracy.
accuracy = multi_classifier_accuracy(theta, train['X'], train['y'])
print(f'Training accuracy: {100 * accuracy:.1f}%')

# Print out test accuracy.
accuracy = multi_classifier_accuracy(theta, test['X'], test['y'])
print(f'Test accuracy: {100 * accuracy:.1f}%')
