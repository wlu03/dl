import numpy as np

from sigmoid import sigmoid


def binary_classifier_accuracy(theta, X, y):
    """Compute accuracy for a binary classifier using logistic regression."""
    predictions = (sigmoid(theta.T @ X) > 0.5).astype(int)
    correct = np.sum(y == predictions)
    accuracy = correct / len(y)
    return accuracy
