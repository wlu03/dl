import numpy as np


def multi_classifier_accuracy(theta, X, y):
    """Compute accuracy for a multi-class classifier (softmax)."""
    labels = np.argmax(theta.T @ X, axis=0) + 1  # 1-based labels
    correct = np.sum(y == labels)
    accuracy = correct / len(y)
    return accuracy
