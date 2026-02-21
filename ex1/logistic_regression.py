import numpy as np

from sigmoid import sigmoid


def logistic_regression(theta, X, y):
    """
    Arguments:
        theta - A column vector containing the parameter values to optimize.
        X     - The examples stored in a matrix.
                X[i, j] is the i'th coordinate of the j'th example.
        y     - The label for each example. y[j] is the j'th example's label.

    Returns:
        f - The objective function value.
        g - The gradient of the objective with respect to theta.
    """
    m = X.shape[1]

    # Initialize objective value and gradient.
    f = 0.0
    g = np.zeros_like(theta)

    for j in range(m): # loop over m values
        xj = X[:, j]
        yj = y[j]
        z = np.dot(theta, xj)    # scalar dot product
        h = sigmoid(z)
        
        # objective function 
        f += -yj * np.log(h) - (1 - yj) * np.log(1 - h)
        g += (h - yj) * xj

    f /= m
    g /= m
    
    return f, g
