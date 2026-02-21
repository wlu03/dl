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

    #
    # TODO: Compute the objective function by looping over the dataset and summing
    #       up the objective values for each example. Store the result in 'f'.
    #
    # TODO: Compute the gradient of the objective by looping over the dataset and summing
    #       up the gradients (df/dtheta) for each example. Store the result in 'g'.
    #
    ### YOUR CODE HERE ###

    return f, g
