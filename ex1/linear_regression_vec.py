import numpy as np


def linear_regression_vec(theta, X, y):
    """
    Arguments:
        theta - A vector containing the parameter values to optimize.
        X     - The examples stored in a matrix.
                X[i, j] is the i'th coordinate of the j'th example.
        y     - The target value for each example. y[j] is the target for example j.

    Returns:
        f - The objective function value.
        g - The gradient of the objective with respect to theta.
    """
    m = X.shape[1]

    # Initialize objective value and gradient.
    f = 0.0
    g = np.zeros_like(theta)

    #
    # TODO: Compute the linear regression objective function and gradient
    #       using vectorized code. (It will be just a few lines of code!)
    #       Store the objective function value in 'f', and the gradient in 'g'.
    #
    ### YOUR CODE HERE ###

    return f, g
