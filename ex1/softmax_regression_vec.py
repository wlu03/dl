import numpy as np


def softmax_regression_vec(theta, X, y):
    """
    Arguments:
        theta - A vector containing the parameter values to optimize.
                In the optimizer, theta is reshaped to a long vector. So we need to
                resize it to an n-by-(num_classes-1) matrix.
                Recall that we assume theta[:, num_classes-1] = 0.
        X     - The examples stored in a matrix.
                X[i, j] is the i'th coordinate of the j'th example.
        y     - The label for each example. y[j] is the j'th example's label.

    Returns:
        f - The objective function value.
        g - The gradient (flattened to a vector).
    """
    m = X.shape[1]
    n = X.shape[0]

    # theta is a vector; need to reshape to n x num_classes.
    theta = theta.reshape(n, -1, order='F')
    num_classes = theta.shape[1] + 1

    # Initialize objective value and gradient.
    f = 0.0
    g = np.zeros_like(theta)

    #
    # TODO: Compute the softmax objective function and gradient using vectorized code.
    #       Store the objective function value in 'f', and the gradient in 'g'.
    #       Before returning g, make sure you flatten it back into a vector with g.flatten(order='F').
    #
    ### YOUR CODE HERE ###

    g = g.flatten(order='F')  # make gradient a vector for the optimizer

    return f, g
