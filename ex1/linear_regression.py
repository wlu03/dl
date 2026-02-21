import numpy as np

def linear_regression(theta, X, y):
    """
    Arguments:
        theta - A vector containing the parameter values to optimize.
        X     - The examples stored in a matrix.
                X[i, j] is the i'th coordinate of the j'th example.
        y     - The target value for each example. y[j] is the target for example j.

    Returns:
        f - The objective function value.
            m is the number of val
        g - The gradient of the objective with respect to theta.
    """
    
    # theta = (n,) : vector of n features
        # one weight per feature
        # for example, if theta = [w1, w2], then theta.shape = (2,)
    # X shape (n, m)
        # n features, m training examples
    # y has a shape (m,) for the correct output of m examples
    
    m = X.shape[1] # number of example
    n = X.shape[0] # number of features

    f = 0.0
    g = np.zeros_like(theta)

    # compute object function and gradient
    
    for j in range(m): # loop over m examples
        x_j = X[:, j] # vector of j-th object
        h = theta @ x_j # prediction (scalar)
        error = h - y[j] # residual
        f += 0.5 * error ** 2 # squared error
        g += error * x_j

    return f, g
