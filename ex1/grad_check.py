import numpy as np


def grad_check(fun, theta0, num_checks, *args):
    """
    Numerically check the gradient of a function.

    Arguments:
        fun        - A function that takes (theta, *args) and returns (f, g).
        theta0     - The parameter vector at which to check the gradient.
        num_checks - The number of random gradient components to check.
        *args      - Additional arguments passed to fun.

    Returns:
        average_error - The average error between the numerical and analytical gradients.
    """
    delta = 1e-3
    sum_error = 0.0

    print(f"{'Iter':>5s}  {'i':>6s}  {'err':>15s}  {'g_est':>15s}  {'g':>15s}  {'f':>15s}")

    for i in range(num_checks):
        T = theta0.copy()
        j = np.random.randint(len(T))

        T0 = T.copy()
        T0[j] -= delta
        T1 = T.copy()
        T1[j] += delta

        f, g = fun(T, *args)
        f0, _ = fun(T0, *args)
        f1, _ = fun(T1, *args)

        g_est = (f1 - f0) / (2 * delta)
        error = abs(g[j] - g_est)

        print(f"{i+1:5d}  {j:6d}  {error:15g}  {g[j]:15f}  {g_est:15f}  {f:15f}")

        sum_error += error

    average_error = sum_error / num_checks
    return average_error
