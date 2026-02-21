import numpy as np


def load_mnist(binary_digits):
    """
    Load the MNIST dataset.

    Uses sklearn.datasets.fetch_openml to fetch MNIST data (equivalent to
    loadMNISTImages/loadMNISTLabels in the MATLAB version).

    Arguments:
        binary_digits - If True, only keep digits 0 and 1.

    Returns:
        train - dict with keys 'X' (n x m_train) and 'y' (m_train,)
        test  - dict with keys 'X' (n x m_test)  and 'y' (m_test,)
    """
    from sklearn.datasets import fetch_openml

    # Fetch MNIST (70,000 images: 60k train + 10k test)
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X_all = mnist.data.T / 255.0        # shape (784, 70000), pixel values in [0, 1]
    y_all = mnist.target.astype(int)     # shape (70000,)

    # Standard split: first 60k = train, last 10k = test
    X_train, y_train = X_all[:, :60000], y_all[:60000]
    X_test, y_test = X_all[:, 60000:], y_all[60000:]

    if binary_digits:
        # Take only the 0 and 1 digits
        train_mask = (y_train == 0) | (y_train == 1)
        X_train, y_train = X_train[:, train_mask], y_train[train_mask]

        test_mask = (y_test == 0) | (y_test == 1)
        X_test, y_test = X_test[:, test_mask], y_test[test_mask]

    # Randomly shuffle training data
    I = np.random.permutation(len(y_train))
    X_train, y_train = X_train[:, I], y_train[I]

    # Standardize so each pixel has roughly zero mean and unit variance
    m = X_train.mean(axis=1, keepdims=True)
    s = X_train.std(axis=1, keepdims=True)
    X_train = (X_train - m) / (s + 0.1)

    # Randomly shuffle test data
    I = np.random.permutation(len(y_test))
    X_test, y_test = X_test[:, I], y_test[I]

    # Standardize test data using training mean and scale
    X_test = (X_test - m) / (s + 0.1)

    train = {'X': X_train, 'y': y_train}
    test = {'X': X_test, 'y': y_test}

    return train, test
