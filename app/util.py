import numpy as np
from sklearn.model_selection import train_test_split


def mnist_preprocessing(data: np.ndarray,
                        TEST_SIZE: float = 0.2):
    """
        return train_test_split result
    """

    X, y = data[:, 1:], data[:, 0]
    X_train, X_valid, y_train, y_valid = \
        train_test_split(X, y, test_size=TEST_SIZE)

    return X_train, X_valid, y_train, y_valid
