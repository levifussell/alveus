import numpy as np
from sklearn.metrics.pairwise import rbf_kernel


def mse(y_true, y_pred):
    """
    Calculates the mean square error (MSE) of y_true and y_pred.
    """
    return np.mean(np.square(y_true - y_pred))


# TODO: add functionality for not having to pass MEAN_OF_DATA every time and
# just calculate it if it is not passed in.
def nrmse(y_true, y_pred, MEAN_OF_DATA):
    """
    Calculates the normalized root mean square error of y_true and y_pred
    where MEAN_OF_DATA is the mean of y_pred.
    """
    y_true = y_true.squeeze()
    y_pred = y_pred.squeeze()

    std = np.sum(np.square(y_true - MEAN_OF_DATA))
    errors = np.sum(np.square(y_true - y_pred))

    return np.sqrt(errors / std)


def gaussian_kernel(X, Y, sigma):
    return np.sum(rbf_kernel(X, Y, gamma=sigma), axis=0)


def MMD(X, Y, kernel=gaussian_kernel, sigma=[None, None, None]):
    xx = kernel(X, X, sigma[0])
    xy = kernel(X, Y, sigma[1])
    yy = kernel(Y, Y, sigma[2])
    mmd = np.mean(xx) - 2*np.mean(xy) + np.mean(yy)
    return np.sqrt(mmd)
