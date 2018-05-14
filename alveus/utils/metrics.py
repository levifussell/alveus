import numpy as np

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
    
