import numpy as np
import matplotlib.pyplot as plt

from DataGenerator.MackeyGlassGenerator import run
from src.EsnModel import EsnModel
from Helper.utils import nrmse

if __name__ == "__main__":
    data = np.array([run(6100)]).reshape(-1, 1)
    # normalising the data seems to stabilise the noise a bit
    data -= np.mean(data)
    data_mean = np.mean(data, axis=0)
    split = 5100
    # adding a bias significantly improves performance
    X_train = np.hstack((np.array(data[:split-1]), np.ones_like(data[:split-1])))
    y_train = np.array(data[1:split])
    X_valid = np.hstack((np.array(data[split-1:-1]), np.ones_like(data[split-1:-1])))
    y_valid = np.array(data[split:])
    data_mean = np.mean(data)

    # esn = ESN(2, 1, 500, echo_param=0.85, regulariser=1e-6)
    esn = EsnModel(2, 1, 500)
    esn.train(X_train, y_train)

    esn_outputs = []

    # generate test-data
    esn_outputs = esn.generate(X_valid[0], len(y_valid))

    error = nrmse(y_valid, esn_outputs, data_mean)
    print('ESN NRMSE: %f' % error)
    
    # plot the test versus train
    f, ax = plt.subplots(figsize=(12, 12))
    ax.plot(range(len(esn_outputs)), esn_outputs, label='ESN')
    ax.plot(range(len(y_valid)), y_valid, label='True')
    plt.legend()
    plt.show()