import numpy as np
import matplotlib.pyplot as plt

from MackeyGlass.MackeyGlassGenerator import run
from ESN.ESN import ESN
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

    esn = ESN(2, 1, 500, echo_param=0.85, regulariser=1e-6)
    esn.initialize_input_weights(scale=1.0)
    esn.initialize_reservoir_weights(spectral_scale=1.25)
    esn.train(X_train, y_train)

    esn_outputs = []

    # GENERATIVE =================================================
    u_n_ESN = np.array(X_valid[0])
    for _ in range(len(y_valid)):
        u_n_ESN = np.array(esn.forward(u_n_ESN))
        esn_outputs.append(u_n_ESN)
        u_n_ESN = np.hstack((u_n_ESN, 1))

    esn_outputs = np.array(esn_outputs).squeeze()

    error = nrmse(y_valid, esn_outputs, data_mean)
    print('ESN NRMSE: %f' % error)
    
    f, ax = plt.subplots(figsize=(12, 12))
    #ax.plot(range(len(esn_outputs)), esn_outputs, label='ESN')
    ax.plot(range(len(esn_outputs)), esn_outputs, label='ESN')
    ax.plot(range(len(y_valid)), y_valid, label='True')
    plt.legend()
    plt.show()