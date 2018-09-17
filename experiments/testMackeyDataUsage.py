import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp

from sys import path
path.insert(0, '/home/thais/dev/alveus/')  # needed to import alveus
path.insert(0, '/home/oem/Documents/Code/2018/Projects/ESN/alveus/')  # needed to import alveus
from alveus.data.generators.MackeyGlassGenerator import run
#from alveus.data.generators.HenonGenerator import runHenon
from alveus.models.DEsnModel import DEsnModel
from alveus.utils.metrics import nrmse

import pickle as pkl

def init_data(warm_up_length, data_size, train_size):
    #warm_up_length = 300*num_reservoirs
    #data_size = 21000 #6100
    #train_size = 20000 #5100
    data = np.array([run(warm_up_length + data_size)]).reshape(-1, 1)
    #data = np.array([runHenon(warm_up_length + data_size, dimensions=1)]).reshape(-1, 1)
    # normalising the data seems to stabilise the noise a bit

    #data -= np.mean(data)
    #data /= np.std(data)
    # data -= 0.5
    # data *= 2.
    data_mean = np.mean(data, axis=0)
    split = warm_up_length + train_size
    # adding a bias significantly improves performance
    X_train = np.hstack((np.array(data[:split-1]), np.ones_like(data[:split-1, :1])))
    y_train = np.array(data[1:split])
    X_valid = np.hstack((np.array(data[split-1:-1]), np.ones_like(data[split-1:-1, :1])))
    y_valid = np.array(data[split:])
    data_mean = np.mean(data)

    #num = 1
    #n_esn_outputs = []
    # compute NRMSE for each timestep


    #reps = 1
    #X_train = np.tile(X_train, (reps,1))
    #y_train = np.tile(y_train, (reps,1))

    return X_train, y_train, X_valid, y_valid, data_mean

def main():

    num_reservoirs = 10
    #data_size = 21000
    #train_size = 20000
    #data_size = 11000
    #train_size = 10000
    warm_up_length = 300*num_reservoirs

    data_size_query = range(4000, 26000, 1000) #[6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000, 22000, 24000, 26000]
    #data_size_query = [21000]
    train_size_query = range(3000, 25000, 1000) #[5000, 7000, 9000, 11000, 13000, 15000, 17000, 19000, 21000, 23000, 25000]
    #train_size_query = [20000]

    for (d,t) in zip(data_size_query, train_size_query):
        X_train, y_train, X_valid, y_valid, data_mean = init_data(warm_up_length, d, t)

        mse_train_to_test = np.sum(np.abs(X_train[-1000:] - X_valid[:1000]))
        print("D: {}, T: {}, MSE: {}".format(d, t, mse_train_to_test))

    num = 1

if __name__ == "__main__":
    main()
