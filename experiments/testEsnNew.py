import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp

from data_generator.MackeyGlassGenerator import run
from data_generator.HenonGenerator import runHenon
from EsnModel import EsnModel
from utils.metrics import nrmse

from sys import path
path.insert(0, '../alveus/')


if __name__ == "__main__":
    # data = np.array([run(6100)]).reshape(-1, 1)
    data = np.array([runHenon(6100, dimensions=1)]).reshape(-1, 1)
    # normalising the data seems to stabilise the noise a bit

    data -= np.mean(data)
    # data -= 0.5
    # data *= 2.
    data_mean = np.mean(data, axis=0)
    split = 5100
    # adding a bias significantly improves performance
    X_train = np.hstack((np.array(data[:split-1]), np.ones_like(data[:split-1, :1])))
    y_train = np.array(data[1:split])
    X_valid = np.hstack((np.array(data[split-1:-1]), np.ones_like(data[split-1:-1, :1])))
    y_valid = np.array(data[split:])
    data_mean = np.mean(data)

    num = 1
    n_esn_outputs = []
    # compute NRMSE for each timestep
    t_nrmse = np.zeros((num, len(y_valid)))
    cov = np.zeros((len(y_valid), len(y_valid)))
    for n in range(num):
        # esn = ESN(2, 1, 500, echo_param=0.85, regulariser=1e-6)
        esn = EsnModel(2, 1, 500)
        # esn = EsnModel(2, 1, 1000,
        #                 spectral_scale=1.0, echo_param=0.85,
        #                 input_weight_scale=1.0, regulariser=1e-5)
        esn.train(X_train, y_train)

        esn_outputs = []

        # generate test-data
        esn_outputs = esn.generate(X_valid, len(y_valid))
        n_esn_outputs.append(esn_outputs)

        error = nrmse(y_valid, esn_outputs, data_mean)
        print('ESN NRMSE: %f' % error)

        for i in range(len(y_valid)):
            y_valid_slice = y_valid[:i]
            esn_outputs_slice = esn_outputs[:i]
            err = nrmse(y_valid_slice, esn_outputs_slice, data_mean)
            t_nrmse[n, i] = err

        # data_share = np.vstack((y_valid.T, esn_outputs[:, None].T)).T
        # print(np.shape(data_share))
        # print(np.std(y_valid, 0))
        # print(np.std(esn_outputs, 0))
        # print(np.shape(y_valid))
        # y_t = np.tile(y_valid, (1, np.shape(y_valid)[0]))
        # x_t = np.tile(range(len(y_valid)), (np.shape(y_valid)[0], 1)).T/len(y_valid)
        # print(np.shape(y_t))
        # e_t = np.tile(esn_outputs[:, None], (1, np.shape(y_valid)[0])).T
        # e_x = np.tile(range(len(y_valid)), (np.shape(y_valid)[0], 1))/len(y_valid)
        # cov_data = np.abs(y_t - e_t)

        # cov_data = np.cov(data_share)/(np.std(y_valid, 0)[0]*np.std(esn_outputs, 0))
        # cov_data = np.corrcoef(data_share)
        # cov_data = sp.stats.spearmanr(y_valid, esn_outputs[:, None])
        # print(np.shape(cov_data))

        data_valid_xy = np.vstack((np.array(range(len(y_valid)))[:, None].T, y_valid.T)).T
        data_esn_xy = np.vstack((np.array(range(len(esn_outputs)))[:, None].T, esn_outputs[:, None].T)).T
        print(np.shape(data_valid_xy))
        print(np.shape(data_esn_xy))

        p = 1.0
        learn_rate = 0.001
        for i in range(100):
            scale_mat = np.array([[p, 0.],[0., 1.]])
            er = np.dot(data_esn_xy, scale_mat)-data_valid_xy
            # print("err: {}".format(er**2))
            grad = np.dot(data_esn_xy.T, er)
            print("grad: {}".format(grad))
            print("p: {}".format(p))
            p -= learn_rate*grad[0, 0]

        scale_mat = np.array([[p, 0.], [0., 1.]])
        data_esn_xy_scaled = np.dot(data_esn_xy, scale_mat)

    # plot the test versus train
    f, ax = plt.subplots(figsize=(12, 12))
    ax.plot(range(len(esn_outputs)), esn_outputs, label='ESN')
    ax.plot(range(len(y_valid)), y_valid, label='True')
    ax.plot(data_esn_xy_scaled[:, 0], data_esn_xy_scaled[:, 1], label="scaled")
    # ax.scatter(esn_outputs[:, 0], esn_outputs[:, 1], label='ESN')
    # ax.scatter(y_valid[:, 0], y_valid[:, 1], label='True')
    ax.legend()

    f2, ax2 = plt.subplots(figsize=(12, 12))
    t_nrmse_m = np.mean(t_nrmse, 0)
    t_nrmse_s = np.std(t_nrmse, 0)
    ax2.plot(range(len(t_nrmse_m)), t_nrmse_m)
    ax2.fill_between(range(len(t_nrmse_s)), t_nrmse_m-t_nrmse_s, t_nrmse_m+t_nrmse_s, alpha=0.2)
    ax2.set_xlabel("timestep")
    ax2.set_ylabel("NRMSE (sum)")

    # heatmap of cov
    # f3, ax3 = plt.subplots(figsize=(12, 12))
    # # ax3.imshow(cov_data, cmap='hot', interpolation='nearest')
    # print(cov_data)
    # sns.heatmap(cov_data, ax=ax3)

    plt.show()
