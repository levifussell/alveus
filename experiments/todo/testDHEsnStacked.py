import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp

from sys import path
path.insert(0, '/home/thais/dev/alveus/')  # needed to import alveus
path.insert(0, '/home/oem/Documents/Code/2018/Projects/ESN/alveus/')  # needed to import alveus
from alveus.data.generators.MackeyGlassGenerator import run
#from alveus.data.generators.HenonGenerator import runHenon
from alveus.models.DMEsnModel import DMEsnModel
from alveus.layers.LayerEncoder import LayerPcaEncoder
from alveus.utils.metrics import nrmse

import pickle as pkl

if __name__ == "__main__":

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

    num_reservoirs = 8 #10
    #data_size = 21000
    #train_size = 20000
    #data_size = 11000
    #train_size = 10000
    warm_up_length = 300*num_reservoirs

    num = 1
    repeats = 1
    n_esn_outputs = []
    t_nrmse = np.zeros((num, 1000))
    cov = np.zeros((1000, 1000))
    _nrmse = np.zeros((num, repeats))

    #data_size_query = [6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000, 22000, 24000, 26000]
    data_size_query = [5000]
    #train_size_query = [5000, 7000, 9000, 11000, 13000, 15000, 17000, 19000, 21000, 23000, 25000]
    train_size_query = [4000]

    num_reservoirs =4 
    n_t = num_reservoirs/2

    model_params = {
                    "input_size": 2,
                    "output_size": 1,
                    "num_reservoirs": num_reservoirs,
                    "reservoir_sizes": [200, 200]*n_t, #np.linspace(200, 400, num_reservoirs, dtype=int).tolist(),
                    "echo_params": [0.4, 0.4]*n_t, #np.linspace(0.5, 0.1, num_reservoirs).tolist(),
                    "spectral_scales": [0.6, 0.6]*n_t, #np.linspace(0.4, 1.2, num_reservoirs).tolist(),
                    "input_weight_scales": [0.1]*num_reservoirs, #np.linspace(0.5, 0.5, num_reservoirs).tolist(),
                    "sparsities": [0.1]*num_reservoirs, #np.linspace(1.0, 1.0, num_reservoirs).tolist(), #[0.1]*num_reservoirs,
                    #"sparsities": np.linspace(1.0, 1.0, num_reservoirs).tolist(), #[0.1]*num_reservoirs,
                    "res_initialise_strategies": ['uniform']*num_reservoirs,
                    "encoder_layers": [LayerPcaEncoder]*(num_reservoirs-1),
                    #"encoder_dimensions": np.linspace(30, 80, num_reservoirs-1, dtype=int).tolist(),
                    "encoder_dimensions": [80]*(num_reservoirs-1),
                    "regulariser": 1e-2
                   }

    # grid search data


    for n in range(num):
        for r in range(repeats):
            X_train, y_train, X_valid, y_valid, data_mean = init_data(warm_up_length, data_size_query[n], train_size_query[n])
            # esn = ESN(2, 1, 500, echo_param=0.85, regulariser=1e-6)
            #esn = EsnModel(input_size=2, output_size=1, reservoir_size=1000)
            #esn = DEsnModel(input_size=2, output_size=1, num_reservoirs=2)
            esn = DMEsnModel(input_size=model_params["input_size"], output_size=model_params["output_size"], 
                            num_reservoirs=model_params["num_reservoirs"],
                            reservoir_sizes=model_params["reservoir_sizes"],
                            echo_params=model_params["echo_params"],
                            spectral_scales=model_params["spectral_scales"],
                            input_weight_scales=model_params["input_weight_scales"],
                            sparsities=model_params["sparsities"],
                            res_initialise_strategies=model_params["res_initialise_strategies"],
                            encoder_layers=model_params["encoder_layers"],
                            encoder_dimensions=model_params["encoder_dimensions"],
                            regulariser=model_params["regulariser"],
                            reuse_encoder=False)
            # esn.layers[0].drop_probability = 0.1
            # esn = EsnModel(2, 1, 1000,
            #                 spectral_scale=1.0, echo_param=0.85,
            #                 input_weight_scale=1.0, regulariser=1e-5)
            esn.train(X_train, y_train, warmup_timesteps=warm_up_length)
            # j = np.random.randint(1000)
            # esn.layers[0].W_res[j, j] = 0.0
            # esn.layers[0].W_res[:, j] = 0.0
            # esn.layers[0].W_res[j, :] = 0.0

            esn_outputs = []

            # esn.layers[0].drop_probability = 0.0
            # generate test-data
            #esn_outputs = esn.generate(X_valid, len(y_valid))
            #error_first = nrmse(y_valid, esn_outputs, data_mean)
            #print('ESN NRMSE: %f' % error_first)

            gen_data = np.vstack((X_train[-warm_up_length:], X_valid))
            esn_outputs = esn.generate(gen_data, len(y_valid), warmup_timesteps=warm_up_length) #, reset_increment=3)
            #esn_outputs = esn.generate(gen_data, len(y_valid), warmup_timesteps=warm_up_length)
            n_esn_outputs.append(esn_outputs)

            error = nrmse(y_valid, esn_outputs, data_mean)
            print('DMESN NRMSE: %f' % error)

            #assert np.round(error, 3) == np.round(error_first, 3), "errors after warm-up and generation must be the same"

            for i in range(len(y_valid)):
                y_valid_slice = y_valid[:i]
                esn_outputs_slice = esn_outputs[:i]
                err = nrmse(y_valid_slice, esn_outputs_slice, data_mean)
                t_nrmse[n, i] = err

            _nrmse[n,r] = error

        print("--ERR. MEAN: {}+-{}".format(np.mean(_nrmse, 1), np.std(_nrmse, 1)))

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

        # p = 1.0
        # learn_rate = 0.001
        # for i in range(100):
        #     scale_mat = np.array([[p, 0.],[0., 1.]])
        #     er = np.dot(data_esn_xy, scale_mat)-data_valid_xy
        #     # print("err: {}".format(er**2))
        #     grad = np.dot(data_esn_xy.T, er)
        #     print("grad: {}".format(grad))
        #     print("p: {}".format(p))
        #     p -= learn_rate*grad[0, 0]

        # scale_mat = np.array([[p, 0.], [0., 1.]])
        # data_esn_xy_scaled = np.dot(data_esn_xy, scale_mat)

    pkl.dump((t_nrmse, _nrmse, model_params), open("DESN_results_henon.pkl", "wb"))

    # plot the test versus train
    f, ax = plt.subplots(figsize=(12, 12))
    ax.plot(range(len(esn_outputs)), esn_outputs, label='ESN')
    ax.plot(range(len(y_valid)), y_valid, label='True')
    # ax.plot(data_esn_xy_scaled[:, 0], data_esn_xy_scaled[:, 1], label="scaled")
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

    f3, ax3 = plt.subplots(figsize=(12, 12))
    m_nrmse = np.mean(_nrmse, 1)
    s_nrmse = np.std(_nrmse, 1)
    ax3.plot(data_size_query, m_nrmse)
    ax3.fill_between(data_size_query, m_nrmse-s_nrmse, m_nrmse+s_nrmse, alpha=0.2)

    # heatmap of cov
    # f3, ax3 = plt.subplots(figsize=(12, 12))
    # # ax3.imshow(cov_data, cmap='hot', interpolation='nearest')
    # print(cov_data)
    # sns.heatmap(cov_data, ax=ax3)

    plt.show()
