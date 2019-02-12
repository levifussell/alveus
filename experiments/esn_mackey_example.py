import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp

import alveus as alv
import alveus.data.generators as generators
import alveus.models as models
from alveus.utils.metrics import nrmse

if __name__ == "__main__":

    '''
    Load the data.
    '''
    data = np.array([generators.generateMackey(5100)]).reshape(-1,1)
    DATA_MEAN = np.mean(data, axis=0)
    split = 4000
    # adding a bias significantly improves performance
    X_train = np.hstack((np.array(data[:split-1]), np.ones_like(data[:split-1, :1])))
    y_train = np.array(data[1:split])
    X_valid = np.hstack((np.array(data[split-1:-1]), np.ones_like(data[split-1:-1, :1])))
    y_valid = np.array(data[split:])


    '''
    Train the ESN.
    '''

    esn = models.EsnModel(input_size=2, output_size=1, reservoir_size=1000)
    esn.train(X_train, y_train, warmup_timesteps=300)

    '''
    Evaluate the ESN.
    '''

    # generate test-data starting from the next data point.
    esn_outputs = esn.generate(X_valid, len(y_valid))
    error_gen = nrmse(y_valid, esn_outputs, DATA_MEAN)
    print('ESN NRMSE VALID 1: %f' % error_gen)

    # generate test data from earlier data requires using a warmup of 300
    #  so that our reservoir state is in synch with the predicted data point.
    warmup = 300
    gen_data = np.vstack((X_train[-warmup:], X_valid))
    esn_outputs = esn.generate(gen_data, len(y_valid), warmup_timesteps=warmup)

    error_pre_gen = nrmse(y_valid, esn_outputs, DATA_MEAN)
    print('ESN NRMSE VALID 2: %f' % error_pre_gen)

    # notice how both methods of training produce the same predictions. Try changing
    #  the 'warmup' to a low value like zero and see if they are still the same.
    assert np.round(error_gen, 3) == np.round(error_pre_gen, 3), "errors after warm-up and generation must be the same"

    # compute the NRMSE for iteratively larger trajectories into the future
    #  so that we can create a plot showing how the ESN slowly diverges.
    t_nrmse = [ nrmse(y_valid[:i], esn_outputs[:i], DATA_MEAN) for i in range(len(y_valid)) ]


    '''
    Plot the results.
    '''

    # plot the test versus train
    f, ax = plt.subplots(figsize=(12, 12))
    ax.plot(range(len(esn_outputs)), esn_outputs, label='ESN')
    ax.plot(range(len(y_valid)), y_valid, label='True')
    ax.legend()

    # plot the NRMSE 
    f2, ax2 = plt.subplots(figsize=(12, 12))
    ax2.plot(range(len(t_nrmse)), t_nrmse)
    ax2.set_xlabel("timestep")
    ax2.set_ylabel("NRMSE (sum)")

    plt.show()
