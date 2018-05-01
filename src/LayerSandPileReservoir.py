import numpy as np
import numpy.linalg as la
import pickle as pkl
import time
import matplotlib.pyplot as plt

from LayerReservoir import LayerReservoir

"""
Notes (from scholarpedia):
    -The SPECTRAL RADIUS of the reservoir weights codetermines:
        (1): (?)
        (2): amount of nonlinear interaction of input components through time 
                (larger spectral radius ==> longer-range interactions)
    -INPUT SCALING codetermines the degree of nonlinearity of the reservoir dynamics. Examples:
        (1): very small input amplitudes ==> reservoir behaves almost like linear medium.
        (2): very large input amplitudes ==> drives the reservoir neurons to the saturation of the
                                              sigmoid, and a binary switching dynamic results.
    -OUTPUT FEEDBACK SCALING determines the extent to which the trained ESN has an autonomous
     generation component.
        (1):      no output feedback: ESN unable to generate predictions for future time steps.
        (2): nonzero output feedbacl: danger of dynamical instability.
    -CONNECTIVITY/SPARSITY of reservoir weight matrix:
        (1) todo
"""

class LayerSandPileReservoir(LayerReservoir):
    """
    (args):
    input_size  :    input signal is input_size dimensions.
    num_units   :    reservoir has num_units units.
    idx         :    unique ID of the reservoir (default=None) -- good for debug/multiple reservoirs
    topple_iters:    number of iterations to run of the model before the next data input
    topple_div  :    amount of sand to divide to the neighbours
    debug       :    when True, this will print live information (default=False)
    (description): reservoir class. Extend this class to create different reservoirs

    """

    def __init__(self, input_size, num_units, topple_iters=1, topple_div=6, idx=None, 
                    debug=False):
        super(LayerSandPileReservoir, self).__init__(input_size, num_units+input_size, num_units)
        self.idx = idx                # <- can assign reservoir a unique ID for debugging
        self.debug = debug

        assert int(np.sqrt(num_units))**2 == num_units, "number of units must be a square number!"

        # input-to-reservoir, reservoir-to-reservoir weights (not yet initialized)
        # self.W_in = np.zeros((self.num_units, self.input_size))
        self.state = np.zeros((int(np.sqrt(num_units)), int(np.sqrt(num_units))))            # <- unit states

        self.topple_iters = topple_iters
        self.topple_div = topple_div

        self.thresholds = None

        # helpful information to track
        self.signals = [] # <- reservoir states over time during training
        self.ins_init = False; self.res_init = False

        self.spectral_scale = None

    def initialize_threshold(self, tresh_init_function, thresh_scale=0.5):
        self.thresholds = tresh_init_function(thresh_scale)

    def threshold_uniform(self, thresh_scale=0.5):
        return np.zeros_like(self.state) + np.random.rand(np.shape(self.state)[0], np.shape(self.state)[1])*thresh_scale

    def threshold_static_uniform(self, thresh_scale=0.5):
        return np.zeros_like(self.state) + np.random.rand()*thresh_scale

    def threshold_unit(self, thresh_scale=0.5):
        return np.zeros_like(self.state) + thresh_scale

    def initialize_reservoir(self, strategy='static', **kwargs):
        if 'spectral_scale' not in kwargs.keys():
            self.spectral_scale = 1.0
        else:
            self.spectral_scale = kwargs['spectral_scale']
            
        if strategy == 'uniform':
            self.state += np.random.rand(np.shape(self.state)[0], np.shape(self.state)[0])*self.spectral_scale
            # s = np.random.rand()*self.spectral_scale
            # print(s)
            # self.state += s
        elif strategy == 'static':
            self.state += self.spectral_scale

        self.res_init = True

    def forward(self, x):
        """
        Forward propagate input signal u(n) (at time n) through reservoir.

        x: input_size-dimensional input vector
        """
        super(LayerSandPileReservoir, self).forward(x)

        assert self.ins_init, "Res. input weights not yet initialized (ID=%d)." % self.idx
        assert self.res_init, "Res. recurrent weights not yet initialized (ID=%d)." % self.idx

        # add the 'sand' on top always
        sand_to_drop = np.reshape(np.dot(self.W_in, x), np.shape(self.state))
        self.state += sand_to_drop  


        for i in range(self.topple_iters):
            # for now I do a simple, ad-hoc sandpile model
            toppled = np.abs(self.state) >= self.thresholds
            # toppled_idx = np.argwhere(toppled)
            # print(toppled)

            # remove sand from toppled points
            # sand_removed = toppled.astype(float) * (self.state - self.thresholds)
            sand_removed = toppled.astype(float) * self.state #self.thresholds

            self.state -= sand_removed #toppled.astype(float) * self.thresholds

            # distribute sand evenly to neighbours (for now)
            distr = sand_removed / self.topple_div #np.shape(self.state)[0]

            leftDistr = np.hstack((distr[:, 1:], distr[:, 0][:, None]))
            rightDistr = np.hstack((distr[:, -1][:, None], distr[:, :-1]))
            upDistr = np.vstack((distr[1:, :], distr[0, :][None, :]))
            downDistr = np.vstack(( distr[-1, :][None, :], distr[:-1, :]))

            self.state += leftDistr + rightDistr + upDistr + downDistr

            # uncomment below to distribute to 15 random cells
            # for i in range(15):
            #     id = (np.random.rand(np.shape(self.state)[0], np.shape(self.state)[1])).astype(int)
            #     self.state[id] += distr
            
            self.signals.append(self.state.tolist())


        # return the reservoir state appended to the input
        output = np.hstack((np.reshape(self.state, (1, np.shape(self.state)[0]**2)).squeeze(), x))

        return output
