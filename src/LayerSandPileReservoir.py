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
    echo_param  :    leaky rate of the reservoir units
    activation  :    activation function of the reservoir units (default=tanh)
    debug       :    when True, this will print live information (default=False)
    (description): reservoir class. Extend this class to create different reservoirs

    """

    def __init__(self, input_size, num_units, idx=None, 
                    debug=False):
        super(LayerSandPileReservoir, self).__init__(input_size, num_units+input_size, num_units)
        self.idx = idx                # <- can assign reservoir a unique ID for debugging
        self.debug = debug

        assert int(np.sqrt(num_units))**2 == num_units, "number of units must be a square number!"

        # input-to-reservoir, reservoir-to-reservoir weights (not yet initialized)
        # self.W_in = np.zeros((self.num_units, self.input_size))
        self.state = np.zeros((int(np.sqrt(num_units)), int(np.sqrt(num_units))))            # <- unit states

        self.thresholds = None

        # helpful information to track
        #if self.debug:
        self.signals = [] # <- reservoir states over time during training
        self.num_to_store = 50
        self.ins_init = False; self.res_init = False

        self.spectral_scale = None

    # def info(self):
    #     """
    #     (args): None
    #     (description):
    #     Print live info about the reservoir
    #     """
    #     out = u'Reservoir(num_units=%d, input_size=%d, \u03B5=%.2f)\n' % (self.num_units, self.input_size, self.echo_param)
    #     out += 'W_res - spec_scale: %.2f, %s init\n' % (self.spectral_scale, self.W_res_init_strategy)
    #     out += 'W_in  -      scale: %.2f, %s init' % (self.input_weights_scale, self.W_in_init_strategy)

    def initialize_threshold(self, tresh_init_function, thresh_scale=0.5):
        self.thresholds = tresh_init_function(thresh_scale)

    def threshold_uniform(self, thresh_scale=0.5):
        return np.zeros_like(self.state) + np.random.rand()*thresh_scale

    def initialize_reservoir(self, strategy='static', **kwargs):
                                     #spectral_scale=1.0, offset=0.5, 
                                     #sparsity=1.0):
        if 'spectral_scale' not in kwargs.keys():
            self.spectral_scale = 1.0
        else:
            self.spectral_scale = kwargs['spectral_scale']
            
        if strategy == 'uniform':
            self.state += np.random.rand(np.shape(self.state)[0], np.shape(self.state)[0])*self.spectral_scale
        elif strategy == 'static':
            self.state += self.spectral_scale

        self.res_init = True

    def forward(self, x):
        """
        Forward propagate input signal u(n) (at time n) through reservoir.

        x: input_size-dimensional input vector
        """
        super(LayerSandPileReservoir, self).forward(x)

        # x = x.squeeze()
        # try:
        #     assert (self.input_size == 1 and x.shape == ()) or x.shape[0] == self.W_in.shape[1], \
        #         "u(n): %s.  W_res: %s (ID=%d)" % (x.shape, self.W_res.shape, self.idx)
        # except:
        #     print(x.shape)
        #     print(self.W_in.shape)
        #     print(self.W_res.shape)
        #     raise
        assert self.ins_init, "Res. input weights not yet initialized (ID=%d)." % self.idx
        assert self.res_init, "Res. recurrent weights not yet initialized (ID=%d)." % self.idx

        # print(np.shape(self.state))
        # print(np.dot(self.W_in, x))

        # add the 'sand' on top always
        self.state += np.reshape(np.dot(self.W_in, x), np.shape(self.state))

        # for now I do a simple, ad-hoc sandpile model
        toppled = self.state > self.thresholds
        toppled_idx = np.argwhere(toppled)

        # remove sand from toppled points
        self.state -= toppled.astype(float) * self.thresholds

        # distribute sand evenly to neighbours (for now)
        distr = self.thresholds[toppled_idx] / 4.0
        # print(toppled_idx)
        leftThreshold = (toppled_idx - np.array([0, 1])) % np.shape(self.state)[0] 
        rightThreshold = (toppled_idx + np.array([0, 1])) % np.shape(self.state)[0] 
        upThreshold = (toppled_idx + np.array([1, 0])) % np.shape(self.state)[0] 
        downThreshold = (toppled_idx - np.array([1, 0])) % np.shape(self.state)[0] 
        self.state[leftThreshold] += distr
        self.state[rightThreshold] += distr
        self.state[upThreshold] += distr
        self.state[downThreshold] += distr

        #if self.debug:
        self.signals.append(self.state[:self.num_to_store].tolist())

        # print(self.state)

        # return the reservoir state appended to the input
        output = np.hstack((np.reshape(self.state, (1, np.shape(self.state)[0]**2)).squeeze(), x))
        # print(output)
        return output
