import numpy as np
import numpy.linalg as la
import pickle as pkl
import time
import matplotlib.pyplot as plt

from Layer import Layer

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

class LayerEsnReservoir(Layer):
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

    def __init__(self, input_size, num_units, echo_param=0.6, idx=None, activation=np.tanh, 
                    debug=False):
        super(LayerEsnReservoir, self).__init__(input_size, num_units+input_size)
        self.num_units = num_units
        self.echo_param = echo_param
        self.activation = activation
        self.idx = idx                # <- can assign reservoir a unique ID for debugging
        self.debug = debug

        # input-to-reservoir, reservoir-to-reservoir weights (not yet initialized)
        self.W_in = np.zeros((self.num_units, self.input_size))
        self.W_res = np.zeros((self.num_units, self.num_units))
        self.state = np.zeros(self.num_units)            # <- unit states

        # These parameters are initialized upon calling initialize_input_weights()
        # and initialize_reservoir_weights().
        self.spectral_scale = None
        self.sparsity = None
        self.W_res_init_strategy = None
        self.input_weights_scale = None
        self.W_in_init_strategy = None
        self.sparsity = None

        # helpful information to track
        #if self.debug:
        self.signals = [] # <- reservoir states over time during training
        self.num_to_store = 50
        self.ins_init = False; self.res_init = False

    def info(self):
        """
        (args): None
        (description):
        Print live info about the reservoir
        """
        out = u'Reservoir(num_units=%d, input_size=%d, \u03B5=%.2f)\n' % (self.num_units, self.input_size, self.echo_param)
        out += 'W_res - spec_scale: %.2f, %s init\n' % (self.spectral_scale, self.W_res_init_strategy)
        out += 'W_in  -      scale: %.2f, %s init' % (self.input_weights_scale, self.W_in_init_strategy)

    def initialize_input_weights(self, strategy='binary', scale=1e-2, offset=0.5, sparsity=1.0):
        """
        (args): 
        strategy    :   how the input weights should be initialised (binary, uniform, guassian)
        scale       :   how much to scale the input weights by after initialisation (default=1e-2)
        offset      :   bias offset to apply to all input weights after initialisation (default=0.5)
        sparsity    :   probability of an input weight being non-zero (default=1.0)
        (description):
        Print live info about the reservoir
        """
        self.input_weights_scale = scale
        self.W_in_init_strategy = strategy
        if strategy == 'binary':
            self.W_in = (np.random.rand(self.num_units, self.input_size) > 0.5).astype(float)
        elif strategy == 'uniform':
            self.W_in = np.random.rand(self.num_units, self.input_size)
        elif strategy == 'gaussian':
            self.W_in = np.random.randn(self.num_units, self.input_size)
        else:
            raise ValueError('unknown input weight init strategy %s' % strategy)

        self.sparsity_input = sparsity
        sparsity_matrix = (np.random.rand(self.num_units, self.input_size) < self.sparsity_input).astype(float)

        self.W_in -= offset
        self.W_in *= sparsity_matrix
        self.W_in *= self.input_weights_scale
        self.ins_init = True

    def initialize_reservoir_weights(self, strategy='uniform', spectral_scale=1.0, offset=0.5, 
                                     sparsity=1.0):
        self.spectral_scale = spectral_scale
        self.W_res_init_strategy = strategy
        self.sparsity = sparsity
        if strategy == 'binary':
            self.W_res = (np.random.rand(self.num_units, self.num_units) > 0.5).astype(float)
        elif strategy == 'uniform':
            self.W_res = np.random.rand(self.num_units, self.num_units)
        elif strategy == 'gaussian':
            self.W_res = np.random.randn(self.num_units, self.num_units)
        else:
            raise ValueError('unknown res. weight init strategy %s' % strategy)

        # apply the sparsity
        self.sparsity = sparsity
        sparsity_matrix = (np.random.rand(self.num_units, self.num_units) < self.sparsity).astype(float)

        self.W_res -= offset
        self.W_res *= sparsity_matrix
        self.W_res /= max(abs(la.eig(self.W_res)[0]))
        self.W_res *= self.spectral_scale
        self.res_init = True

    def forward(self, x):
        """
        Forward propagate input signal u(n) (at time n) through reservoir.

        x: input_size-dimensional input vector
        """
        super(LayerEsnReservoir, self).forward(x)

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

        in_to_res = np.dot(self.W_in, x).squeeze()
        res_to_res = np.dot(self.state.reshape(1, -1), self.W_res)

        # Equation (1) in "Formalism and Theory" of Scholarpedia page
        self.state = (1. - self.echo_param) * self.state + self.echo_param * self.activation(in_to_res + res_to_res)
        #if self.debug:
        self.signals.append(self.state[:self.num_to_store].tolist())

        # return the reservoir state appended to the input
        output = np.hstack((self.state.squeeze(), x))

        return output
