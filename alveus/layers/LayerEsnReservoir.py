import numpy as np
import numpy.linalg as la

from collections import deque

from .LayerReservoir import LayerReservoir

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


class LayerEsnReservoir(LayerReservoir):
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

    def __init__(self, input_size, num_units, echo_param=0.6, idx=None,
                 activation=np.tanh, debug=False):
        super(LayerEsnReservoir, self).__init__(input_size,
                                                num_units+input_size,
                                                num_units)
        self.echo_param = echo_param
        self.activation = activation
        self.idx = idx  # <- can assign reservoir a unique ID for debugging
        self.debug = debug

        # input-to-reservoir, reservoir-to-reservoir weights (not yet initialized)
        self.W_res = np.zeros((self.num_units, self.num_units))
        self.state = np.zeros(self.num_units)            # <- unit states

        # These parameters are initialized upon calling initialize_input_weights()
        # and initialize_reservoir_weights().
        self.spectral_scale = None
        self.sparsity = None
        self.W_res_init_strategy = None
        self.sparsity = None

        # helpful information to track
        # self.signals = []  # <- reservoir states over time during training
        self.max_signal_store = 100
        self.signals = deque(maxlen=self.max_signal_store) #<- reservoir states over time during training
        self.num_to_store = 50
        self.ins_init = False
        self.res_init = False

    def info(self):
        """
        (args): None
        (description):
        Print live info about the reservoir
        """
        out = u'Reservoir(num_units=%d, input_size=%d, \u03B5=%.2f)\n' % (self.num_units, self.input_size, self.echo_param)
        out += 'W_res - spec_scale: %.2f, %s init\n' % (self.spectral_scale, self.W_res_init_strategy)
        out += 'W_in  -      scale: %.2f, %s init' % (self.input_weights_scale, self.W_in_init_strategy)

    def initialize_reservoir(self, strategy='uniform', **kwargs):
        if 'spectral_scale' not in kwargs.keys():
            self.spectral_scale = 1.0
        else:
            self.spectral_scale = kwargs['spectral_scale']
        if 'strategy' not in kwargs.keys():
            self.W_res_init_strategy = 'uniform'
        else:
            self.W_res_init_strategy = kwargs['strategy']
        if 'sparsity' not in kwargs.keys():
            self.sparsity = 1.0
        else:
            self.sparsity = kwargs['sparsity']
        if 'offset' not in kwargs.keys():
            offset = 0.5
        else:
            offset = kwargs['offset']

        if self.W_res_init_strategy == 'binary':
            self.W_res = (np.random.rand(self.num_units, self.num_units) > 0.5).astype(float)
        elif self.W_res_init_strategy == 'uniform':
            self.W_res = np.random.rand(self.num_units, self.num_units)
        elif self.W_res_init_strategy == 'gaussian':
            self.W_res = np.random.randn(self.num_units, self.num_units)
        else:
            raise ValueError('unknown res. weight init strategy %s' %
                             self.W_res_init_strategy)

        # apply the sparsity
        sparsity_matrix = (np.random.rand(self.num_units, self.num_units) < self.sparsity).astype(float)

        self.W_res -= offset
        self.W_res *= sparsity_matrix
        self.W_res /= max(abs(la.eig(self.W_res)[0]))
        self.W_res *= self.spectral_scale
        self.res_init = True

    def reset(self):
        super(LayerEsnReservoir, self).reset()
        self.state = np.zeros(self.num_units) 

    def forward(self, x):
        """
        Forward propagate input signal u(n) (at time n) through reservoir.

        x: input_size-dimensional input vector
        """
        super(LayerEsnReservoir, self).forward(x)
        assert self.ins_init, "Res. input weights not yet initialized (ID=%d)." % self.idx
        assert self.res_init, "Res. recurrent weights not yet initialized (ID=%d)." % self.idx

        prob = 0.9
        # dropped = (np.random.rand(*np.shape(self.W_res)) < prob).astype(float)
        mask_n = (np.random.rand(self.num_units,1) < prob).astype(float)
        # print("V", np.repeat(mask_n, self.num_units, axis=1))
        # print("H", np.repeat(mask_n.T, self.num_units, axis=0))
        mask_v = np.repeat(mask_n, self.num_units, axis=1)
        dropped = mask_v * mask_v.T

        in_to_res = np.dot(self.W_in, x).squeeze()
        res_to_res = np.dot(self.state.reshape(1, -1), self.W_res * dropped)

        # Equation (1) in "Formalism and Theory" of Scholarpedia page
        self.state = (1. - self.echo_param) * self.state + self.echo_param * self.activation(in_to_res + res_to_res)
        # self.signals.append(self.state[:self.num_to_store].tolist())

        # return the reservoir state appended to the input
        output = np.hstack((self.state.squeeze(), x))

        return output
