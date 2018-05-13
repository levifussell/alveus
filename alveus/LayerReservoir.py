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

class LayerReservoir(Layer):
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

    def __init__(self, input_size, output_size, num_units, debug=False):
        super(LayerReservoir, self).__init__(input_size, output_size)
        self.num_units = num_units
        self.debug = debug

        self.input_weights_scale = None
        self.W_in = np.zeros((self.num_units, self.input_size))
        self.W_in_init_strategy = None

    def info(self):
        """
        (args): None
        (description):
        Here you describe a printout of your reservoir
        """
        pass

    def initialize_input_weights(self, strategy='uniform-binary-sign', scale=1e-2, offset=0.5, sparsity=1.0):
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
        elif strategy == 'uniform-binary-sign':
            # based on: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=5629375    
            self.W_in = np.random.rand(self.num_units, self.input_size) * ((np.random.rand(self.num_units, self.input_size) > 0.5).astype(float) * 2 - 1.0)
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

    def initialize_reservoir(self, strategy='uniform', **kwargs):
        pass

    def forward(self, x):
        """
        Forward propagate input signal u(n) (at time n) through reservoir.

        x: input_size-dimensional input vector
        """
        super(LayerReservoir, self).forward(x)

        pass
