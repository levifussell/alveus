from ..layers.LayerEsnReservoir import LayerEsnReservoir
from ..layers.LayerLinearRegression import LayerLinearRegression
from .LayeredModel import LayeredModel

import numpy as np

class EsnModel(LayeredModel):

    def __init__(self, input_size, output_size, reservoir_size,
                 spectral_scale=1.25, echo_param=0.85,
                 input_weight_scale=1.0, regulariser=1e-5, sparsity=1.0,
                 res_initialise_strategy='uniform',
                 activation=np.tanh):
        """
        input_size          : input dimension of the data
        output_size         : output dimension of the data
        reservoir_size      : size of the reservoir
        spectral_scale      : how much to scale the reservoir weights
        echo_param          : 'leaky' rate of the activations of the reservoir
                              units
        input_weight_scale  : how much to scale the input weights by
        regulariser         : regularisation parameter for the linear
                              regression output
        """
        layer_res = LayerEsnReservoir(input_size, reservoir_size, 
                                    output_size=reservoir_size+input_size, echo_param=echo_param, activation=activation)#activation=(lambda x : (x > 0).astype(float)*x))
        layer_res.initialize_input_weights(scale=input_weight_scale,
                                           strategy=res_initialise_strategy)
        layer_res.initialize_reservoir(spectral_scale=spectral_scale, sparsity=sparsity)
        layer_lr = LayerLinearRegression(reservoir_size+input_size,
                                         output_size, regulariser=regulariser)
        layers = [layer_res, layer_lr]

        super(EsnModel, self).__init__(layers)

    def forward(self, x, end_layer=None):

        # if doing the full feedforward, override the last reservoir to append the input
        if end_layer is None or end_layer == len(self.layers):
            y_p = super(EsnModel, self).forward(x=x, end_layer=len(self.layers)-1)
            y_p = np.hstack((y_p, x))
            y_p = self.layers[-1].forward(y_p)
        # if doing only the reservoir feedforward, append the input at the end
        else:
            y_p = super(EsnModel, self).forward(x=x, end_layer=end_layer)
            if end_layer == len(self.layers)-1:
                y_p = np.hstack((y_p, x))

        return y_p
