from ..layers.LayerEsnReservoir import LayerEsnReservoir
from ..layers.LayerLinearRegression import LayerLinearRegression
from .LayeredModel import LayeredModel

import numpy as np

class DEsnModel(LayeredModel):

    def __init__(self, input_size, output_size, num_reservoirs=1, reservoir_sizes=[1000],
                 spectral_scales=[1.25], echo_params=[0.85],
                 input_weight_scales=[1.0], regulariser=1e-5, sparsities=[1.0],
                 res_initialise_strategies=['uniform']):
        """
        num_reservoirs       : number of reservoirs
        input_size           : input dimension of the data
        output_size          : output dimension of the data
        reservoir_sizes      : list of reservoir sizes
        spectral_scales      : how much to scale the reservoir weights
        echo_params          : 'leaky' rate of the activations of the reservoir
                              units
        input_weight_scales  : how much to scale the input weights by
        regulariser          : regularisation parameter for the linear
                              regression output
        """

        assert (len(reservoir_sizes) == 
                len(spectral_scales) == 
                len(echo_params) == 
                len(input_weight_scales) == 
                len(sparsities) == 
                len(res_initialise_strategies)), "length of parameter lists must match the number of reservoirs"

        # if a list of length one is created, duplicate this value to the number of reservoirs
        if len(reservoir_sizes) == 1:
            reservoir_sizes *= num_reservoirs
            spectral_scales *= num_reservoirs
            echo_params *= num_reservoirs
            input_weight_scales *= num_reservoirs
            sparsities *= num_reservoirs
            res_initialise_strategies *= num_reservoirs

        layers = []
        # add the reservoirs
        for r in range(num_reservoirs):
            if r == 0:
                in_size = input_size
            else:
                in_size = reservoir_sizes[r-1]

            if r == num_reservoirs - 1:
                out_size = reservoir_sizes[r] + input_size
            else:
                out_size = reservoir_sizes[r]

            layer_res = LayerEsnReservoir(input_size=in_size, num_units=reservoir_sizes[r], output_size=out_size,
                                            echo_param=echo_params[r], activation=np.tanh, idx=r)#activation=(lambda x : (x > 0).astype(float)*x))
            layer_res.initialize_input_weights(scale=input_weight_scales[r],
                                               strategy=res_initialise_strategies[r])
            layer_res.initialize_reservoir(spectral_scale=spectral_scales[r], sparsity=sparsities[r])
            layers.append(layer_res)


        layer_lr = LayerLinearRegression(reservoir_sizes[-1]+input_size,
                                         output_size, regulariser=regulariser)
        layers.append(layer_lr)

        super(DEsnModel, self).__init__(layers)

    def forward(self, x, end_layer=None):

        # if doing the full feedforward, override the last reservoir to append the input
        if end_layer is None or end_layer == len(self.layers):
            y_p = super(DEsnModel, self).forward(x=x, end_layer=len(self.layers)-1)
            y_p = np.hstack((y_p, x))
            y_p = self.layers[-1].forward(y_p)
        # if doing only the reservoir feedforward, append the input at the end
        else:
            y_p = super(DEsnModel, self).forward(x=x, end_layer=end_layer)
            if end_layer == len(self.layers)-1:
                y_p = np.hstack((y_p, x))

        return y_p
