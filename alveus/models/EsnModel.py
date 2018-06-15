from ..layers.LayerEsnReservoir import LayerEsnReservoir
from ..layers.LayerLinearRegression import LayerLinearRegression
from .LayeredModel import LayeredModel


class EsnModel(LayeredModel):

    def __init__(self, input_size, output_size, reservoir_size,
                 spectral_scale=1.25, echo_param=0.85,
                 input_weight_scale=1.0, regulariser=1e-5):
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
        layer_res = LayerEsnReservoir(input_size, reservoir_size, echo_param)
        layer_res.initialize_input_weights(scale=input_weight_scale,
                                           strategy="binary")
        layer_res.initialize_reservoir(spectral_scale=spectral_scale)
        layer_lr = LayerLinearRegression(reservoir_size+input_size,
                                         output_size, regulariser=regulariser)
        layers = [layer_res, layer_lr]

        super(EsnModel, self).__init__(layers)
