import numpy as np


class Layer(object):
    """
    Abstract class representing a layer for a feedforward model.
    """
    def __init__(self, input_size, output_size):
        """
        input_size      : size of the input data to this layer
        output_size     : size of the output data from this layer
        """
        self.input_size = input_size
        self.output_size = output_size

    def reset(self):
        pass

    def forward(self, x):
        """
        x   : data to feed through the layer
        """
        x = x.squeeze()
        assert (self.input_size == 1 and x.shape == ()) or len(x) == self.input_size, "unexpected input dimensionality of {}, expected: {}".format(np.shape(x), self.input_size)


class LayerTrainable(Layer):
    """
    Abstract layer class that can be trained from data
    """
    def __init__(self, input_size, output_size):
        super(LayerTrainable, self).__init__(input_size, output_size)

    def train(self, y_pred, y_actual):
        """
        y_pred      :   the predicted output of the model
        y_actual    :   the true output the model should have
        """
        assert np.shape(y_actual)[1] == self.output_size, "training data has the wrong output dimensions: {} vs {}".format(np.shape(y_actual)[1], self.output_size)
        assert np.shape(y_pred)[1] == self.input_size, "training data has the wrong input dimensions: {} vs {}".format(np.shape(y_pred)[1], self.input_size)
