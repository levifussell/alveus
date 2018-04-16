import numpy as np

from Layer import Layer, LayerTrainable

class LayeredModel(object):

    def __init__(self, layers):
        """
        layers  :   a list of layers. Treated as a feed-forward model
        """
        assert len(layers) > 0, "Model layers must be non-empty"

        # check that the output of each layer is the same size as the input of the next layer
        for l1,l2 in zip(layers[:-1],layers[1:]):
            assert l1.output_size == l2.input_size, "layers do not match input to output in the model"

        self.layers = layers

    def forward(self, x, end_layer=None):
        """
        x           : data to be trained on
        end_layer   : the layer to stop the forward movement of the data. Used for training. (default=None)
        """
        x = x.squeeze()

        assert (self.layers[0].input_size == 1 and x.shape == ()) or len(x) == self.layers[0].input_size, "unexpected input dimensionality (check bias)"

        # if an end layer has not been named, feedforward the entire model
        if end_layer is None:
            f_layers = self.layers
        else:
            f_layers = self.layers[:end_layer]

        for l in f_layers:
            x = np.array(l.forward(x)) 

        return x

    def train(self, X, y, warmup_timesteps=100):
        """
        x                   : input data to train on
        y                   : output data to train on
        warmup_timesteps    : number of timesteps to run the data before training (default=100)
        """
        assert isinstance(self.layers[-1], LayerTrainable), "This model cannot be trained because the final layer of type {} is not trainable".format(type(self.layers[-1]))

        #############TODO: for now we assume ONLY the last layer can be trained

        # warmup stage
        for x in X[:warmup_timesteps]:
            _ = self.forward(x, len(self.layers)-1)

        # training stage
        y_forward = np.zeros((np.shape(X[warmup_timesteps:])[0], 
                                self.layers[-1].input_size))
        for idx,x in enumerate(X[warmup_timesteps:]):
            y_p = self.forward(x, len(self.layers)-1)
            y_forward[idx, :] = y_p

        y_nonwarmup = y[warmup_timesteps:]

        self.layers[-1].train(y_forward, y_nonwarmup)

    def generate(self, x_start, count):
        """
        Given a single datapoint, the model will feed this back into itself 
        to produce generative output data.

        x_start     : single data point to start a generative process from
        count       : number of times to run the generative process
        """
        y_outputs = []
        x = np.array(x_start)
        for _ in range(count):
            x = np.array(self.forward(x))
            y_outputs.append(x)
            x = np.hstack((x, 1))

        return np.array(y_outputs).squeeze()

    # TODO: below needs to be refactored to the pep18 style
    def get_output_size(self):
        return self.layers[-1].output_size

    def get_input_size(self):
        return self.layers[0].input_size
    
        



