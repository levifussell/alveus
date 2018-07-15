import numpy as np

from ..layers.Layer import LayerTrainable


class LayeredModel(object):

    def __init__(self, layers):
        """
        layers  :   a list of layers. Treated as a feed-forward model
        """
        assert len(layers) > 0, "Model layers must be non-empty"

        # check that the output of each layer is the same size as the input of
        # the next layer
        for l1, l2 in zip(layers[:-1], layers[1:]):
            assert l1.output_size == l2.input_size, "layers do not match input to output in the model"

        self.layers = layers

    def reset(self):
        for l in self.layers:
            l.reset()

    def forward(self, x, end_layer=None):
        """
        x           : data to push through the network
        end_layer   : the layer to stop the forward movement of the data. Used for training. (default=None)
        """
        x = x.squeeze()

        assert (self.layers[0].input_size == 1 and x.shape == ()) or len(x) == self.layers[0].input_size, "unexpected input dimensionality (check bias)"

        # if an end layer has not been named, feedforward the entire model
        if end_layer is None:
            f_layers = self.layers
        else:
            f_layers = self.layers[:end_layer]

        # for l in f_layers:
        #     x = np.array(l.forward(x))

        for l in f_layers:
            x = l.forward(x)

        return x

    def train(self, X, y, warmup_timesteps=100, data_repeats=1):
        """
        x                   : input data to train on
        y                   : output data to train on
        warmup_timesteps    : number of timesteps to run the data before training (default=100)
        """
        assert isinstance(self.layers[-1], LayerTrainable), "This model cannot be trained because the final layer of type {} is not trainable".format(type(self.layers[-1]))

        # TODO: for now we assume ONLY the last layer can be trained

        # warmup stage
        # for x in X[:warmup_timesteps]:
        #     # some function that allows us to display
        #     self.display()

        #     _ = self.forward(x, len(self.layers)-1)

        # # training stage
        # y_forward = np.zeros((np.shape(X[warmup_timesteps:])[0],
        #                       self.layers[-1].input_size))
        # for idx, x in enumerate(X[warmup_timesteps:]):
        #     # some function that allows us to display
        #     self.display()

        #     y_p = self.forward(x, len(self.layers)-1)
        #     y_forward[idx, :] = y_p

        # y_nonwarmup = y[warmup_timesteps:]
        y_forward = np.zeros((np.shape(X)[0] - data_repeats*warmup_timesteps,
                              self.layers[-1].input_size)) 
        y_nonwarmup = np.zeros((np.shape(y)[0] - data_repeats*warmup_timesteps,
                                np.shape(y)[1]))
        y_idx = 0
        data_rate = np.shape(X)[0] / data_repeats
        # print(data_rate)
        # print(X[:10])
        # print(X[data_rate:(data_rate+10)])
        for idx,x in enumerate(X):
            # some function that allows us to display
            self.display()

            # if idx % data_rate == 0:
            #     print(x)
            #     self.reset()
            
            if idx % data_rate < warmup_timesteps:
                _ = self.forward(x, len(self.layers)-1)
            else:
                y_p = self.forward(x, len(self.layers)-1)
                y_forward[y_idx, :] = y_p
                y_nonwarmup[y_idx, :] = y[idx, :]
                y_idx += 1

        # training stage
        # y_forward = np.zeros((np.shape(X[warmup_timesteps:])[0],
        #                       self.layers[-1].input_size))
        # for idx, x in enumerate(X[warmup_timesteps:]):
        #     # some function that allows us to display
        #     self.display()

        #     y_p = self.forward(x, len(self.layers)-1)
        #     y_forward[idx, :] = y_p

        # y_nonwarmup = y[warmup_timesteps:]

        self.layers[-1].train(y_forward, y_nonwarmup)

    def generate(self, x_data, count, reset_increment=-1, warmup_timesteps=0):
        """
        Given a single datapoint, the model will feed this back into itself
        to produce generative output data.

        x_data          : data to generate from (the first data point will be used unless reset_increment != -1)
        count           : number of times to run the generative process
        reset_increment : how often to feed the generator the 'real' data value (default=-1 <= no reset)
        """
        # y_outputs = []
        y_outputs = np.zeros(count)
        # x = np.array(x_data[0])
        x = x_data[0]
        for e in range(-warmup_timesteps, count, 1):

            # some function that allows us to display
            self.display()

            # if we enable reseting, feed the 'real' data in (e == 0) is for warm-up swap
            if e == 0 or (reset_increment != -1 and e % reset_increment == 0):
                assert e < len(x_data), "generating data is less than the specified count"
                x = x_data[e + warmup_timesteps]

            # forward generating without 'warmup'
            if e >= 0:
                x = self.forward(x)
                y_outputs[e] = x
                x = np.hstack((x, 1))
            # forward generating with 'warmup'
            else:
                _ = self.forward(x_data[e + warmup_timesteps])

        # return np.array(y_outputs).squeeze()
        return y_outputs.squeeze()

    def get_output_size(self):
        return self.layers[-1].output_size

    def get_input_size(self):
        return self.layers[0].input_size

    def display(self):
        pass
