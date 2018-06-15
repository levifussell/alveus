import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from ..layers.LayerSandPileReservoir import LayerSandPileReservoir
from ..layers.LayerLinearRegression import LayerLinearRegression
from .LayeredModel import LayeredModel


class SandPileModel(LayeredModel):

    def __init__(self, input_size, output_size, reservoir_size,
                 # spectral_scale=0.29401253252, thresh_scale=1.1142252352,
                 spectral_scale=0.2, thresh_scale=3.0,
                 input_weight_scale=0.01, regulariser=1e-6):
        """
        input_size          : input dimension of the data
        output_size         : output dimension of the data
        reservoir_size      : size of the reservoir
        spectral_scale      : how much to scale the reservoir weights
        echo_param          : 'leaky' rate of the activations of the reservoir units
        input_weight_scale  : how much to scale the input weights by
        regulariser         : regularisation parameter for the linear regression output
        """

        # live info
        self.live_im = None
        self.live_fig = None

        layer_res = LayerSandPileReservoir(input_size, reservoir_size)
        layer_res.initialize_input_weights(scale=input_weight_scale, strategy="uniform", offset=0.0, sparsity=0.1)
        # print(layer_res.W_in)
        # layer_res.initialize_threshold(layer_res.threshold_uniform, thresh_scale=thresh_scale)
        layer_res.initialize_threshold(layer_res.threshold_unit, thresh_scale=thresh_scale)
        layer_res.initialize_reservoir(strategy='uniform', spectral_scale=spectral_scale)

        layer_lr = LayerLinearRegression(reservoir_size+input_size, output_size, regulariser=regulariser)
        self.layers = [layer_res, layer_lr]

        super(SandPileModel, self).__init__(self.layers)

    def plot_reservoir(self):
        signals = self.layers[0].signals

        signals_shape = np.reshape(signals, (np.shape(signals)[0], -1))

        # print(np.shape(signals_shape))

        sns.heatmap(signals_shape)

        plt.plot()

    # def display(self):
    #     signals = self.layers[0].state
    #     signals_shape = np.reshape(signals, (np.shape(signals)[0], -1))
    #     # print(signals_shape)
    #     # print(np.shape(signals_shape))
    #     # create the figure
    #     if self.live_fig == None:
    #         self.live_fig = plt.figure()
    #         ax = self.live_fig.add_subplot(111)
    #         self.live_im = ax.imshow(signals_shape, cmap="Reds")
    #         # self.live_im = ax.imshow(self.weights,cmap="Reds")
    #         plt.show(block=False)
    #     else:
    #         # draw some data in loop
    #         # wait for a second
    #         time.sleep(0.1)
    #         # replace the image contents
    #         self.live_im.set_array(signals_shape)
    #         # self.live_im.set_array(self.weights)
    #         # redraw the figure
    #         self.live_fig.canvas.draw()
    #         plt.pause(0.001)
