import numpy as np
import numpy.linalg as la

from .Layer import LayerTrainable


class LayerLinearRegression(LayerTrainable):

    def __init__(self, input_size, output_size, regulariser=1e-8, debug=False):
        """
        input_size      : size of the input data to this layer
        output_size     : size of the output data from this layer
        regulariser     : reuglarisation for linear regression (default=1e-8)
        debug           : whether to print live training output (default=False)
        """
        super(LayerLinearRegression, self).__init__(input_size, output_size)

        self.input_size = input_size
        self.output_size = output_size
        self.regulariser = regulariser
        self.debug = debug

        # output weights to train
        self.W_out = np.ones((self.output_size, self.input_size))

    def forward(self, x):
        super(LayerLinearRegression, self).forward(x)

        # add the bias
        # x = np.array(np.hstack((x, 1)))
        x = np.hstack((x, 1))

        # linear forward
        output = np.dot(self.W_out, x)

        return output.squeeze()

    def train(self, y_pred, y_actual):
        super(LayerLinearRegression, self).train(y_pred, y_actual)

        # add a bias
        S = np.hstack((y_pred, np.ones((y_pred.shape[0], 1))))
        # Solve (W_out)(S.T) = (D) by least squares
        T1 = np.dot(y_actual.T, S)
        # compute the regulariser
        reg = self.regulariser * np.eye(np.shape(S)[1])
        # no regularisation on the bias
        reg[-1, -1] = 0
        T2 = la.inv(np.dot(S.T, S) + reg)
        self.W_out = np.dot(T1, T2)
