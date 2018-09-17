import numpy as np
from sklearn.decomposition import PCA
from ..layers.Layer import LayerTrainable

class PcaWrapper():
    def __init__(self, dimension_reduction):
        """
        TODO
        """
        self.pca = PCA(n_components = dimension_reduction)
        self.has_been_trained = False

    def forward(self, x):
        assert self.has_been_trained, "PCA encoder has not been trained before feeding forward."
        y = self.pca.transform(x.reshape(1, -1)).squeeze()
        
        return y

    def train(self, x_data):

        # sklearn PCA automatically zero-means the data so we don't have to.
        self.pca.fit(x_data)
        self.has_been_trained = True

class LayerEncoder(LayerTrainable):

    def __init__(self, input_size, dimension_reduction, encoder):
        """
        TODO
        """
        # the output size of the encoder is its latent dimensionality
        super(LayerEncoder, self).__init__(input_size, dimension_reduction)

        self.encoder = encoder

    def forward(self, x):
        super(LayerEncoder, self).forward(x)

        y = self.encoder.forward(x)

        return y.squeeze()

    def train(self, x_data):

        self.encoder.train(x_data)

class LayerPcaEncoder(LayerEncoder):

    def __init__(self, input_size, dimension_reduction):
        """
        TODO
        """
        super(LayerPcaEncoder, self).__init__(input_size=input_size, 
                                            dimension_reduction=dimension_reduction,
                                            encoder=PcaWrapper(dimension_reduction))

