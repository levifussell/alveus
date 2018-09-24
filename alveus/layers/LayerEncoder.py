import numpy as np
from sklearn.decomposition import PCA
from ..layers.Layer import LayerTrainable
from ..models.VAE import VAE

import torch as th
from torch.autograd import Variable

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

class VaeWrapper():
    def __init__(self, input_size, dimension_reduction):
        """
        TODO
        """
        self.vae = VAE(input_size=input_size, latent_variable_size=dimension_reduction, epochs=300, learning_rate=1e-3, batch_size=128)
        self.has_been_trained = False

    def forward(self, x):
        assert self.has_been_trained, "VAE encoder has not been trained before feeding forward."
        x_var = Variable(th.FloatTensor(x.reshape(1, -1).squeeze()))
        mu, logvar = self.vae.encode(x_var)
        mu_np = mu.data.numpy()
        
        return mu_np

    def train(self, x_data):

        # normalising the VAE input is important (I think) - but the VAE does this automatically
        x_data_th = th.FloatTensor(x_data)
        self.vae.train_full(train_data=x_data_th, verbose=True)
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

class LayerVaeEncoder(LayerEncoder):

    def __init__(self, input_size, dimension_reduction):
        """
        TODO
        """
        super(LayerVaeEncoder, self).__init__(input_size=input_size, 
                                            dimension_reduction=dimension_reduction,
                                            encoder=VaeWrapper(input_size=input_size, dimension_reduction=dimension_reduction))
