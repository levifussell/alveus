from ..layers.LayerEsnReservoir import LayerEsnReservoir
from ..layers.LayerLinearRegression import LayerLinearRegression
from ..layers.LayerEncoder import LayerPcaEncoder
from .LayeredModel import LayeredModel

import numpy as np

class DMEsnModel(LayeredModel):

    def __init__(self, input_size, output_size, num_reservoirs=1, reservoir_sizes=[1000],
                 spectral_scales=[1.25], echo_params=[0.85],
                 input_weight_scales=[1.0], regulariser=1e-5, sparsities=[1.0],
                 res_initialise_strategies=['uniform'],
                 encoder_layers=[LayerPcaEncoder],
                 encoder_dimensions=[60],
                    # if set to true, the first encoder will be trained ONCE and used for all subsequent encoders 
                    #    (NOTE: there is an assumption that all encoder dimensions are the same in the encoder_dimensions list)
                 reuse_encoder=False, 
                 activation=np.tanh):
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
                len(encoder_layers) + 1 == # one less encoder layer
                len(encoder_dimensions) + 1 == # one less encoder layer
                len(res_initialise_strategies)), "length of parameter lists must match the number of reservoirs. res_size {}, spec {}, echo {}, input_w {}, sparse {}, encode_l {}, encode_dim {}, red_init {}".format(len(reservoir_sizes), len(spectral_scales), len(echo_params), len(input_weight_scales), len(sparsities), len(encoder_layers), len(encoder_dimensions), len(res_initialise_strategies))

        # if a list of length one is created, duplicate this value to the number of reservoirs
        if len(reservoir_sizes) == 1:
            reservoir_sizes *= num_reservoirs
            spectral_scales *= num_reservoirs
            echo_params *= num_reservoirs
            input_weight_scales *= num_reservoirs
            sparsities *= num_reservoirs
            res_initialise_strategies *= num_reservoirs
            encoder_layers *= num_reservoirs - 1
            encoder_dimensions *= num_reservoirs - 1

        self.reuse_encoder = reuse_encoder

        layers = []
        # add the reservoirs
        for r in range(num_reservoirs):
            if r == 0:
                in_size = input_size
            else:
                in_size = encoder_dimensions[r-1]

            if r == num_reservoirs - 1:
                out_size = reservoir_sizes[r] + input_size + np.sum(encoder_dimensions)
            else:
                out_size = reservoir_sizes[r]

            layer_res = LayerEsnReservoir(input_size=in_size, num_units=reservoir_sizes[r], output_size=out_size,
                                            echo_param=echo_params[r], activation=activation, idx=r)
            layer_res.initialize_input_weights(scale=input_weight_scales[r],
                                               strategy=res_initialise_strategies[r])
            layer_res.initialize_reservoir(spectral_scale=spectral_scales[r], sparsity=sparsities[r])
            layers.append(layer_res)

            # add encoding layers after all reservoirs except the final one
            if r < num_reservoirs - 1:
                layer_enc = encoder_layers[r](input_size=reservoir_sizes[r], dimension_reduction=encoder_dimensions[r])
                layers.append(layer_enc)

        # input to the linear regression should be the size of the encoders and the final reservoir
        layer_lr = LayerLinearRegression(reservoir_sizes[-1]+input_size+np.sum(encoder_dimensions),
                                         output_size, regulariser=regulariser)
        layers.append(layer_lr)

        self.num_reservoirs = num_reservoirs
        self.encoder_dimensions = encoder_dimensions

        super(DMEsnModel, self).__init__(layers)

    def forward_encoder(self, x, end_layer=None):
        """
        This function feeds through the encoders and collects their compressed representations
        """
        # if an end layer has not been named, feedforward the entire model
        if end_layer is None:
            f_layers = self.layers
        else:
            f_layers = self.layers[:end_layer]

        encoder_data = np.zeros(np.sum(self.encoder_dimensions))

        enc_idx = 0
        enc_culm = 0
        for idx,l in enumerate(f_layers):
            if (idx+1) % 2 == 0 and idx < len(self.layers) - 2: # if it is an encoder layer!
                if self.reuse_encoder:
                    x = self.layers[1].forward(x) # always use the first encoder
                else:
                    x = l.forward(x)
                encoder_data[enc_culm:(enc_culm + self.encoder_dimensions[enc_idx])] = x.squeeze()
                enc_culm += self.encoder_dimensions[enc_idx]
                enc_idx += 1
            else:
                x = l.forward(x)

        return x, encoder_data  # tuple of the feedforward data and the encoder data

    def train(self, X, y, warmup_timesteps=100):
        y_forward = np.zeros((np.shape(X)[0] - warmup_timesteps,
                              self.layers[-1].input_size)) 
        #y_nonwarmup = np.zeros((np.shape(y)[0] - warmup_timesteps,
        #                        np.shape(y)[1]))
        y_idx = 0
        #data_rate = np.shape(X)[0] / data_repeats
        # print(data_rate)
        # print(X[:10])
        # print(X[data_rate:(data_rate+10)])

        # warm-up rate
        warmup_interval = int(warmup_timesteps / self.num_reservoirs)

        res_idx = 0 # keeping track of reservoir index
        enc_idx = 0 # keeping track of encoder index
        y_enc_offset = self.layers[-2].num_units
        # custom dataset for reservoir
        dataset_res_X = np.copy(X)
        print(np.shape(dataset_res_X))
        for l in range(len(self.layers)-1):
            # encoder layer
            if (l+1) % 2 == 0:
                # the new data which we will fill
                dataset_res_X_new = np.zeros((np.shape(dataset_res_X)[0], self.layers[l].output_size))
                # fit the encoder
                #try:
                #if (self.reuse_encoder):
                #    self.layers[1].train(dataset_res_X)
                if (self.reuse_encoder and l == 1) or not self.reuse_encoder:
                    try:
                        self.layers[l].train(dataset_res_X)
                    except:
                        return False
                # move the data through it
                for idx,x in enumerate(dataset_res_X):
                    try:
                        if self.reuse_encoder:
                            y_p = self.layers[1].forward(x)
                        else:
                            y_p = self.layers[l].forward(x)
                    except:
                        return False
                    # update the dataset as we move through it
                    dataset_res_X_new[idx, :] = y_p

                    # add the data
                    #y_forward[y_idx, :] = y_p # 'predicted' output
                    #y_nonwarmup[y_idx, :] = y[idx, :] # true output
                    if idx < np.shape(X)[0] - warmup_timesteps:
                        y_forward[idx, y_enc_offset:(y_enc_offset+self.layers[l].output_size)] = y_p
                y_enc_offset += self.layers[l].output_size
                enc_idx += 1
            # reservoir layer
            else:
                # the new data which we will fill
                dataset_res_X_new = np.zeros((np.shape(dataset_res_X[warmup_interval:])[0], self.layers[l].num_units))
                # first warm-up
                for x in dataset_res_X[:warmup_interval]:
                    _ = self.layers[l].forward(x)
                # move rest of data through reservoir
                for idx,x in enumerate(dataset_res_X[warmup_interval:]):
                    y_p = self.layers[l].forward(x)
                    # update the dataset as we move through it
                    dataset_res_X_new[idx, :] = y_p
                    #y_forward[y_idx, :] = y_p
                    #y_nonwarmup[y_idx, :] = y[idx, :]
                    #y_idx += 1

                    # add the last reservoir to the output
                    if l == len(self.layers) - 2:
                        y_forward[idx, :self.layers[-2].num_units] = y_p
                res_idx += 1

            # update the dataset
            dataset_res_X = np.copy(dataset_res_X_new)
            print(np.shape(dataset_res_X))

        # update the true data by removing the warmup
        y_nonwarmup = y[warmup_timesteps:, :]

        # training stage
        # y_forward = np.zeros((np.shape(X[warmup_timesteps:])[0],
        #                       self.layers[-1].input_size))
        # for idx, x in enumerate(X[warmup_timesteps:]):
        #     # some function that allows us to display
        #     self.display()

        #     y_p = self.forward(x, len(self.layers)-1)
        #     y_forward[idx, :] = y_p

        # y_nonwarmup = y[warmup_timesteps:]

        # train the linear regression output
        self.layers[-1].train(y_forward, y_nonwarmup)

        return True

    def forward(self, x, end_layer=None):

        # if doing the full feedforward, override the last reservoir to append the input
        if end_layer is None or end_layer == len(self.layers):
            #y_p = super(DMEsnModel, self).forward(x=x, end_layer=len(self.layers)-1)
            y_p, e_d = self.forward_encoder(x=x, end_layer = len(self.layers)-1)
            y_p = np.hstack((y_p, e_d)) # add encoder data
            y_p = np.hstack((y_p, x))   # add input data
            y_p = self.layers[-1].forward(y_p)
        # if doing only the reservoir feedforward, append the input at the end
        else:
            #y_p = super(DEsnModel, self).forward(x=x, end_layer=end_layer)
            y_p, e_d = self.forward_encoder(x=x, end_layer = end_layer)
            if end_layer == len(self.layers)-1:
                y_p = np.hstack((y_p, e_d)) # add encoder data
                y_p = np.hstack((y_p, x))   # add input data

        return y_p
