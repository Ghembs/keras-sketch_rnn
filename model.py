# -*- coding: utf-8 -*-
"""
Created on 19/02/2018
@author: Giuliano
"""
# model

from keras import backend as K
from edward.models import Normal, Categorical, Mixture
import tensorflow as tf
from keras.layers import Input, Dense, Lambda, LSTM, Dropout, Bidirectional, Reshape
from keras.models import Model
from recurrentshop import RecurrentModel, LSTMCell


class Vae:
    epsilon_std = 1.0

    def __init__(self):
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.latent_dim = self.encoder.latent_dim

        out = self.encoder.model(self.encoder.input)
        out = Lambda(self.sampling)([out, self.encoder.log_sigma])
        out = self.decoder.model(out)

        self.model = Model(inputs = self.encoder.input, outputs = out)

        print(self.model.summary())

    def sampling(self, args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape = (self.latent_dim,), mean = 0.,
                                  stddev = self.epsilon_std)
        return z_mean + K.exp(z_log_sigma) * epsilon


class Encoder:

    original_dim = 250
    latent_dim = 128

    # input layer
    input = Input(shape = (original_dim, 5,))

    # encoding layers
    enc_1 = Bidirectional(LSTM(units = 5, return_sequences = True))
    enc_2 = Bidirectional(LSTM(units = 256))
    enc_mean = Dense(latent_dim, activation = 'relu')
    enc_log_sigma = Dense(latent_dim, activation = 'relu')

    def __init__(self):

        enc = self.enc_1(self.input)
        enc = self.enc_2(enc)
        self.mean = self.enc_mean(enc)
        self.log_sigma = self.enc_log_sigma(enc)

        self.model = Model(inputs = self.input, outputs = self.mean)


class Decoder:

    latent_dim = 128
    intermediate_dim = 512
    original_dim = 5

    # Input layers
    input = Input(shape = (latent_dim,))
    decoder_input = Input(shape = (1,))
    decoder_input_1 = Input(shape = (latent_dim,))
    decoder_input_2 = Input(shape = (intermediate_dim,))
    h_in = Input(shape = (latent_dim,))
    c_in = Input(shape = (latent_dim,))
    h_in_1 = Input(shape = (intermediate_dim,))
    c_in_1 = Input(shape = (intermediate_dim,))
    h_in_2 = Input(shape = (5,))
    c_in_2 = Input(shape = (5,))
    readout_in = Input(shape = (1,))
    readout_in_1 = Input(shape = (latent_dim,))
    readout_in_2 = Input(shape = (intermediate_dim,))

    # decoding layers
    dec_1 = Dense(latent_dim, activation = 'relu', input_dim = latent_dim)
    dec_2 = Reshape((latent_dim, 1,), input_shape = (latent_dim,))

    dec_3 = LSTMCell(latent_dim)
    dec_4 = LSTMCell(intermediate_dim)
    dec_5 = LSTMCell(original_dim)

    def __init__(self):
        dec_out, h, c = self.dec_3([self.decoder_input, self.h_in, self.c_in])

        self.rnn = RecurrentModel(input = self.decoder_input,
                                  initial_states = [self.h_in, self.c_in],
                                  output = dec_out, final_states = [h, c],
                                  readout_input = self.readout_in,
                                  return_sequences = True)

        dec_out_1, h_1, c_1 = self.dec_4([self.decoder_input_1, self.h_in_1,
                                          self.c_in_1])

        self.rnn_1 = RecurrentModel(input = self.decoder_input_1,
                                    initial_states = [self.h_in_1, self.c_in_1],
                                    output = dec_out_1, final_states = [h_1, c_1],
                                    readout_input = self.readout_in_1,
                                    return_sequences = True)

        dec_out_2, h_2, c_2 = self.dec_5([self.decoder_input_2, self.h_in_2,
                                          self.c_in_2])

        self.rnn_2 = RecurrentModel(input = self.decoder_input_2,
                                    initial_states = [self.h_in_2, self.c_in_2],
                                    output = dec_out_2, final_states = [h_2, c_2],
                                    readout_input = self.readout_in_2,
                                    return_sequences = True)

        self.model = self.set_decoder()

    def set_decoder(self):

        out = self.dec_1(self.input)
        out = self.dec_2(out)
        out = self.rnn(out)
        out = self.rnn_1(out)
        out = self.rnn_2(out)

        return Model(inputs = self.input, outputs = out)


class MixtureDensityNetwork:
    """
    Mixture density network for outputs y on inputs x.
    p((x,y), (z,theta))
    = sum_{k=1}^K pi_k(x; theta) Normal(y; mu_k(x; theta), sigma_k(x; theta))
    where pi, mu, sigma are the output of a neural network taking x
    as input and with parameters theta. There are no latent variables
    z, which are hidden variables we aim to be Bayesian about.
    """

    hidden = Decoder()

    def __init__(self, k):
        self.K = k  # here K is the amount of Mixtures

    def mapping(self, x):
        """pi, mu, sigma = NN(x; theta)"""
        hidden1 = Dense(15, activation='relu')(x)  # fully-connected layer with 15 hidden units
        hidden2 = Dense(15, activation='relu')(hidden1)
        self.mus = Dense(self.K)(hidden2)  # the means
        self.sigmas = Dense(self.K, activation=K.exp)(hidden2)  # the variance
        self.pi = Dense(self.K, activation=K.softmax)(hidden2)  # the mixture components

    def log_prob(self, xs, zs = None):
        """log p((xs,ys), (z,theta)) = sum_{n=1}^N log p((xs[n,:],ys[n]), theta)"""
        # Note there are no parameters we're being Bayesian about. The
        # parameters are baked into how we specify the neural networks.
        X, y = xs
        self.mapping(X)
        result = tf.exp(Normal.logpdf(y, self.mus, self.sigmas))
        result = tf.mul(result, self.pi)
        result = tf.reduce_sum(result, 1)
        result = tf.log(result)
        return tf.reduce_sum(result)
