# -*- coding: utf-8 -*-
"""
Created on 19/02/2018
@author: Giuliano
"""
# model

from keras import backend as K
from keras.engine.topology import Layer
import tensorflow as tf
from keras.layers import Input, Dense, Lambda, LSTM, Dropout, Bidirectional, Reshape, concatenate
from keras.models import Model
from recurrentshop import RecurrentModel, LSTMCell
import numpy as np


class MixtureDensity(Layer):
    def __init__(self, kernelDim, numComponents, **kwargs):
        self.kernelDim = kernelDim
        self.numComponents = numComponents
        super(MixtureDensity, self).__init__(**kwargs)

    def build(self, inputShape):
        self.inputDim = inputShape[1]
        self.outputDim = self.numComponents * (1 + self.kernelDim) + 3
        self.Wo = K.variable(np.random.normal(scale=0.5, size = (self.inputDim, self.outputDim)))
        self.bo = K.variable(np.random.normal(scale=0.5, size = self.outputDim))

        self.trainable_weights = [self.Wo, self.bo]

    def call(self, x, mask=None):
        output = K.dot(x, self.Wo) + self.bo
        return output

    def compute_output_shape(self, inputShape):
        return inputShape[0], self.outputDim


class Vae:
    input = Input((250, 5,))
    decoder_input = Input(shape = (133,))
    h_in = Input(shape = (512,))
    c_in = Input(shape = (512,))
    readout_in = Input(shape = (133,))
    enc_1 = Bidirectional(LSTM(256))
    enc_mean = Dense(128)
    enc_log_sigma = Dense(128)
    h_init = Dense(1024)
    dec_1 = Dense(5)
    dec_2 = Dense(123)
    dec_3 = LSTMCell(512)
    mdn = MixtureDensity(5, 20)

    def __init__(self):
        a = self.enc_1(self.input)
        self.mean = self.enc_mean(a)
        self.log_sigma = self.enc_log_sigma(a)
        z = Lambda(self.sampling)([self.mean, self.log_sigma])
        _h = self.h_init(z)

        _h = Reshape((512, 2,))(_h)

        _h_1 = Lambda(lambda x: x[:, :, 0])(_h)
        _h_2 = Lambda(lambda x: x[:, :, 1])(_h)

        z_ = Reshape((1, 128,))(z)
        z_out = z_
        for i in range(249):
            z_out = concatenate([z_out, z_], axis = 1)

        z_out = concatenate([z_out, self.input], axis = 2)

        dec_out, h, c = self.dec_3([self.decoder_input, self.h_in, self.c_in])

        dec_out = self.mdn(dec_out)

        rnn = RecurrentModel(input = self.decoder_input,
                             initial_states = [self.h_in, self.c_in],
                             output = dec_out, final_states = [h, c],
                             readout_input = self.readout_in,
                             return_sequences = True)

        out = rnn(z_out, initial_state = [_h_1, _h_2])

        self.vae = Model(self.input, out)

    def sampling(self, args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape = (128,), mean = 0.,
                                  stddev = 1.0)
        return z_mean + K.exp(z_log_sigma) * epsilon
