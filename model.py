# -*- coding: utf-8 -*-
"""
Created on 19/02/2018
@author: Giuliano
"""
# model

from keras import backend as K
from keras.engine.topology import Layer
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
        self.inputDim = inputShape[2]
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
    max_len = 250
    input = Input((max_len, 5,))
    decoder_input = Input(shape = (133,))
    h_in = Input(shape = (512,))
    c_in = Input(shape = (512,))
    readout_in = Input(shape = (133,))
    enc_1 = Bidirectional(LSTM(256))
    enc_mean = Dense(128)
    enc_log_sigma = Dense(128)
    h_init = Dense(1024)
    dec_1 = LSTM(512)
    dec_3 = LSTMCell(512)
    mdn = MixtureDensity(5, 20)

    def __init__(self, generate = False):
        self.h_out = None
        self.c_out = None

        if not generate:
            self.encoder = self.build_encoder()
            self.mean, self.log_sigma = self.encoder(self.input)
            self.z = Lambda(self.sampling)([self.mean, self.log_sigma])
        else:
            self.z = Input(shape = (128,))

        if self.h_out is None:
            _h = self.h_init(self.z)

            _h = Reshape((512, 2,))(_h)

            _h_1 = Lambda(lambda x: x[:, :, 0])(_h)
            _c_1 = Lambda(lambda x: x[:, :, 1])(_h)
        else:
            _h_1 = self.h_out
            _c_1 = self.c_out

        z_ = Reshape((1, 128,))(self.z)
        z_ = Lambda(self.tile)(z_)

        z_ = concatenate([z_, self.input], axis = 2)

        dec_out, h, c = self.dec_3([self.decoder_input, self.h_in, self.c_in])

        # dec_out = self.mdn(dec_out)

        rnn = RecurrentModel(input = self.decoder_input,
                             initial_states = [self.h_in, self.c_in],
                             output = dec_out, final_states = [h, c],
                             return_sequences = True, return_states = True)

        out, self.h_out, self.c_out = rnn(z_, initial_state = [_h_1, _c_1])

        out = Reshape((-1, 512,))(out)

        out = self.mdn(out)

        if generate:
            self.vae = Model([self.z, self.input], out)
        else:
            self.vae = Model(self.input, out)

        print(self.vae.summary())

    def tile(self, tensor):
        return K.tile(tensor, [1, self.max_len, 1])

    def build_encoder(self):
        a = self.enc_1(self.input)
        mean = self.enc_mean(a)
        log_sigma = self.enc_log_sigma(a)
        encoder = Model(self.input, [mean, log_sigma])
        return encoder

    def sampling(self, args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape = (128,), mean = 0.,
                                  stddev = 1.0)
        return z_mean + K.exp(z_log_sigma) * epsilon
