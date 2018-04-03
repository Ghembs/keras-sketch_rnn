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
from recurrentshop import LSTMCell, RecurrentModel
import numpy as np


class MixtureDensity(Layer):
    def __init__(self, kernelDim, numComponents, **kwargs):
        self.kernelDim = kernelDim
        self.numComponents = numComponents
        super(MixtureDensity, self).__init__(**kwargs)

    def build(self, inputShape):
        self.inputDim = inputShape[2]
        self.outputDim = self.numComponents * (1 + self.kernelDim) + 3
        self.Wo = K.variable(np.random.normal(scale=0.5, size = (self.inputDim, self.outputDim)),
                             name = 'W')
        self.bo = K.variable(np.random.normal(scale=0.5, size = self.outputDim), name = 'b')

        super(MixtureDensity, self).build(inputShape)

        self.trainable_weights = [self.Wo, self.bo]

    def call(self, x, mask=None):
        output = K.dot(x, self.Wo) + self.bo
        return output

    def compute_output_shape(self, inputShape):
        return inputShape[0], inputShape[1], self.outputDim


# TODO check activation functions
class Vae:
    dec_input = Input(shape = (128,))
    decoder_input = Input(shape = (133,))
    h_in = Input(shape = (512,))
    c_in = Input(shape = (512,))
    readout_in = Input(shape = (133,))
    enc_1 = Bidirectional(LSTM(256, name = 'Enc_RNN'), name = 'BiDir')
    enc_mean = Dense(128, name = 'mean')
    enc_log_sigma = Dense(128, name = 'log_sigma')
    h_init = Dense(1024, activation = 'tanh', name = 'state_init')
    dec_1 = LSTM(512)
    dec_3 = LSTMCell(512)
    mdn = MixtureDensity(5, 20)

    def __init__(self, generate = False, max_len = 250):
        self.generate = generate
        self.max_len = max_len
        self.input = Input((max_len, 5,), name = "stroke_batch")

        self.decoder = self.build_decoder()

        if not self.generate:
            self.encoder = self.build_encoder()
            encoded = self.encoder(self.input)
            self.mean = Lambda(lambda x: x[:, :128])(encoded)
            self.log_sigma = Lambda(lambda x: x[:, 128:])(encoded)
            self.z = Lambda(self.sampling)([self.mean, self.log_sigma])
            out = self.decoder([self.z, self.input])
            self.vae = Model(self.input, [out, encoded])

            print(self.vae.summary())

    def tile(self, tensor):
        return K.tile(tensor, [1, self.max_len, 1])

    def build_encoder(self):
        a = self.enc_1(self.input)
        mean = self.enc_mean(a)
        log_sigma = self.enc_log_sigma(a)
        out = concatenate([mean, log_sigma])
        encoder = Model(self.input, out, name = 'encoder')
        return encoder

    def build_decoder(self):
        _h = self.h_init(self.dec_input)

        _h = Reshape((512, 2,))(_h)

        _h_1 = Lambda(lambda x: x[:, :, 0])(_h)
        _c_1 = Lambda(lambda x: x[:, :, 1])(_h)

        z_ = Reshape((1, 128,))(self.dec_input)
        z_ = Lambda(self.tile)(z_)

        z_ = concatenate([z_, self.input], axis = 2)

        dec_out, h, c = self.dec_3([self.decoder_input, self.h_in, self.c_in])

        # dec_out = self.mdn(dec_out)

        rnn = RecurrentModel(input = self.decoder_input,
                             initial_states = [self.h_in, self.c_in],
                             output = dec_out, final_states = [h, c],
                             return_sequences = True, name = 'Dec_RNN')

        out = rnn(z_, initial_state = [_h_1, _c_1])

        out = self.mdn(out)

        decoder = Model([self.dec_input, self.input], out, name = 'decoder')

        return decoder

    def sampling(self, args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape = (128,), mean = 0.,
                                  stddev = 1.0)
        return z_mean + K.exp(z_log_sigma) * epsilon
