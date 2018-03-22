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
import math


class MixtureDensity(Layer):
    def __init__(self, kernelDim, numComponents, **kwargs):
        # self.hiddenDim = 24
        self.kernelDim = kernelDim
        self.numComponents = numComponents
        super(MixtureDensity, self).__init__(**kwargs)

    def build(self, inputShape):
        self.inputDim = inputShape[2]
        self.outputDim = self.numComponents * (1 + self.kernelDim) + 3
        # self.Wh = K.variable(np.random.normal(scale=0.5, size = (self.inputDim, self.hiddenDim)))
        # self.bh = K.variable(np.random.normal(scale=0.5, size = self.hiddenDim))
        self.Wo = K.variable(np.random.normal(scale=0.5, size = (self.inputDim, self.outputDim)))
        self.bo = K.variable(np.random.normal(scale=0.5, size = self.outputDim))

        self.trainable_weights = [self.Wo, self.bo]  # [self.Wh, self.bh, self.Wo, self.bo]

    def call(self, x, mask=None):
        print x.shape[:]
        # hidden = K.tanh(K.dot(x, self.Wh) + self.bh)
        print self.Wo.shape[:]
        print self.bo.shape[:]
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

        rnn = RecurrentModel(input = self.decoder_input,
                             initial_states = [self.h_in, self.c_in],
                             output = dec_out, final_states = [h, c],
                             readout_input = self.readout_in,
                             return_sequences = True)

        out = rnn(z_out, initial_state = [_h_1, _h_2])

        # out = self.dec_2(out)
        out = self.mdn(out)

        self.vae = Model(self.input, out)

        print self.vae.summary()

    def sampling(self, args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape = (128,), mean = 0.,
                                  stddev = 1.0)
        return z_mean + K.exp(z_log_sigma) * epsilon


# def get_mixture_coef(output, numComonents = 24, outputDim = 1):
#     out_pi = output[:, :numComonents]
#     out_sigma = output[:, numComonents:2*numComonents]
#     out_mu = output[:, 2*numComonents:]
#     out_mu = K.reshape(out_mu, [-1, numComonents, outputDim])
#     out_mu = K.permute_dimensions(out_mu, [1, 0, 2])
#     # use softmax to normalize pi into prob distribution
#     max_pi = K.max(out_pi, axis = 1, keepdims=True)
#     out_pi = out_pi - max_pi
#     out_pi = K.exp(out_pi)
#     normalize_pi = 1 / K.sum(out_pi, axis=1, keepdims=True)
#     out_pi = normalize_pi * out_pi
#     # use exponential to make sure sigma is positive
#     out_sigma = K.exp(out_sigma)
#     return out_pi, out_sigma, out_mu
#
#
# def tf_normal(y, mu, sigma):
#     oneDivSqrtTwoPI = 1 / math.sqrt(2*math.pi)
#     result = y - mu
#     result = K.permute_dimensions(result, [2, 1, 0])
#     result = result * (1 / (sigma + 1e-8))
#     result = -K.square(result)/2
#     result = K.exp(result) * (1/(sigma + 1e-8))*oneDivSqrtTwoPI
#     result = K.prod(result, axis=[0])
#     return result
#
#
# def get_lossfunc(out_pi, out_sigma, out_mu, y):
#     result = tf_normal(y, out_mu, out_sigma)
#     result = result * out_pi
#     result = K.sum(result, axis=1, keepdims=True)
#     result = -K.log(result + 1e-8)
#     return K.mean(result)
#
#
# def mdn_loss(numComponents = 24, outputDim = 1):
#     def loss(y, output):
#         out_pi, out_sigma, out_mu = get_mixture_coef(output, numComponents, outputDim)
#         return get_lossfunc(out_pi, out_sigma, out_mu, y)
#     return loss

cacca = Vae()
