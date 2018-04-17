# -*- coding: utf-8 -*-
"""
Created on 19/02/2018
@author: Giuliano
"""
# model

from keras import backend as K
from keras import optimizers
from keras.engine.topology import Layer
from keras.layers import Input, Dense, Lambda, LSTM, Bidirectional, Reshape, concatenate
from keras.models import Model
from recurrentshop import LSTMCell  # RecurrentModel
from dataloader import *


def sampling(args):
    z_mean, z_log_sigma = args
    epsilon = K.random_normal(shape = (128,), mean = 0.,
                              stddev = 1.0)
    return z_mean + K.exp(z_log_sigma/2) * epsilon


class MixtureDensity(Layer):
    def __init__(self, kerneldim, numcomponents, **kwargs):
        self.kernelDim = kerneldim
        self.numComponents = numcomponents
        super(MixtureDensity, self).__init__(**kwargs)

    def build(self, inputshape):
        self.inputDim = inputshape[2]
        self.outputDim = self.numComponents * (1 + self.kernelDim) + 3
        # self.Wo = K.variable(np.random.normal(scale=0.5, size = (self.inputDim, self.outputDim)),
        #                      name = 'W')
        # self.bo = K.variable(np.random.normal(scale=0.5, size = self.outputDim), name = 'b')

        self.Wo = self.add_weight(name = 'output_w', shape = (self.inputDim, self.outputDim),
                                  initializer = 'glorot_uniform')
        self.bo = self.add_weight(name = 'output_b', shape = (self.outputDim,),
                                  initializer = 'glorot_uniform')

        super(MixtureDensity, self).build(inputshape)

        # self.trainable_weights = [self.Wo, self.bo]

    def call(self, x, mask=None):
        output = K.dot(x, self.Wo) + self.bo
        return output

    def compute_output_shape(self, inputshape):
        return inputshape[0], inputshape[1], self.outputDim


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
    dec_1 = LSTM(512, return_sequences = True, name = 'Dec_RNN')
    dec_3 = LSTMCell(512)
    mdn = MixtureDensity(5, 20, name = 'mdn')
    kl_tolerance = 0.2
    kl_weight_start = 0.01
    kl_weight = 0.5
    learning_rate = 0.001
    decay_rate = 0.9999
    kl_decay_rate = 0.99995
    min_learning_rate = 0.00001

    def __init__(self, max_len = 250):
        self.curr_kl_weight = self.kl_weight_start
        self.max_len = max_len
        self.input = Input((max_len, 5,), name = "stroke_batch")
        self.output = Input((max_len, 5,), name = "stroke_target")

        self.build_model()

        print(self.model.summary())

    def tile(self, tensor):
        return K.tile(tensor, [1, self.max_len, 1])

    def build_model(self):
        # ====================== ENCODER =============================
        a = self.enc_1(self.output)
        self.mean = self.enc_mean(a)
        self.log_sigma = self.enc_log_sigma(a)

        self.kl_loss = - 0.5 * K.mean(1 + self.log_sigma - K.square(self.mean) -
                                      K.exp(self.log_sigma), axis = [0, 1])
        self.kl_loss = K.maximum(self.kl_loss, self.kl_tolerance)

        # encoded = concatenate([mean, log_sigma])
        self.encoder = Model(self.output, self.mean, name = 'encoder')

        # self.mean = Lambda(lambda x: x[:, :128])(encoded)
        # self.log_sigma = Lambda(lambda x: x[:, 128:])(encoded)
        self.z = Lambda(sampling)([self.mean, self.log_sigma])

        # ====================== VAE ==============================
        _h = self.h_init(self.z)

        _h_1 = Lambda(lambda x: x[:, :512])(_h)
        _c_1 = Lambda(lambda x: x[:, 512:])(_h)

        z_ = Reshape((1, 128,))(self.z)
        z_ = Lambda(self.tile)(z_)

        z_ = concatenate([z_, self.input], axis = 2)

        out = self.dec_1(z_, initial_state = [_h_1, _c_1])

        out = self.mdn(out)

        self.model = Model([self.output, self.input], out)

        # ====================== DECODER ===========================
        _h_ = self.h_init(self.dec_input)

        _h_1_ = Lambda(lambda x: x[:, :512])(_h_)
        _c_1_ = Lambda(lambda x: x[:, 512:])(_h_)

        _z_ = Reshape((1, 128,))(self.dec_input)
        _z_ = Lambda(self.tile)(_z_)

        _z_ = concatenate([_z_, self.input], axis = 2)

        out_ = self.dec_1(_z_, initial_state = [_h_1_, _c_1_])

        out_ = self.mdn(out_)

        self.decoder = Model([self.dec_input, self.input], out_)

        def get_mixture_coef(output):
            out_pi = output[:, :20]
            out_mu_x = output[:, 20:40]
            out_mu_y = output[:, 40:60]
            out_sigma_x = output[:, 60:80]
            out_sigma_y = output[:, 80:100]
            out_ro = output[:, 100:120]
            pen_logits = output[:, 120:123]

            # use softmax to normalize pi and q into prob distribution
            # max_pi = K.max(out_pi, axis = 1, keepdims=True)
            # out_pi = out_pi - max_pi
            # out_pi = K.exp(out_pi)
            # normalize_pi = 1 / (K.sum(out_pi, axis = 1, keepdims = True))
            # out_pi = normalize_pi * out_pi
            out_pi = K.softmax(out_pi)

            out_q = K.softmax(pen_logits)
            # max_q = K.max(pen_logits, axis = 1, keepdims = True)
            # out_q = pen_logits - max_q
            # out_q = K.exp(pen_logits)
            # normalize_q = 1 / (K.sum(out_q, axis = 1, keepdims = True))
            # out_q = normalize_q * out_q

            # sue tanh to normalize correlation coefficient
            out_ro = K.tanh(out_ro)

            # use exponential to make sure sigma is positive
            out_sigma_x = K.exp(out_sigma_x)
            out_sigma_y = K.exp(out_sigma_y)

            return out_pi, out_mu_x, out_mu_y, out_sigma_x, out_sigma_y, out_ro, pen_logits, out_q

        def tf_bi_normal(x, y, mu_x, mu_y, sigma_x, sigma_y, ro):
            x_ = K.reshape(x, (-1, 1))
            y_ = K.reshape(y, (-1, 1))
            norm1 = x_ - mu_x
            norm2 = y_ - mu_y
            sigma = sigma_x * sigma_y
            z = (K.square(norm1 / (sigma_x + 1e-8)) + K.square(norm2 / (sigma_y + 1e-8)) -
                 (2 * ro * norm1 * norm2) / (sigma + 1e-8) + 1e-8)
            ro_opp = 1 - K.square(ro)
            result = K.exp(-z / (2 * ro_opp + 1e-8))
            denom = 2 * np.pi * sigma * K.sqrt(ro_opp) + 1e-8
            result = result / denom + 1e-8
            return result

        def get_lossfunc(out_pi, out_mu_x, out_mu_y, out_sigma_x, out_sigma_y, out_ro, out_q, x, y,
                         logits):
            # L_r loss term calculation, L_s part
            result = tf_bi_normal(x, y, out_mu_x, out_mu_y, out_sigma_x, out_sigma_y, out_ro)
            result = result * out_pi
            result = K.sum(result, axis = 1, keepdims = True)
            result = -K.log(result + 1e-8)
            fs = 1.0 - logits[:, 2]
            fs = K.reshape(fs, (-1, 1))
            result = result * fs
            # L_r loss term, L_p part
            result1 = K.categorical_crossentropy(out_q, logits, from_logits = True)
            result1 = K.reshape(result1, (-1, 1))
            result = result + result1
            return K.mean(result, axis = [0, 1])

        output = K.reshape(out, [-1, 123])

        self.pi, self.mu_x, self.mu_y, self.sigma_x, self.sigma_y, self.ro, \
        self.logits, self.q = get_mixture_coef(output)

        target = K.reshape(self.output, [-1, 5])
        val_x = target[:, 0]
        val_y = target[:, 1]
        pen = target[:, 2:]

        self.rec_loss = get_lossfunc(self.pi, self.mu_x, self.mu_y, self.sigma_x, self.sigma_y,
                                     self.ro, self.logits, val_x, val_y, pen)

        self.loss = self.rec_loss + self.curr_kl_weight * self.kl_loss

        adam = optimizers.Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999)
        self.model.add_loss(self.loss)
        self.model.compile(optimizer = adam,
                           metrics = ['accuracy'])

    # def build_decoder(self):
        # dec_out, h, c = self.dec_3([self.decoder_input, self.h_in, self.c_in])

        # dec_out = self.mdn(dec_out)

        # rnn = RecurrentModel(input = self.decoder_input,
        #                      initial_states = [self.h_in, self.c_in],
        #                      output = dec_out, final_states = [h, c],
        #                      return_sequences = True, name = 'Dec_RNN')

        # rnn(z_, initial_state = [_h_1, _c_1])

    def update_params(self, step):
        curr_learning_rate = ((self.learning_rate - self.min_learning_rate) *
                              (self.decay_rate ** step) + self.min_learning_rate)
        self.curr_kl_weight = (self.kl_weight - (self.kl_weight - self.kl_weight_start) *
                               (self.kl_decay_rate ** step))
        K.set_value(self.model.optimizer.lr, curr_learning_rate)
