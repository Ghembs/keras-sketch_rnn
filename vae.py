# -*- coding: utf-8 -*-
"""
Created on 20/02/2018
@author: Giuliano
"""
# vae

import numpy as np
from keras import metrics
import keras.backend as K
from keras.layers import subtract, multiply, Reshape
import dataloader as dl
import model as vae
import random


def get_mixture_coef(output):
    out_pi = output[:, :, :20]
    out_mu_x = output[:, :, 20:40]
    out_mu_y = output[:, :, 40:60]
    out_sigma_x = output[:, :, 60:80]
    out_sigma_y = output[:, :, 80:100]
    out_ro = output[:, :, 100:120]
    pen_logits = output[:, :, 120:123]
    # out_mu = K.reshape(out_mu, [-1, numComonents*2, outputDim])
    out_mu_x = K.permute_dimensions(out_mu_x, [0, 2, 1])
    out_mu_y = K.permute_dimensions(out_mu_y, [0, 2, 1])
    # use softmax to normalize pi and q into prob distribution
    max_pi = K.max(out_pi, axis = 1, keepdims=True)
    out_pi = out_pi - max_pi
    out_pi = K.exp(out_pi)
    normalize_pi = 1 / K.sum(out_pi, axis=1, keepdims=True)
    out_pi = normalize_pi * out_pi
    out_q = K.softmax(pen_logits)
    out_ro = K.tanh(out_ro)
    # use exponential to make sure sigma is positive
    out_sigma_x = K.exp(out_sigma_x)
    out_sigma_y = K.exp(out_sigma_y)
    return out_pi, out_mu_x, out_mu_y, out_sigma_x, out_sigma_y, out_ro, out_q


def tf_bi_normal(x, y, mu_x, mu_y, sigma_x, sigma_y, ro):
    norm1 = subtract([x, mu_x])
    norm2 = subtract([y, mu_y])
    norm1 = K.permute_dimensions(norm1, [0, 2, 1])
    norm2 = K.permute_dimensions(norm2, [0, 2, 1])
    sigma = multiply([sigma_x, sigma_y])
    z = (K.square(norm1 / (sigma_x + 1e-8)) + K.square(norm2 / (sigma_y + 1e-8)) - 2 *
         multiply([ro, norm1, norm2]) / (sigma + 1e-8) + 1e-8)
    ro_opp = 1 - K.square(ro)
    result = K.exp(-z / (2 * ro_opp + 1e-8))
    denom = 2 * np.pi * multiply([sigma, K.square(ro_opp)]) + 1e-8
    result = result / denom + 1e-8
    return result


# TODO sort dimensions out
def get_lossfunc(out_pi, out_mu_x, out_mu_y, out_sigma_x, out_sigma_y, out_ro, out_q, x, y, logits):
    # L_r loss term calculation, L_s part
    result = tf_bi_normal(x, y, out_mu_x, out_mu_y, out_sigma_x, out_sigma_y, out_ro)
    result = multiply([result, out_pi])
    result = K.permute_dimensions(result, [0, 2, 1])
    result = K.sum(result, axis=1, keepdims=True)
    result = -K.log(result + 1e-8)
    fs = 1.0 - logits[:, 2]
    fs = Reshape((-1, 1))(fs)
    result = multiply([result, fs])
    result = K.permute_dimensions(result, [0, 2, 1])
    # L_r loss term, L_p part
    result1 = K.categorical_crossentropy(logits, out_q, from_logits = True)
    result1 = Reshape((-1, 1))(result1)
    result = result + result1
    return K.mean(result)


def mdn_loss(x, y, pen, output):
    out_pi, out_mu_x, out_mu_y, out_sigma_x, out_sigma_y, out_ro, out_q = get_mixture_coef(
        output)
    L_r = get_lossfunc(out_pi, out_mu_x, out_mu_y, out_sigma_x,
                        out_sigma_y, out_ro, out_q, x, y, pen)
    return L_r


class Compiler:

    batch_size = 100
    original_dim = 250
    epochs = 3

    def __init__(self, names, generate = False):
        self.vae = vae.Vae(generate)
        if not generate:
            self.z_mean = self.vae.mean
            self.z_log_sigma = self.vae.log_sigma
        self.x_train = None
        self.x_valid = None
        self.x_test = None
        self.load_dataset(names)

    def load_dataset(self, names):
        for name in names:
            # TODO check validity of "bytes" encoding
            dataset = np.load("Dataset/" + name + ".full.npz", encoding = 'bytes')
            if self.x_train is None:
                self.x_train = dataset["train"][:2500]
                self.x_valid = dataset["valid"]
                self.x_test = dataset["test"]
            else:
                self.x_train = np.concatenate((self.x_train, dataset["train"][:2500]))
                self.x_valid = np.concatenate((self.x_valid, dataset["valid"]))
                self.x_test = np.concatenate((self.x_test, dataset["test"]))

        self.x_train = dl.DataLoader(self.x_train, self.batch_size)
        normal_scale_factor = self.x_train.calculate_normalizing_scale_factor()
        self.x_train.normalize(normal_scale_factor)

        self.x_valid = dl.DataLoader(self.x_valid, self.batch_size)
        self.x_valid.normalize(normal_scale_factor)

        self.x_test = dl.DataLoader(self.x_test, self.batch_size)
        self.x_test.normalize(normal_scale_factor)

    def vae_loss(self, x, x_decoded):
        val_x = x[:, :, 0]
        val_y = x[:, :, 1]
        pen = x[:, :, 2:]
        rec_loss = mdn_loss(val_x, val_y, pen, x_decoded)
        try:
            kl_loss = - 0.5 * K.mean(1 + self.z_log_sigma - K.square(self.z_mean) -
                                     K.exp(self.z_log_sigma), axis = -1)
            return rec_loss + 0.5 * kl_loss
        except AttributeError:
            print("no encoder")
            return rec_loss

    def set_batches(self):
        batches = None
        val_batches = None
        for i in range(50):
            a, b, c = self.x_train.get_batch(i)
            if batches is None:
                batches = b
            else:
                batches = np.append(batches, b, axis = 0)

        for i in range(10):
            a, b, c = self.x_valid.get_batch(i)
            if val_batches is None:
                val_batches = b
            else:
                val_batches = np.append(val_batches, b, axis = 0)

        return batches, val_batches

    def load_weights(self):
        try:
            self.vae.vae.load_weights("vae_model")
            print("Loaded weights from file")
        except IOError:
            print("Weights not found")

    def compile_fit(self):
        self.vae.vae.compile(loss = self.vae_loss, optimizer = 'adam',
                               metrics = ['accuracy'])

        self.load_weights()

        batches, val_batches = self.set_batches()

        print(batches.shape[:])
        print(val_batches.shape[:])

        self.vae.vae.fit(batches, batches, shuffle = False,
                           batch_size = self.batch_size, epochs = self.epochs,
                           validation_data = (val_batches, val_batches))
        self.vae.vae.save_weights("vae_model", True)


def adjust_temp(pi_pdf, temp):
    pi_pdf = np.log(pi_pdf) / temp
    pi_pdf -= pi_pdf.max()
    pi_pdf = np.exp(pi_pdf)
    pi_pdf /= pi_pdf.sum()
    return pi_pdf


def get_pi_idx(x, pdf, temp=1.0, greedy=False):
    """Samples from a pdf, optionally greedily."""
    if greedy:
        return np.argmax(pdf)
    pdf = adjust_temp(np.copy(pdf), temp)
    accumulate = 0
    for i in range(0, pdf.size):
        accumulate += pdf[i]
        if accumulate >= x:
            return i
    print('Error with sampling ensemble.')
    return -1


def sample_gaussian_2d(mu1, mu2, s1, s2, rho, temp=1.0, greedy=False):
    if greedy:
        return mu1, mu2
    mean = [mu1, mu2]
    s1 *= temp * temp
    s2 *= temp * temp
    cov = [[s1 * s1, rho * s1 * s2], [rho * s1 * s2, s2 * s2]]
    x = np.random.multivariate_normal(mean, cov, 1)
    return x[0][0], x[0][1]


def sample(model, seq_len=250, temperature=1.0, greedy_mode=False,
           z=None):
    """Samples a sequence from a pre-trained model."""

    prev_x = np.zeros((1, 1, 5), dtype=np.float32)
    prev_x[0, 0, 2] = 1
    if z is None:
        z = np.random.randn(1, 128)

    strokes = np.zeros((seq_len, 5), dtype=np.float32)
    mixture_params = []
    greedy = False
    temp = 1.0

    for i in range(seq_len):

        sample, h_out, c_out = model.predict([z, prev_x])

        params = get_mixture_coef(sample)
        [o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_pen] = params

        if i < 0:
            greedy = False
            temp = 1.0
        else:
            greedy = greedy_mode
            temp = temperature

        idx = get_pi_idx(random.random(), o_pi[0], temp, greedy)

        idx_eos = get_pi_idx(random.random(), o_pen[0], temp, greedy)
        eos = [0, 0, 0]
        eos[idx_eos] = 1

        next_x1, next_x2 = sample_gaussian_2d(o_mu1[0][idx], o_mu2[0][idx],
                                              o_sigma1[0][idx], o_sigma2[0][idx],
                                              o_corr[0][idx], np.sqrt(temp), greedy)

        strokes[i, :] = [next_x1, next_x2, eos[0], eos[1], eos[2]]

        params = [
            o_pi[0], o_mu1[0], o_mu2[0], o_sigma1[0], o_sigma2[0], o_corr[0],
            o_pen[0]
        ]

        mixture_params.append(params)

        prev_x = np.zeros((1, 1, 5), dtype=np.float32)
        prev_x[0][0] = np.array(
            [next_x1, next_x2, eos[0], eos[1], eos[2]], dtype=np.float32)

        model.h_out = h_out
        model.c_out = c_out

    return strokes, mixture_params


def draw():
    vae = Compiler(["cat", "flying_saucer"], True)
    vae.load_weights()

    stroke_, m = sample(vae.vae.vae, 250, 1, False, None)

    # sample = vae.x_train.random_sample()
    # sample = dl.to_big_strokes(sample)
    # samp = samp.reshape(1, samp.shape[0], samp.shape[1])
    # stroke_ = vae.vae.vae.predict(sample)
    print(stroke_)
    print(stroke_.shape[:])
    stroke_ = stroke_.reshape(stroke_.shape[1], stroke_.shape[2])

    stroke_ = dl.to_normal_strokes(stroke_)
    dl.draw_strokes(stroke_)


# model = Compiler(["cat", "flying_saucer"])
# model.compile_fit()
draw()
