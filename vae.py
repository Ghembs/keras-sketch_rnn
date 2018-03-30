# -*- coding: utf-8 -*-
"""
Created on 20/02/2018
@author: Giuliano
"""
# vae

import numpy as np
# from keras import metrics
import keras.backend as K
from keras import optimizers
import dataloader as dl
import model as vae
import random

# TODO keep checking if better with max_pi, max_q, check with simple K.softmax too
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
    out_pi = K.exp(out_pi)
    normalize_pi = 1 / (K.sum(out_pi, axis=1, keepdims=True))
    out_pi = normalize_pi * out_pi
    # out_pi = K.softmax(out_pi)
    # out_q = K.softmax(pen_logits)
    # max_q = K.max(pen_logits, axis = 1, keepdims = True)
    # out_q = pen_logits - max_q
    out_q = K.exp(pen_logits)
    normalize_q = 1 / (K.sum(out_q, axis = 1, keepdims = True))
    out_q = normalize_q * out_q
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
    z = (K.square(norm1 / (sigma_x + 1e-8)) + K.square(norm2 / (sigma_y + 1e-8)) - (2 *
         ro * norm1 * norm2) / (sigma + 1e-8) + 1e-8)
    ro_opp = 1 - K.square(ro)
    result = K.exp(-z / (2 * ro_opp + 1e-8))
    denom = 2 * np.pi * sigma * K.square(ro_opp) + 1e-8
    result = result / denom + 1e-8
    return result


def get_lossfunc(out_pi, out_mu_x, out_mu_y, out_sigma_x, out_sigma_y, out_ro, out_q, x, y, logits):
    # L_r loss term calculation, L_s part
    result = tf_bi_normal(x, y, out_mu_x, out_mu_y, out_sigma_x, out_sigma_y, out_ro)
    result = result * out_pi
    # result = K.permute_dimensions(result, [0, 2, 1])
    result = K.sum(result, axis=1, keepdims=True)
    result = -K.log(result + 1e-8)
    fs = 1.0 - logits[:, 2]
    fs = K.reshape(fs, (-1, 1))
    result = result * fs
    # L_r loss term, L_p part
    result1 = K.categorical_crossentropy(out_q, logits, from_logits = True)
    result1 = K.reshape(result1, (-1, 1))
    result = result + result1
    return K.mean(result)


def mdn_loss(x, y, pen, output):
    out_pi, out_mu_x, out_mu_y, out_sigma_x, out_sigma_y, out_ro, out_q, o_pen = get_mixture_coef(
        output)
    L_r = get_lossfunc(out_pi, out_mu_x, out_mu_y, out_sigma_x,
                        out_sigma_y, out_ro, out_q, x, y, pen)
    return L_r


class Compiler:

    batch_size = 100
    original_dim = 250
    epochs = 5

    def __init__(self, names, generate = False, max_len = 250):
        self.generate = generate
        self.vae = vae.Vae(self.generate, max_len)
        self.x_train = None
        self.x_valid = None
        self.x_test = None
        self.load_dataset(names)

    def load_dataset(self, names):
        for name in names:
            dataset = np.load("Dataset/" + name + ".full.npz", encoding = 'bytes')
            if self.x_train is None:
                self.x_train = dataset["train"][:50000]
                self.x_valid = dataset["valid"]
                self.x_test = dataset["test"]
            else:
                self.x_train = np.concatenate((self.x_train, dataset["train"][:50000]))
                self.x_valid = np.concatenate((self.x_valid, dataset["valid"]))
                self.x_test = np.concatenate((self.x_test, dataset["test"]))

        self.x_train = dl.DataLoader(self.x_train, self.batch_size,
                                     random_scale_factor = 0.15, augment_stroke_prob = 0.1)
        normal_scale_factor = self.x_train.calculate_normalizing_scale_factor()
        self.x_train.normalize(normal_scale_factor)

        self.x_valid = dl.DataLoader(self.x_valid, self.batch_size)
        self.x_valid.normalize(normal_scale_factor)

        self.x_test = dl.DataLoader(self.x_test, self.batch_size)
        self.x_test.normalize(normal_scale_factor)

    def vae_loss(self, x, x_decoded):
        target = K.reshape(x, [-1, 5])
        val_x = target[:, 0]
        val_y = target[:, 1]
        pen = target[:, 2:]
        output = K.reshape(x_decoded, [-1, 123])
        rec_loss = mdn_loss(val_x, val_y, pen, output)
        try:
            kl_loss = - 0.5 * K.mean(1 + self.vae.log_sigma - K.square(self.vae.mean) -
                                     K.exp(self.vae.log_sigma), axis = -1)
            return rec_loss + 1 * kl_loss
        except AttributeError:
            print("no encoder")
            return rec_loss

    def set_batches(self):
        batches = None
        val_batches = None
        for i in range(1000):
            a, b, c = self.x_train.get_batch(i)
            if batches is None:
                batches = b
            else:
                batches = np.append(batches, b, axis = 0)

        for i in range(20):
            a, b, c = self.x_valid.get_batch(i)
            if val_batches is None:
                val_batches = b
            else:
                val_batches = np.append(val_batches, b, axis = 0)

        return batches, val_batches

    def load_weights(self):
        try:
            if not self.generate:
                self.vae.encoder.load_weights("vae_enc")
            self.vae.decoder.load_weights("vae_dec")
            print("Loaded weights from file")
        except IOError:
            print("Weights not found")

    # TODO check if useful to separate the losses, training on single batches and calculating them
    # TODO on the go
    def compile_fit(self):
        adam = optimizers.Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999,
                               decay = 0.999)
        self.vae.vae.compile(loss = self.vae_loss, optimizer = adam, metrics = ['accuracy'])

        self.load_weights()

        batches, val_batches = self.set_batches()

        # _batches = np.reshape(batches, [-1, 5])
        # _batches = _batches[:5000, :]
        # _val_batches = np.reshape(val_batches, [-1, 5])
        # _val_batches = _val_batches[:1000, :]
        # TODO check best shuffle value
        self.vae.vae.fit(batches, batches, batch_size = self.batch_size,
                         epochs = self.epochs, validation_data = (val_batches, val_batches))
        self.vae.encoder.save_weights("vae_enc", True)
        self.vae.decoder.save_weights("vae_dec", True)


def adjust_temp(pi_pdf, temp):
    pi_pdf = np.log(pi_pdf + 1e-8) / (temp + 1e-8)
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
    print(mean)
    s1 *= temp * temp
    s2 *= temp * temp
    cov = [[s1 * s1, rho * s1 * s2], [rho * s1 * s2, s2 * s2]]
    print(cov)
    x = np.random.multivariate_normal(mean, cov, 1)
    return x[0][0], x[0][1]


# TODO check why end of sketch comes too soon/randomly
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
        print(i)

        sample, h_out, c_out = model.predict([z, prev_x])

        sample = K.reshape(sample, [1, -1])

        o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, out_q, o_pen = get_mixture_coef(sample)

        if i < 0:
            greedy = False
            temp = 1.0
        else:
            greedy = greedy_mode
            temp = temperature

        idx = get_pi_idx(random.random(), K.eval(o_pi[0]), temp, greedy)

        idx_eos = get_pi_idx(random.random(), K.eval(o_pen[0]), temp, greedy)
        eos = [0, 0, 0]
        eos[idx_eos] = 1

        next_x1, next_x2 = sample_gaussian_2d(K.eval(o_mu1[0][idx]), K.eval(o_mu2[0][idx]),
                                              K.eval(o_sigma1[0][idx]), K.eval(o_sigma2[0][idx]),
                                              K.eval(o_corr[0][idx]), np.sqrt(temp), greedy)

        strokes[i, :] = [next_x1, next_x2, eos[0], eos[1], eos[2]]

        params = [
            o_pi[0], o_mu1[0], o_mu2[0], o_sigma1[0], o_sigma2[0], o_corr[0],
            o_pen[0]
        ]

        mixture_params.append(params)

        prev_x = np.zeros((1, 1, 5), dtype=np.float32)
        prev_x[0][0] = np.array(
            [next_x1, next_x2, eos[0], eos[1], eos[2]], dtype=np.float32)

    return strokes, mixture_params


def draw():
    vae = Compiler(["cat", "flying_saucer"], True, 1)
    vae.load_weights()

    stroke_, m = sample(vae.vae.decoder, 100, 1, False, None)

    # sample = vae.x_train.random_sample()
    # sample = dl.to_big_strokes(sample)
    # samp = samp.reshape(1, samp.shape[0], samp.shape[1])
    # stroke_ = vae.vae.vae.predict(sample)
    print(stroke_)
    print(stroke_.shape[:])
    # stroke_ = stroke_.reshape(stroke_.shape[0], stroke_.shape[1])

    stroke_ = dl.to_normal_strokes(stroke_)
    dl.draw_strokes(stroke_)


model = Compiler(["cat", "flying_saucer"])
model.compile_fit()
# draw()
