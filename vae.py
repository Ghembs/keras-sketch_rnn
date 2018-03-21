# -*- coding: utf-8 -*-
"""
Created on 20/02/2018
@author: Giuliano
"""
# vae

import numpy as np
from keras import metrics
import keras.backend as K
import dataloader as dl
import model as vae


class Compiler:

    batch_size = 100
    original_dim = 250
    epochs = 3

    def __init__(self, names):
        self.vae = vae.Vae()
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

    def vae_loss(self, x, x_decoded_mean):
        rec_loss = self.original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.sum(1 + self.z_log_sigma - K.square(self.z_mean) -
                                K.exp(self.z_log_sigma), axis = -1)
        return K.mean(rec_loss + kl_loss)

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
        # TODO fit the proper loss function
        self.vae.vae.compile(loss = 'categorical_crossentropy', optimizer = 'rmsprop',
                               metrics = ['accuracy'])

        self.load_weights()

        batches, val_batches = self.set_batches()

        print(batches.shape[:])
        print(val_batches.shape[:])

        # y = np.empty((5000, 128, 5))
        # val_y = np.empty((1000, 128, 5))

        # for i in range(5000):
        #     y[i] = batches[i][:128]
        #
        #     if i < 1000:
        #         val_y[i] = val_batches[i][:128]

        self.vae.vae.fit(batches, batches, shuffle = False,
                           batch_size = self.batch_size, epochs = self.epochs,
                           validation_data = (val_batches, val_batches))
        self.vae.vae.save_weights("vae_model", True)


def draw(vae):
    vae.load_weights()

    stroke = np.random.randn(128)
    stroke = stroke.reshape(1, stroke.shape[0])
    stroke_ = vae.vae.vae.predict(stroke)
    print(stroke_)
    stroke_ = stroke_.reshape(stroke_.shape[1], stroke_.shape[2])

    stroke_ = dl.to_normal_strokes(stroke_)
    dl.draw_strokes(stroke_)


def predict(classifier):
    classifier.load_weights()
    cose = None
    for i in range(10):
        cfier = classifier.x_test.random_sample()
        cfiera = dl.to_big_strokes(cfier)
        dl.draw_strokes(cfier, svg_filename = "stroke_" + str(i) + ".svg")
        cfiera = cfiera.reshape(1, 250 * 5)
        if cose is None:
            cose = cfiera
        else:
            cose = np.append(cose, cfiera, axis = 0)
    print(classifier.model.predict(cose))


model = Compiler(["cat", "flying_saucer"])
model.compile_fit()
draw(model)
