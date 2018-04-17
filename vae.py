# -*- coding: utf-8 -*-
"""
Created on 20/02/2018
@author: Giuliano
"""
# vae

from model import *
from dataloader import *
from keras import callbacks


# TODO keep checking if better with max_pi, max_q, check with simple K.softmax too

def get_max_len(strokes):
    """Return the maximum length of an array of strokes."""
    max_len = 0
    for stroke in strokes:
        ml = len(stroke)
        if ml > max_len:
            max_len = ml
    return max_len


class Compiler:

    batch_size = 100
    max_seq_len = 250
    epochs = 5

    def __init__(self, names, generate = False):
        self.x_train = None
        self.x_valid = None
        self.x_test = None

        self.board = callbacks.TensorBoard(log_dir = 'log', histogram_freq = 0.5,
                                           batch_size = 100, write_graph = True,
                                           write_grads = True, write_images = True,
                                           embeddings_freq = 1,
                                           embeddings_layer_names = ['BiDir',
                                                                     'mean',
                                                                     'log_sigma',
                                                                     'state_init',
                                                                     'Dec_RNN',
                                                                     'mdn'])
        self.load_dataset(names)

        if generate:
            self.net = Vae(max_len = 1)
        else:
            self.net = Vae(max_len = self.max_seq_len)

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

        all_strokes = np.concatenate((self.x_train, self.x_valid, self.x_test))

        # calculate the max strokes we need.
        self.max_seq_len = get_max_len(all_strokes)

        self.x_train = DataLoader(self.x_train, self.batch_size,
                                  random_scale_factor = 0.15, augment_stroke_prob = 0.1,
                                  max_seq_length = self.max_seq_len)
        normal_scale_factor = self.x_train.calculate_normalizing_scale_factor()
        self.x_train.normalize(normal_scale_factor)

        self.x_valid = DataLoader(self.x_valid, self.batch_size, max_seq_length = self.max_seq_len)
        self.x_valid.normalize(normal_scale_factor)

        self.x_test = DataLoader(self.x_test, self.batch_size, max_seq_length = self.max_seq_len)
        self.x_test.normalize(normal_scale_factor)

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
            self.net.encoder.load_weights("weights/vae_enc")
            self.net.decoder.load_weights("weights/vae_dec")
            print("Loaded weights from file")
        except IOError:
            print("Weights not found")

    def compile_fit(self):
        self.load_weights()

        batches, val_batches = self.set_batches()

        target = batches[:, 1:self.max_seq_len+1, :]
        inp = batches[:, :self.max_seq_len, :]

        val_target = val_batches[:, 1:self.max_seq_len+1, :]
        val_inp = val_batches[:, :self.max_seq_len, :]

        # enc = np.zeros(shape = (batches.shape[0], 256))
        # val_enc = np.zeros(shape = (val_batches.shape[0], 256))

        self.net.model.fit([target, inp],  # target,
                           batch_size = self.batch_size, epochs = self.epochs,
                           validation_data = ([val_target, val_inp], None))
        self.net.encoder.save_weights("weights/vae_enc", True)
        self.net.decoder.save_weights("weights/vae_dec", True)

    def fit_eacn_batch(self):
        self.load_weights()

        for step in range(1000000):
            a, b, c = self.x_train.random_batch()
            # a_, b_, c_ = self.x_valid.random_batch()
            loss = self.net.model.train_on_batch([b[:, 1:self.max_seq_len+1, :],
                                                 b[:, :self.max_seq_len, :]])  #,
                                                 # b[:, 1:self.max_seq_len+1, :])
            # loss = self.vae.vae.fit([b[:, :self.max_seq_len, :], b[:, 1:self.max_seq_len+1, :]],
            #                         [b[:, :self.max_seq_len, :], np.zeros(shape=(100, 256))],
            #                         verbose = 0,
            #                         batch_size = self.batch_size, epochs = 1,
            #                         validation_data = ([b_[:, :self.max_seq_len, :],
            #                                             b_[:, 1:self.max_seq_len+1, :]],
            #                                            [b_[:, 1:self.max_seq_len+1, :],
            #                                             np.zeros(shape=(100, 256))]))
            if step % 20 == 0:
                # rec_loss: {:.5f}, kl_loss: {:.4f}, self.rec_loss, self.kl_loss,
                print("Step: {:d}, total_loss: {:.5f},"
                      " acc: {:.5f}, lr: {:.6f}, kl_weight: "
                      "{:.4f}".format(step, loss[0],
                                      loss[1], K.get_value(self.net.model.optimizer.lr),
                                      self.net.curr_kl_weight))
                # print("Step: {:d}, total_loss: {:.5f}, rec_loss: {:.5f}, "
                #       "kl_loss: {:.4f}, acc: {:.5f}, val_loss: {:.5f}, "
                #       "lr: {:.5f}, kl_weight: {:.4f}".format(step, loss.history['loss'][-1],
                #                                              loss.history['decoder_loss'][-1],
                #                                              loss.history['encoder_loss'][-1],
                #                                              loss.history['decoder_acc'][-1],
                #                                              loss.history['encoder_loss'][-1],
                #                                              K.get_value(self.vae.vae.optimizer.lr),
                #                                              self.vae.vae.loss_weights[1]))
            if step % 500 == 0 and step > 0:
                self.net.encoder.save_weights("weights/vae_enc", True)
                self.net.decoder.save_weights("weights/vae_dec", True)

            self.net.update_params(step)


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
def sample(network, seq_len=250, temperature=1.0, greedy_mode=False,
           z=None):
    """Samples a sequence from a pre-trained model."""

    prev_x = np.zeros((1, 1, 5), dtype=np.float32)
    prev_x[0, 0, 2] = 1
    if z is None:
        z = np.random.randn(1, 128)

    strokes = np.zeros((seq_len, 5), dtype=np.float32)
    mixture_params = []
    # greedy = False
    # temp = 1.0

    for i in range(seq_len):
        print(i)

        samp = network.decoder.predict([z, prev_x])

        samp = K.reshape(samp, [1, -1])

        o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, out_q, o_pen = get_mixture_coef(samp)

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


def draw(conditional = True):
    decoder = Compiler(["cat", "flying_saucer"], True)
    decoder.load_weights()

    z = None
    if conditional:
        encoder = Compiler(["cat", "flying_saucer"])
        encoder.load_weights()
        stroke = encoder.x_train.random_sample()
        stroke = to_big_strokes(stroke, max_len = 129)
        stroke = np.reshape(stroke, [1, stroke.shape[0], stroke.shape[1]])
        z = encoder.net.encoder.predict(stroke)

    stroke_, m = sample(decoder.net, 50, 0.1, False, z)

    print(stroke_)
    print(stroke_.shape[:])
    stroke_ = to_normal_strokes(stroke_)
    print(stroke_)
    draw_strokes(stroke_)


model = Compiler(["cat", "flying_saucer"])
model.compile_fit()
draw(True)

