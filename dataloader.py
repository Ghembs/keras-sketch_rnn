# -*- coding: utf-8 -*-
"""
Created on 12/12/17
@author: giuliano
"""
# dataloader

import os
import random
import tensorflow as tf
import numpy as np
import svgwrite
from IPython.display import SVG, display


def augment_strokes(strokes, prob=0.0):
    """Perform data augmentation by randomly dropping out strokes."""
    # drop each point within a line segments with a probability of prob
    # note that the logic in the loop prevents points at the ends to be dropped.
    result = []
    prev_stroke = [0, 0, 1]
    count = 0
    stroke = [0, 0, 1]  # Added to be safe.
    for i in range(len(strokes)):
        candidate = [strokes[i][0], strokes[i][1], strokes[i][2]]
        if candidate[2] == 1 or prev_stroke[2] == 1:
            count = 0
        else:
            count += 1
        urnd = np.random.rand()  # uniform random variable
        if candidate[2] == 0 and prev_stroke[2] == 0 and count > 2 and urnd < prob:
            stroke[0] += candidate[0]
            stroke[1] += candidate[1]
        else:
            stroke = candidate
            prev_stroke = stroke
            result.append(stroke)
    return np.array(result)


def get_bounds(data, factor=10):
    """Return bounds of data."""
    min_x = 0
    max_x = 0
    min_y = 0
    max_y = 0

    abs_x = 0
    abs_y = 0
    for i in range(len(data)):
        x = float(data[i, 0]) / factor
        y = float(data[i, 1]) / factor
        abs_x += x
        abs_y += y
        min_x = min(min_x, abs_x)
        min_y = min(min_y, abs_y)
        max_x = max(max_x, abs_x)
        max_y = max(max_y, abs_y)

    return min_x, max_x, min_y, max_y


# little function that displays vector images and saves them to .svg
def draw_strokes(data, factor=0.2, svg_filename = 'sample.svg'):
    tf.gfile.MakeDirs(os.path.dirname(svg_filename))
    min_x, max_x, min_y, max_y = get_bounds(data, factor)
    dims = (50 + max_x - min_x, 50 + max_y - min_y)
    dwg = svgwrite.Drawing(svg_filename, size=dims)
    dwg.add(dwg.rect(insert=(0, 0), size=dims, fill='white'))
    lift_pen = 1
    abs_x = 25 - min_x
    abs_y = 25 - min_y
    p = "M%s,%s " % (abs_x, abs_y)
    command = "m"
    for i in range(len(data)):
        if lift_pen == 1:
            command = "m"
        elif command != "l":
            command = "l"
        else:
            command = ""
        x = float(data[i, 0])/factor
        y = float(data[i, 1])/factor
        lift_pen = data[i, 2]
        p += command+str(x)+","+str(y)+" "
    the_color = "black"
    stroke_width = 1
    dwg.add(dwg.path(p).stroke(the_color, stroke_width).fill("none"))
    dwg.save()
    display(SVG(dwg.tostring()))


# generate a 2D grid of many vector drawings
def make_grid_svg(s_list, grid_space=10.0, grid_space_x=16.0):
  def get_start_and_end(x):
    x = np.array(x)
    print(x)
    x = x[:, 0:2]
    x_start = x[0]
    x_end = x.sum(axis=0)
    x = x.cumsum(axis=0)
    x_max = x.max(axis=0)
    x_min = x.min(axis=0)
    center_loc = (x_max+x_min)*0.5
    return x_start-center_loc, x_end
  x_pos = 0.0
  y_pos = 0.0
  result = [[x_pos, y_pos, 1]]
  for sample in s_list:
    s = sample[0]
    grid_loc = sample[1]
    grid_y = grid_loc[0]*grid_space+grid_space*0.5
    grid_x = grid_loc[1]*grid_space_x+grid_space_x*0.5
    start_loc, delta_pos = get_start_and_end([s])

    loc_x = start_loc[0]
    loc_y = start_loc[1]
    new_x_pos = grid_x+loc_x
    new_y_pos = grid_y+loc_y
    result.append([new_x_pos-x_pos, new_y_pos-y_pos, 0])

    result += s.tolist()
    result[-1][2] = 1
    x_pos = new_x_pos+delta_pos[0]
    y_pos = new_y_pos+delta_pos[1]
  return np.array(result)


def to_big_strokes(stroke, max_len=250):
    """Converts from stroke-3 to stroke-5 format and pads to given length."""
    # (But does not insert special start token).

    result = np.zeros((max_len, 5), dtype=float)
    l = len(stroke)
    assert l <= max_len
    result[0:l, 0:2] = stroke[:, 0:2]
    result[0:l, 3] = stroke[:, 2]
    result[0:l, 2] = 1 - result[0:l, 3]
    result[l:, 4] = 1
    return result


def to_normal_strokes(big_stroke):
    """Convert from stroke-5 format (from sketch-rnn paper) back to stroke-3."""
    l = 0
    for i in range(len(big_stroke)):
        if np.any(big_stroke[i, 4] > 0):  # TODO check this np.any out
            l = i
            break
    if l == 0:
        l = len(big_stroke)
    result = np.zeros((l, 3))
    result[:, 0:2] = big_stroke[0:l, 0:2]
    result[:, 2] = big_stroke[0:l, 3]
    return result


class DataLoader(object):
    """Class for loading data."""

    def __init__(self, strokes, batch_size=100, max_seq_length=250,
                 scale_factor=1.0, random_scale_factor=0.0,
                 augment_stroke_prob=0.0, limit=1000):
        self.batch_size = batch_size  # minibatch size
        self.max_seq_length = max_seq_length  # N_max in sketch-rnn paper
        self.scale_factor = scale_factor  # divide offsets by this factor
        self.random_scale_factor = random_scale_factor  # data augmentation method
        # Removes large gaps in the data. x and y offsets are clamped to have
        # absolute value no greater than this limit.
        self.limit = limit
        self.augment_stroke_prob = augment_stroke_prob  # data augmentation method
        self.start_stroke_token = [0, 0, 1, 0, 0]  # S_0 in sketch-rnn paper
        # sets self.strokes (list of ndarrays, one per sketch, in stroke-3 format,
        # sorted by size)
        self.strokes = []
        self.num_batches = self.preprocess(strokes)

    def random_sample(self):
        """Return a random sample, in stroke-3 format as used by draw_strokes."""
        sample = np.copy(random.choice(self.strokes))
        return sample

    def normalize(self, scale_factor=None):
        """Normalize entire dataset (delta_x, delta_y) by the scaling factor."""
        if scale_factor is None:
            scale_factor = self.calculate_normalizing_scale_factor()
        self.scale_factor = scale_factor
        for i in range(len(self.strokes)):
            self.strokes[i][:, 0:2] /= self.scale_factor

    def random_batch(self):
        """Return a randomised portion of the training data."""
        idx = np.random.permutation(range(0, len(self.strokes)))[0:self.batch_size]
        return self._get_batch_from_indices(idx)

    def get_batch(self, idx):
        """Get the idx'th batch from the dataset."""
        assert idx >= 0, "idx must be non negative"
        assert idx < self.num_batches, "idx must be less than the number of batches"
        start_idx = idx * self.batch_size
        indices = range(start_idx, start_idx + self.batch_size)
        return self._get_batch_from_indices(indices)

    # ========================= INTERNAL USE ==================================

    def preprocess(self, strokes):
        """Remove entries from strokes having > max_seq_length points."""
        raw_data = []
        seq_len = []
        count_data = 0

        for i in range(len(strokes)):
            data = strokes[i]
            if len(data) <= self.max_seq_length:
                count_data += 1
                # removes large gaps from the data
                data = np.minimum(data, self.limit)
                data = np.maximum(data, -self.limit)
                data = np.array(data, dtype=np.float32)
                data[:, 0:2] /= self.scale_factor
                raw_data.append(data)
                seq_len.append(len(data))
        seq_len = np.array(seq_len)  # nstrokes for each sketch
        idx = np.argsort(seq_len)
        for i in range(len(seq_len)):
            self.strokes.append(raw_data[idx[i]])
        print("total images <= max_seq_len is %d" % count_data)
        return int(count_data / self.batch_size)

    def pad_batch(self, batch, max_len):
        """Pad the batch to be stroke-5 bigger format as described in paper."""
        result = np.zeros((self.batch_size, max_len + 1, 5), dtype=float)
        assert len(batch) == self.batch_size
        for i in range(self.batch_size):
            length = len(batch[i])
            assert length <= max_len
            result[i, 0:length, 0:2] = batch[i][:, 0:2]
            result[i, 0:length, 3] = batch[i][:, 2]
            result[i, 0:length, 2] = 1 - result[i, 0:length, 3]
            result[i, length:, 4] = 1
            # put in the first token, as described in sketch-rnn methodology
            result[i, 1:, :] = result[i, :-1, :]
            result[i, 0, :] = 0
            result[i, 0, 2] = self.start_stroke_token[2]  # setting S_0 from paper.
            result[i, 0, 3] = self.start_stroke_token[3]
            result[i, 0, 4] = self.start_stroke_token[4]
        return result

    def random_scale(self, data):
        """Augment data by stretching x and y axis randomly [1-e, 1+e]."""
        x_scale_factor = (
                             np.random.random() - 0.5) * 2 * self.random_scale_factor + 1.0
        y_scale_factor = (
                             np.random.random() - 0.5) * 2 * self.random_scale_factor + 1.0
        result = np.copy(data)
        result[:, 0] *= x_scale_factor
        result[:, 1] *= y_scale_factor
        return result

    def _get_batch_from_indices(self, indices):
        """Given a list of indices, return the potentially augmented batch."""
        x_batch = []
        seq_len = []
        for idx in range(len(indices)):
            i = indices[idx]
            data = self.random_scale(self.strokes[i])
            data_copy = np.copy(data)
            if self.augment_stroke_prob > 0:
                data_copy = augment_strokes(data_copy, self.augment_stroke_prob)
            x_batch.append(data_copy)
            length = len(data_copy)
            seq_len.append(length)
        seq_len = np.array(seq_len, dtype=int)
        # We return three things: stroke-3 format, stroke-5 format, list of seq_len.
        return x_batch, self.pad_batch(x_batch, self.max_seq_length), seq_len

    def calculate_normalizing_scale_factor(self):
        """Calculate the normalizing factor explained in appendix of sketch-rnn."""
        data = []
        for i in range(len(self.strokes)):
            if len(self.strokes[i]) > self.max_seq_length:
                continue
            for j in range(len(self.strokes[i])):
                data.append(self.strokes[i][j, 0])
                data.append(self.strokes[i][j, 1])
        data = np.array(data)
        return np.std(data)
