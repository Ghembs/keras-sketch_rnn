# -*- coding: utf-8 -*-
"""
Created on 03/04/2018
@author: Giuliano
Tool to test the sketch-rnn experiments made for the thesis: "reproduction and analysis
of a generative model for vector images creation"
https://github.com/Ghembs/sketch_thesis
Code adapted from:
https://github.com/tensorflow/magenta-demos/blob/master/jupyter-notebooks/Sketch_RNN.ipynb
"""
# test

# import the required libraries
from six.moves import xrange

# libraries required for visualisation:
from IPython.display import SVG, display
import svgwrite
# import our command line tools
from magenta.models.sketch_rnn.sketch_rnn_train import *
from magenta.models.sketch_rnn.model import *
from magenta.models.sketch_rnn.utils import *
from magenta.models.sketch_rnn.rnn import *

# set numpy output to something sensible
np.set_printoptions(precision=8, edgeitems=6, linewidth=200, suppress=True)
path = 'D:/Documenti/UNI/magenta/magenta/models/sketch_rnn/'


# =========================== UTILITIES ======================================
# little function that displays vector images and saves them to .svg
def draw_strokes(data, factor=0.2, svg_filename = path + 'sample.svg'):
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
    for i in xrange(len(data)):
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
    for elem in s_list:
        s = elem[0]
        grid_loc = elem[1]
        grid_y = grid_loc[0]*grid_space+grid_space*0.5
        grid_x = grid_loc[1]*grid_space_x+grid_space_x*0.5
        start_loc, delta_pos = get_start_and_end(s)

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


def interpolation(z0, z1, spherical = True, n = 10):
    z_list = []
    for t in np.linspace(0, 1, n):
        if spherical:
            z_list.append(slerp(z0, z1, t))
        else:
            z_list.append(lerp(z0, z1, t))
    return z_list


# ==================================== CLASS ======================================
class Tester:

    def __init__(self, data, model, conditional = True):

        self.cond = conditional
        data_dir = path + data
        model_dir = path + model
        if conditional:
            [self.train_set, self.valid_set, self.test_set, hps_model, eval_hps_model,
             sample_hps_model] = load_env(data_dir, model_dir)
        else:
            [hps_model, eval_hps_model, sample_hps_model] = load_model(model_dir)

        # construct the sketch-rnn model here:
        reset_graph()
        self.model = Model(hps_model)
        self.eval_model = Model(eval_hps_model, reuse = True)
        self.sample_model = Model(sample_hps_model, reuse = True)

        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

        # loads the weights from checkpoint into our model
        load_checkpoint(self.sess, model_dir)

    def encode(self, input_strokes, seq_len):
        strokes = to_big_strokes(input_strokes).tolist()
        strokes.insert(0, [0, 0, 1, 0, 0])
        strokes = strokes[:seq_len]
        seq_len = [len(input_strokes)]
        draw_strokes(to_normal_strokes(np.array(strokes)))
        return self.sess.run(self.eval_model.batch_z,
                             feed_dict={self.eval_model.input_data: [strokes],
                                        self.eval_model.sequence_lengths: seq_len})[0]

    def decode(self, z_input=None, draw_mode=True, temperature=0.1, factor=0.2):
        z = None
        if z_input is not None:
            z = [z_input]
        sample_strokes, m = sample(self.sess, self.sample_model,
                                   seq_len=self.eval_model.hps.max_seq_len,
                                   temperature=temperature, z=z)
        strokes = to_normal_strokes(sample_strokes)
        if draw_mode:
            draw_strokes(strokes, factor)
        return strokes

    def draw_stroke_list(self, latent = None, vary_temp = False, start_temp = 0.5, n = 10):
        reconstructions = []
        for i in range(n):
            if self.cond:
                if type(latent) is list:
                    # for every latent vector in latent, sample a vector image
                    reconstructions.append([self.decode(latent[i], draw_mode = False), [0, i]])
                else:
                    reconstructions.append([self.decode(latent, draw_mode = False,
                                            temperature = 0.1 * i + vary_temp), [0, i]])
            else:
                # randomly unconditionally generate n examples
                reconstructions.append([self.decode(temperature = start_temp, draw_mode = False),
                                        [0, i]])

        stroke_grid = make_grid_svg(reconstructions)
        draw_strokes(stroke_grid)


def initialize_vae():
    manual = "Please choose the model you'd like to test\n"

    i = -1
    while i < 0:
        try:
            i = int(input(manual))
        except ValueError:
            print("Please insert a number\n")

    if i == 0:
        framework = Tester('datasets', 'cat_saucer_model')
    elif i == 1:
        framework = Tester('datasets', 'owl_model', False)
    else:
        framework = Tester('datasets', 'little_prince_model')

    return framework


if __name__ == "__main__":
    welcome = "This is a tool to experiment with some sketch rnn implementation.\n"\
              "The available models are:\n"\
              "0 - A simple VAE trained on cats and flying saucers\n"\
              "1 - An auto-regressive unconditional generator\n"\
              "2 - A complex VAE trained on elephants, hats and snakes\n"

    print(welcome)
    model = initialize_vae()
