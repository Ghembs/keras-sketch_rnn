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
import sys

# libraries required for visualisation
from IPython.display import SVG, display
import svgwrite
# import our command line tools
from magenta.models.sketch_rnn.sketch_rnn_train import *
from magenta.models.sketch_rnn.model import *
from magenta.models.sketch_rnn.utils import *
from magenta.models.sketch_rnn.rnn import *

# set numpy output to something sensible
np.set_printoptions(precision=8, edgeitems=6, linewidth=200, suppress=True)

# Path for the sketch-rnn folder
path = 'D:/Documenti/UNI/magenta/magenta/models/sketch_rnn/'


# =========================== UTILITIES ======================================
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

    def __init__(self, data, model, max_seq_len, conditional = True):

        self.cond = conditional
        self.max_seq_len = max_seq_len
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

    def encode(self, input_strokes, name = 'enc_sample.svg'):
        strokes = to_big_strokes(input_strokes).tolist()
        strokes.insert(0, [0, 0, 1, 0, 0])
        strokes = strokes[:self.max_seq_len]
        seq_len = [len(input_strokes)]
        draw_strokes(to_normal_strokes(np.array(strokes)), svg_filename = name)
        return self.sess.run(self.eval_model.batch_z,
                             feed_dict={self.eval_model.input_data: [strokes],
                                        self.eval_model.sequence_lengths: seq_len})[0]

    def decode(self, z_input=None, draw_mode=True, temperature=0.2, factor=0.2,
               name = 'dec_sample.svg'):
        z = None
        if z_input is not None:
            z = [z_input]
        sample_strokes, m = sample(self.sess, self.sample_model,
                                   seq_len=self.eval_model.hps.max_seq_len,
                                   temperature=temperature, z=z)
        strokes = to_normal_strokes(sample_strokes)
        if draw_mode:
            draw_strokes(strokes, factor, svg_filename = name)
        return strokes

    def draw_stroke_list(self, latent = None, start_temp = 0.5, n = 10):
        reconstructions = []
        for i in range(n):
            if self.cond:
                if type(latent) is list:
                    name = 'interpolated_sample.svg'
                    # for every latent vector in latent, sample a vector image
                    reconstructions.append([self.decode(latent[i], draw_mode = False),
                                            [0, i]])
                else:
                    name = 'temperature_sample.svg'
                    reconstructions.append([self.decode(latent, draw_mode = False,
                                            temperature = 0.1 * i + 0.1),
                                            [0, i]])
            else:
                name = 'unconditioned_sample.svg'
                # randomly unconditionally generate n examples
                reconstructions.append([self.decode(temperature = start_temp, draw_mode = False),
                                        [0, i]])

        stroke_grid = make_grid_svg(reconstructions)
        draw_strokes(stroke_grid, svg_filename = name)


def load_dataset(dataset):
    dataset = path + "datasets/" + dataset + ".full.npz"

    test = np.load(dataset, encoding = 'bytes')["test"]

    test = DataLoader(test)
    scale_factor = test.calculate_normalizing_scale_factor()
    test.normalize(scale_factor)

    return test


def extract_stroke(model, dataset, wrong = False):
    if wrong:
        test = load_dataset(dataset)
        stroke = test.random_sample()
    else:
        stroke = model.test_set.random_sample()

    return stroke


def conditional_generation(model, operation, temp, datasets, wrong = False):
    stroke = extract_stroke(model, datasets[0], wrong)
    z0 = model.encode(stroke, name = 'z0_sample.svg')

    if operation == 0:
        i = 0
        while i < 1:
            try:
                i = int(input("Insert number of pictures to generate from single sample "
                              "(blank for 1):\n"))
            except ValueError:
                i = 1
        for j in range(i):
            _ = model.decode(z0, temperature = temp, name = 'dec' + str(j) + '_sample.svg')
    else:
        if operation == 2:
            stroke = extract_stroke(model, datasets[1], wrong)
            z1 = model.encode(stroke, name = 'z1_sample.svg')

            z0 = interpolation(z0, z1, n = 50)

            for i in range(len(z0)):
                _ = model.decode(z0[i], temperature = temp, name = 'frame_' + str(i) + '.svg')

        model.draw_stroke_list(z0, n = 50)


def unconditional_generation(model, temp):
    model.draw_stroke_list(start_temp = temp)


def run_test(framework, arguments):
    try:
        temp = float(arguments[0])
    except (ValueError, IndexError):
        temp = 0.5

    if temp < 0:
        temp = 0
    elif temp > 1:
        temp = 1

    if framework.cond:
        try:
            operation = int(arguments[1])
        except (ValueError, IndexError):
            operation = 0
        try:
            wrong = bool(arguments[2])
        except (ValueError, IndexError):
            wrong = False

        dataset1 = ''
        dataset2 = ''

        if wrong:
            try:
                dataset1 = arguments[3]
            except (ValueError, IndexError):
                dataset1 = 'cookie'
            if operation == 2:
                try:
                    dataset2 = arguments[4]
                except (ValueError, IndexError):
                    dataset2 = 'sword'
        conditional_generation(framework, operation, temp, [dataset1, dataset2], wrong)

    else:
        unconditional_generation(framework, temp)


def main(arguments):
    try:
        i = int(arguments[0])
    except ValueError:
        i = 0

    if i == 0:
        framework = Tester('datasets', 'cat_saucer_model', 130)
    elif i == 1:
        framework = Tester('datasets', 'little_prince_model', 147)
    else:
        i = 2
        framework = Tester('datasets', 'owl_model', False)

    try:
        run_test(framework, arguments[1:])
    except IndexError:
        run_test(framework, [0.5])

    message = "Run more tests?\n " \
              "(insert new parameters except for [model], leave empty to end program)\n"
    more = True

    while more:
        params = input(message).split()
        if len(params) == 0:
            more = False
        else:
            run_test(framework, params)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        welcome = "This is a tool to experiment with some sketch-rnn implementations.\n"\
                   "Arguments: "\
                   "[model] [temperature] [operation] [custom] [dataset1] [dataset2]\n\n"\
                   "[model] an integer to choose which model to test:\n"\
                   "\t 0 - Simple VAE trained on cats and flying saucers\n"\
                   "\t 1 - Complex VAE trained on elephants, hats and snakes\n"\
                   "\t 2 - Autoregressive RNN trained on owls\n\n" \
                   "[temperature] a float in range (0.0, 1.0), temperature parameter for " \
                   "the model.\n\n"\
                   "[operation] an integer to choose what to generate (only required " \
                   "for VAEs):\n"\
                   "\t 0 - Conditional generation\n"\
                   "\t 1 - Conditional reconstruction (varying temperature)\n"\
                   "\t 2 - Conditional interpolation\n\n"\
                   "[custom] a boolean to choose if to generate from default sketches " \
                   "(only for VAEs).\n\n"\
                   "[dataset1] name of a dataset in path/datasets folder " \
                   "(only if [custom] == True).\n\n"\
                   "[dataset2] name of a dataset in path/datasets folder " \
                   "(only if [custom] == True and [operation] == 2).\n"

        print(welcome)
    else:
        main(sys.argv[1:])
