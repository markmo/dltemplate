import json
import logging
import numpy as np
import os
from os import listdir
from os.path import isfile, join
from shutil import copyfile
import subprocess
import sys
from threading import Timer
import time
import yaml


JSON_TYPE = 'json'
YAML_TYPE = 'yaml'


def minibatches(data_generator, minibatch_size):
    """

    :param data_generator: generator of (img, formulas) tuples
    :param minibatch_size: (int)
    :return: list of tuples
    """
    x_batch, y_batch = [], []
    for (x, y) in data_generator:
        if len(x_batch) == minibatch_size:
            yield x_batch, y_batch
            x_batch, y_batch = [], []

        x_batch += [x]
        y_batch += [y]

    if len(x_batch) != 0:
        yield x_batch, y_batch


def run(cmd, timeout_sec):

    def kill_proc(p):
        p.kill()

    proc = subprocess.Popen(cmd, shell=True)
    timer = Timer(timeout_sec, kill_proc, [proc])
    try:
        timer.start()
        # stdout, stderr = proc.communicate()
        proc.communicate()
    finally:
        timer.cancel()


def get_logger(filename):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.INFO)
    logging.basicConfig(format='%(message)s', level=logging.INFO)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.INFO)
    # noinspection SpellCheckingInspection
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)
    return logger


def init_dir(dir_name):
    """
    Creates directory if it doesn't exist

    :param dir_name:
    :return:
    """
    if dir_name is not None:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)


def init_file(file_path, mode='a'):
    """
    Ensure that a given file exists

    :param file_path:
    :param mode:
    :return:
    """
    with open(file_path, mode) as _:
        pass


def get_files(dir_name):
    files = [f for f in listdir(dir_name) if isfile(join(dir_name, f))]
    return files


def delete_file(file_path):
    # noinspection PyBroadException
    try:
        os.remove(file_path)
    except Exception:
        pass


class Config(object):
    """
    Loads hyperparameters from JSON or YAML file
    """

    def __init__(self, source):
        self.source = source
        self._file_type = None
        if type(source) is dict:
            self.__dict__.update(source)
        elif type(source) is list:
            for s in source:
                self.load(s)
        else:
            self.load(source)

    def load(self, source):
        file_ext = source.split('.')[-1]
        with open(source) as f:
            if file_ext == 'json':
                data = json.load(f)
                self._file_type = JSON_TYPE
            elif file_ext == 'yml' or file_ext == 'yaml':
                data = yaml.load(f)
                self._file_type = YAML_TYPE
            else:
                raise NotImplementedError('Unsupported file type {}'.format(file_ext))

            self.__dict__.update(data)

    def save(self, dir_name):
        init_dir(dir_name)
        if type(self.source) is list:
            for s in self.source:
                c = Config(s)
                c.save(dir_name)
        elif type(self.source) is dict:
            if self._file_type == JSON_TYPE:
                json.dumps(self.source, indent=4)
            elif self._file_type == YAML_TYPE:
                yaml.dump(self.source)
        else:
            copyfile(self.source, dir_name + self.export_name)


class ProgressBar(object):
    """
    Progress bar class inspired by Keras
    """

    def __init__(self, max_step, width=30):
        self.max_step = max_step
        self.width = width
        self.last_width = 0
        self.sum_values = {}
        self.start = time.time()
        self.last_step = 0
        self.info = ''
        self.bar = ''

    def _update_values(self, curr_step, values):
        for k, v in values:
            if k not in self.sum_values:
                self.sum_values[k] = [v * (curr_step - self.last_step), curr_step - self.last_step]
            else:
                self.sum_values[k][0] += v * (curr_step - self.last_step)
                self.sum_values[k][1] += (curr_step - self.last_step)

    def _write_bar(self, curr_step):
        last_width = self.last_width
        sys.stdout.write('\b' * last_width)
        sys.stdout.write('\r')

        n_digits = int(np.floor(np.log10(self.max_step))) + 1
        bar = '%%%dd/%%%dd [' % (n_digits, n_digits)
        bar = bar % (curr_step, self.max_step)
        progress = float(curr_step) / self.max_step
        progress_width = int(self.width * progress)
        if progress_width > 0:
            bar += ('=' * (progress_width - 1))
            if curr_step < self.max_step:
                bar += '>'
            else:
                bar += '='

        bar += ('.' * (self.width - progress_width))
        bar += ']'
        sys.stdout.write(bar)
        return bar

    def _get_eta(self, curr_step):
        now = time.time()
        if curr_step:
            time_per_unit = (now - self.start) / curr_step
        else:
            time_per_unit = 0

        eta = time_per_unit * (self.max_step - curr_step)

        if curr_step < self.max_step:
            info = ' - ETA: %ds' % eta
        else:
            info = ' - %ds' % (now - self.start)

        return info

    def _get_values_sum(self):
        info = ''
        for name, value in self.sum_values.items():
            info += ' - %s: %.4f' % (name, value[0] / max(1, value[1]))

        return info

    def _write_info(self, curr_step):
        info = ''
        info += self._get_eta(curr_step)
        info += self._get_values_sum()

        sys.stdout.write(info)

        return info

    def _update_width(self, curr_step):
        curr_width = len(self.bar) + len(self.info)
        if curr_width < self.last_width:
            sys.stdout.write(' ' * (self.last_width - curr_width))

        if curr_step >= self.max_step:
            sys.stdout.write('\n')

        sys.stdout.flush()

        self.last_width = curr_width

    def update(self, curr_step, values):
        """
        Updates the progress bar

        :param curr_step:
        :param values: list of tuples (name, value_for_last_step)
                       The progress bar will display averages for these values
        :return:
        """
        self._update_values(curr_step, values)
        self.bar = self._write_bar(curr_step)
        self.info = self._write_info(curr_step)
        self._update_width(curr_step)
        self.last_step = curr_step
