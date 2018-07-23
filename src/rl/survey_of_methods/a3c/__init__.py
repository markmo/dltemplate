from argparse import ArgumentParser
from common.util import load_hyperparams, merge_dict
import multiprocessing
import os
from rl.survey_of_methods.a3c.model_setup import ACNetwork, Worker
import tensorflow as tf
import threading
from time import sleep
from vizdoom import *


# noinspection PyUnusedLocal,PyPep8
def run(constant_overwrites):
    config_path = os.path.join(os.path.dirname(__file__), 'hyperparams.yml')
    constants = merge_dict(load_hyperparams(config_path), constant_overwrites)

    max_episode_length = constants['max_episode_length']
    gamma = constants['gamma']
    s_size = constants['s_size']
    a_size = constants['a_size']
    load_model = constants['load_model']
    model_path = constants['model_path']

    tf.reset_default_graph()

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # Create a directory to save episode playback gifs
    if not os.path.exists('./frames'):
        os.makedirs('./frames')

    with tf.device('/cpu:0'):
        global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)

        # Generate global network
        master_network = ACNetwork(s_size, a_size, 'global', None)

        # Set workers to number of available CPU threads
        n_workers = multiprocessing.cpu_count()
        workers = []

        # Create worker classes
        for i in range(n_workers):
            workers.append(Worker(DoomGame(), i, s_size, a_size, optimizer, model_path, global_episodes))

        saver = tf.train.Saver(max_to_keep=5)

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        if load_model:
            print('Loading model...')
            ckpt = tf.train.get_checkpoint_state(model_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())

        # This is where the asynchronous magic happens.
        # Start the "work" process for each worker in a separate thread
        worker_threads = []
        for worker in workers:
            work = lambda: worker.work(max_episode_length, gamma, sess, coord, saver)
            t = threading.Thread(target=work)
            t.start()
            sleep(0.5)
            worker_threads.append(t)

        coord.join(worker_threads)


if __name__ == '__main__':
    # read args
    parser = ArgumentParser(description='Run A3C Agent')
    parser.add_argument('--model-path', dest='model_path', type=str, default='./model', help='file path to saved model')
    parser.add_argument('--load-model', dest='load_model', help='load model flag', action='store_true')
    parser.set_defaults(load_model=False)
    args = parser.parse_args()

    run(vars(args))
