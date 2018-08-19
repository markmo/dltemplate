""" main file """
__credits__ = 'giorgio@ac.upc.edu'

from argparse import ArgumentParser
from common.util import load_hyperparams, merge_dict
import json
import keras
import numpy as np
import os
from rl.routing_optimization.environment import OmnetBalancerEnv, OmnetLinkweightEnv, vector_to_file
from rl.routing_optimization.model_setup import ActorNetwork, CriticNetwork
from rl.routing_optimization.traffic import OUProcess
from rl.routing_optimization.util import format_float, ReplayBuffer, setup_experiment, setup_run
import tensorflow as tf


def play(constants, is_training=True):
    if is_training:
        folder = setup_run(constants)
    else:
        folder = constants['experiment']

    if constants['seed'] == 0:
        constants['seed'] = None

    np.random.seed(constants['seed'])
    active_nodes = constants['active_nodes']

    # generate an environment
    if constants['env'] == 'balancing':
        env = OmnetBalancerEnv(constants, folder)
    elif constants['env'] == 'label':
        env = OmnetLinkweightEnv(constants, folder)
    else:
        raise ValueError("Value of hyperparam 'env' expected to be one of ['balancing', 'label']")

    action_dim, state_dim = env.a_dim, env.s_dim
    mu, theta, sigma = [constants[k] for k in ('mu', 'theta', 'sigma')]
    ou = OUProcess(action_dim, mu, theta, sigma)  # Ornstein-Uhlenbeck Process
    buffer_size, batch_size, gamma, exploration, n_episodes, max_steps = \
        [constants[k] for k in ('buffer_size', 'batch_size', 'gamma', 'exploration', 'n_episodes', 'max_steps')]
    if exploration <= 1:
        exploration = n_episodes * max_steps * exploration

    wise = False
    step = 0
    epsilon = 1

    # TensorFlow GPU optimization
    conf = tf.ConfigProto()
    conf.gpu_options.allow_growth = True
    sess = tf.Session(config=conf)
    keras.backend.set_session(sess)

    actor = ActorNetwork(sess, state_dim, action_dim, constants)
    critic = CriticNetwork(sess, state_dim, action_dim, constants)
    buff = ReplayBuffer(buffer_size)

    ltm = ['a_h0', 'a_h1', 'a_V', 'c_w1', 'c_a1', 'c_h1', 'c_h3', 'c_V']
    layers_to_mind = {}
    l2 = {}
    for k in ltm:
        layers_to_mind[k] = 0
        l2[k] = 0

    vector_to_file(ltm, folder + 'weights_l2_log.csv', 'w')

    # load the weights
    # noinspection PyBroadException
    try:
        actor.model.load_weights(folder + 'actor_model.h5')
        critic.model.load_weights(folder + 'critic_model.h5')
        actor.target_model.load_weights(folder + 'actor_model.h5')
        critic.target_model.load_weights(folder + 'critic_model.h5')
        print('Weights loaded successfully')
    except Exception:
        print('Error loading weights')

    print('OMNeT++ start experiment...')

    # initial state of simulator
    s_t = env.reset()

    for i in range(n_episodes):
        print('Episode: {} Replay Buffer: {}'.format(i, buff.count()))
        total_reward = 0
        for j in range(max_steps):
            epsilon -= 1. / exploration
            a_t = np.zeros([1, action_dim])
            noise_t = np.zeros([1, action_dim])
            a_t_original = actor.model.predict(s_t.reshape(1, s_t.shape[0]))
            if is_training and epsilon > 0 and (step % 1000) // 100 != 9:
                noise_t[0] = epsilon * ou.evolve()

            a = a_t_original[0]
            n = noise_t[0]
            a_t[0] = np.where((a + n > 0) & (a + n < 1), a + n, a - n).clip(min=0, max=1)

            # execute action
            s_t1, r_t, done = env.step(a_t[0])
            buff.add(s_t, a_t[0], r_t, s_t1, done)  # Add to replay buffer

            # noinspection PyPep8
            scale_ = lambda x: x

            # do the batch update
            batch = buff.get_batch(batch_size)
            states = scale_(np.asarray([e[0] for e in batch]))
            actions = scale_(np.asarray([e[1] for e in batch]))
            rewards = scale_(np.asarray([e[2] for e in batch]))
            new_states = scale_(np.asarray([e[3] for e in batch]))
            dones = np.asarray([e[4] for e in batch])

            y_t = np.zeros([len(batch), action_dim])
            target_q_values = critic.target_model.predict([new_states, actor.target_model.predict(new_states)])

            for k in range(len(batch)):
                if dones[k]:
                    y_t[k] = rewards[k]
                else:
                    y_t[k] = rewards[k] + gamma * target_q_values[k]

            if is_training and len(batch) >= batch_size:
                loss = critic.model.train_on_batch([states, actions], y_t)
                a_for_grad = actor.model.predict(states)
                grads = critic.gradients(states, a_for_grad)
                actor.train(states, grads)
                actor.train_target()
                critic.train_target()
                with open(folder + 'loss_log.csv', 'a') as f:
                    f.write(format_float(loss) + '\n')

            total_reward += r_t
            s_t = s_t1

            for layer in actor.model.layers + critic.model.layers:
                if layer.name in layers_to_mind:
                    l2[layer.name] = np.linalg.norm(np.ravel(layer.get_weights()[0]) - layers_to_mind[layer.name])
                    # vector_to_file(np.ravel(layer.get_weights()[0]),
                    #                '{}weights_{}_log.csv'.format(folder, layer.name), 'a')
                    layers_to_mind[layer.name] = np.ravel(layer.get_weights()[0])

                # if max(l2.values()) <= 0.02:
                #     wise = True

            if is_training and len(batch) >= batch_size:
                vector_to_file([l2[x] for x in ltm], folder + 'weights_l2_log.csv', 'a')

            vector_to_file(a_t_original[0], folder + 'action_log.csv', 'a')
            vector_to_file(noise_t[0], folder + 'noise_log.csv', 'a')

            if 'print' in constants and constants['print']:
                print('Episode', '%5d' % i, 'Step', '%5d' % step, 'Reward', '%.6f' % r_t)
                print('Epsilon', '%.6f' % max(epsilon, 0))
                att = np.split(a_t[0], active_nodes)
                for k in range(active_nodes):
                    att[k] = np.insert(att[k], k, -1)

                att = np.concatenate(att)
                print('Action\n', att.reshape(active_nodes, active_nodes))
                print(max(l2, key=l2.get), format_float(max(l2.values())))

            step += 1
            if done or wise:
                break

        if np.mod(i + 1, 2) == 0:  # writes every second episode
            if is_training:
                actor.model.save_weights(folder + 'actor_model.h5', overwrite=True)
                actor.model.save_weights(folder + 'actor_model{}.h5'.format(step))
                with open(folder + 'actor_model.json', 'w') as f:
                    f.write(actor.model.to_json(indent=4) + '\n')

                critic.model.save_weights(folder + 'critic_model.h5', overwrite=True)
                critic.model.save_weights(folder + 'critic_model{}.h5'.format(step))
                with open(folder + 'critic_model.json', 'w') as f:
                    f.write(critic.model.to_json(indent=4) + '\n')

        print('TOTAL_REWARD @ {}-th Episode: {}'.format(i, total_reward))
        print('Total Steps:', str(step))
        print('')

    env.end()  # shutdown
    print('Done')


def run(constant_overwrites):
    config_path = os.path.join(os.path.dirname(__file__), 'hyperparams.yml')
    constants = merge_dict(load_hyperparams(config_path), constant_overwrites)
    if constants['play']:
        if 'folder' in constants:
            folder = constants['folder']
            with open(folder + '/hyperparams.json') as f:
                constants = merge_dict(constants, json.load(f))

            # check for slash at end
            experiment = folder if folder[-1] == '/' else folder + '/'
            constants['experiment'] = experiment

            if 'traffic_folder' in constants:
                if constants['traffic'] == 'dir:':
                    constants['traffic'] += constants['traffic_folder']

            play(constants, is_training=False)
        else:
            print('Folder must be specific to play')
    else:
        constants['experiment'] = setup_experiment()
        play(constants, is_training=True)


if __name__ == '__main__':
    # read args
    parser = ArgumentParser(description='Run Routing Optimizer')
    parser.add_argument('--episodes', dest='n_episodes', type=int, help='number episodes')
    parser.add_argument('--nodes', dest='active_nodes', type=int, help='number active nodes')
    parser.add_argument('--batch_size', dest='batch_size', type=int, help='batch_size')
    parser.add_argument('--h1', dest='n_hidden1', type=int, help='size hidden layer 1')
    parser.add_argument('--h2', dest='n_hidden2', type=int, help='size hidden layer 2')
    parser.add_argument('--lr-actor', dest='lr_actor', type=float, help='actor learning rate')
    parser.add_argument('--lr-critic', dest='lr_critic', type=float, help='critic learning rate')
    parser.add_argument('--epsilon', dest='epsilon', type=float, help='exploration rate')
    parser.add_argument('--seed', dest='seed', help='random seed')
    parser.add_argument('--env', dest='env', type=str, help='environment')
    parser.add_argument('--reward', dest='reward_fn', type=str, help='reward calculation method')
    parser.add_argument('--state', dest='state_type', type=str, help='state type')
    parser.add_argument('--traffic', dest='traffic', type=str, help='traffic generation method')
    parser.add_argument('--traffic-folder', dest='traffic_folder', type=str, help='path to folder with traffic file')
    parser.add_argument('--folder', dest='folder', type=str, help='path to model folder')
    parser.add_argument('--print', dest='print', help='print logs during training', action='store_true')
    parser.add_argument('--play', dest='play', help='play without training', action='store_true')
    parser.set_defaults(print=False)
    parser.set_defaults(play=False)
    args = parser.parse_args()

    run(vars(args))
