import numpy as np
from rl.survey_of_methods.a3c.util import discount, normalized_columns_initializer
from rl.survey_of_methods.a3c.util import process_frame, update_target_graph
from rl.survey_of_methods.deep_recurrent_q_network.util import make_gif
import tensorflow as tf
import tensorflow.contrib.slim as slim
from vizdoom import *


class ACNetwork(object):

    def __init__(self, s_size, a_size, scope, optimizer):
        with tf.variable_scope(scope):
            # input and visual encoding layers
            self.inputs = tf.placeholder(shape=[None, s_size], dtype=tf.float32)
            self.image_in = tf.reshape(self.inputs, shape=[-1, 84, 84, 1])
            self.conv1 = slim.conv2d(activation_fn=tf.nn.elu, inputs=self.image_in, num_outputs=16,
                                     kernel_size=[8, 8], stride=[4, 4], padding='VALID')
            self.conv2 = slim.conv2d(activation_fn=tf.nn.elu, inputs=self.conv1, num_outputs=32,
                                     kernel_size=[4, 4], stride=[2, 2], padding='VALID')
            hidden = slim.fully_connected(slim.flatten(self.conv2), 256, activation_fn=tf.nn.elu)

            # Recurrent network for temporal dependencies
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(256, state_is_tuple=True)
            c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
            h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
            self.state_init = [c_init, h_init]
            c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
            h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
            self.state_in = (c_in, h_in)
            rnn_in = tf.expand_dims(hidden, [0])
            step_size = tf.shape(self.image_in)[:1]
            state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(lstm_cell, rnn_in,
                                                         initial_state=state_in,
                                                         sequence_length=step_size,
                                                         time_major=False)
            lstm_c, lstm_h = lstm_state
            self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
            rnn_out = tf.reshape(lstm_outputs, [-1, 256])

            # output layers for policy and value estimations
            self.policy = slim.fully_connected(rnn_out, a_size,
                                               activation_fn=tf.nn.softmax,
                                               weights_initializer=normalized_columns_initializer(0.01),
                                               biases_initializer=None)
            self.value = slim.fully_connected(rnn_out, 1,
                                              activation_fn=None,
                                              weights_initializer=normalized_columns_initializer(1.0),
                                              biases_initializer=None)

            # only the worker network needs ops for loss and gradient update
            if scope != 'global':
                self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
                self.actions_onehot = tf.one_hot(self.actions, a_size, dtype=tf.float32)
                self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)
                self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)
                self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])

                # loss functions
                self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value, [-1])))
                self.entropy = -tf.reduce_sum(self.policy * tf.log(self.policy))
                self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs) * self.advantages)
                self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.01

                # get gradients from local network using local losses
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss, local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, 40.)

                # apply local gradients to global network
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = optimizer.apply_gradients(zip(grads, global_vars))


class Worker(object):

    def __init__(self, game, name, s_size, a_size, optimizer, model_path, global_episodes):
        self.name = 'worker_{}'.format(name)
        self.number = name
        self.model_path = model_path
        self.optimizer = optimizer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.summary_writer = tf.summary.FileWriter('train_{}'.format(self.number))

        # create the local copy of the network and the tensorflow op
        # to copy global parameters to the local network
        self.local_ac = ACNetwork(s_size, a_size, self.name, optimizer)
        self.update_local_ops = update_target_graph('global', self.name)

        # the following code is for setting up the Doom environment
        # this corresponds to the simple task we will pose our agent
        game.set_doom_scenario_path('basic.wad')
        game.set_doom_map('map01')
        game.set_screen_resolution(ScreenResolution.RES_160X120)
        game.set_screen_format(ScreenFormat.GRAY8)
        game.set_render_hud(False)
        game.set_render_crosshair(False)
        game.set_render_weapon(True)
        game.set_render_decals(False)
        game.set_render_particles(False)
        game.add_available_button(Button.MOVE_LEFT)
        game.add_available_button(Button.MOVE_RIGHT)
        game.add_available_button(Button.ATTACK)
        game.add_available_game_variable(GameVariable.AMMO2)
        game.add_available_game_variable(GameVariable.POSITION_X)
        game.add_available_game_variable(GameVariable.POSITION_Y)
        game.set_episode_timeout(300)
        game.set_episode_start_time(10)
        game.set_window_visible(False)
        game.set_sound_enabled(False)
        game.set_living_reward(-1)
        game.set_mode(Mode.PLAYER)
        game.init()
        self.actions = np.identity(a_size, dtype=bool).tolist()
        # end Doom setup

        self.env = game

        self.rewards_plus = None
        self.value_plus = None
        self.batch_rnn_state = None

    def train(self, rollout, sess, gamma, bootstrap_value):
        rollout = np.array(rollout)
        observations = rollout[:, 0]
        actions = rollout[:, 1]
        rewards = rollout[:, 2]
        # next_observations = rollout[:, 3]
        values = rollout[:, 5]

        # Take the rewards and values from the rollout, and use them to
        # generate the advantage and discounted returns.
        # The advantage function uses "Generalized Advantage Estimation"
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus, gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages, gamma)

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        feed_dict = {
            self.local_ac.target_v: discounted_rewards,
            self.local_ac.inputs: np.vstack(observations),
            self.local_ac.actions: actions,
            self.local_ac.advantages: advantages,
            self.local_ac.state_in[0]: self.batch_rnn_state[0],
            self.local_ac.state_in[1]: self.batch_rnn_state[1]
        }
        v1, p1, e1, g_n, v_n, self.batch_rnn_state, _ = \
            sess.run([self.local_ac.value_loss,
                      self.local_ac.policy_loss,
                      self.local_ac.entropy,
                      self.local_ac.grad_norms,
                      self.local_ac.var_norms,
                      self.local_ac.state_out,
                      self.local_ac.apply_grads
                      ], feed_dict=feed_dict)
        rollout_length = len(rollout)

        return v1 / rollout_length, p1 / rollout_length, e1 / rollout_length, g_n, v_n

    # noinspection PyUnboundLocalVariable,PyUnresolvedReferences
    def work(self, max_episode_length, gamma, sess, coord, saver):
        episode_count = sess.run(self.global_episodes)
        n_steps = 0
        print('Starting worker', str(self.number))
        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                sess.run(self.update_local_ops)
                episode_buffer, episode_values, episode_frames = [], [], []
                episode_reward = 0
                episode_step_count = 0

                self.env.new_episode()
                s = self.env.get_state().screen_buffer
                episode_frames.append(s)
                s = process_frame(s)
                rnn_state = self.local_ac.state_init
                self.batch_rnn_state = rnn_state
                while not self.env.is_episode_finished():
                    # Take an action using probabilities from policy network output
                    a_dist, v, rnn_state = sess.run([self.local_ac.policy,
                                                     self.local_ac.value,
                                                     self.local_ac.state_out], feed_dict={
                        self.local_ac.inputs: [s],
                        self.local_ac.state_in[0]: rnn_state[0],
                        self.local_ac.state_in[1]: rnn_state[1]
                    })
                    a = np.random.choice(a_dist[0], p=a_dist[0])
                    a = np.argmax(a_dist == a)

                    reward = self.env.make_action(self.actions[a]) / 100.
                    done = self.env.is_episode_finished()
                    if done:
                        s1 = s
                    else:
                        s1 = self.env.get_state().screen_buffer
                        episode_frames.append(s1)
                        s1 = process_frame(s1)

                    episode_buffer.append([s, a, reward, s1, done, v[0, 0]])
                    episode_values.append(v[0, 0])

                    episode_reward += reward
                    s = s1
                    n_steps += 1
                    episode_step_count += 1

                    # If the episode hasn't ended, but the experience buffer is full, then we
                    # make an update step using that experience rollout.
                    if len(episode_buffer) == 30 and not done and episode_step_count != max_episode_length - 1:
                        # Since we don't know what the true final return is,
                        # we "bootstrap" from our current value estimation.
                        v1 = sess.run(self.local_ac.value, feed_dict={
                            self.local_ac.inputs: [s],
                            self.local_ac.state_in[0]: rnn_state[0],
                            self.local_ac.state_in[1]: rnn_state[1]
                        })[0, 0]
                        v1, p1, e1, g_n, v_n = self.train(episode_buffer, sess, gamma, v1)
                        episode_buffer = []
                        sess.run(self.update_local_ops)
                    if done:
                        break

                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))

                # Update the network using the episode buffer at the end of the episode
                if len(episode_buffer) != 0:
                    v1, p1, e1, g_n, v_n = self.train(episode_buffer, sess, gamma, 0.)

                # Periodically save episode gifs, model parameters, and summary statistics
                if episode_count % 5 == 0 and episode_count != 0:
                    if self.name == 'worker_0' and episode_count % 25 == 0:
                        time_per_step = 0.05
                        images = np.array(episode_frames)
                        make_gif(images, './frames/image{}.gif'.format(episode_count),
                                 duration=len(images) * time_per_step,
                                 true_image=True, salience=False)

                if episode_count % 250 == 0 and self.name == 'worker_0':
                    saver.save(sess, '{}/model-{}'.format(self.model_path, episode_count))
                    # print('Model saved')

                mean_reward = np.mean(self.episode_rewards[-5:])
                mean_length = np.mean(self.episode_lengths[-5:])
                mean_value = np.mean(self.episode_mean_values[-5:])
                summary = tf.Summary()
                summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
                summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
                summary.value.add(tag='Loss/Value', simple_value=float(v1))
                summary.value.add(tag='Loss/Policy', simple_value=float(p1))
                summary.value.add(tag='Loss/Entropy', simple_value=float(e1))
                summary.value.add(tag='Loss/Grad Norm', simple_value=float(g_n))
                summary.value.add(tag='Loss/Var Norm', simple_value=float(v_n))
                self.summary_writer.add_summary(summary, episode_count)
                self.summary_writer.flush()

            if self.name == 'worker_0':
                sess.run(self.increment)

            episode_count += 1
