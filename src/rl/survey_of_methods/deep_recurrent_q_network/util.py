import csv
import moviepy.editor as mpy
import numpy as np
import os
import random
from rl.survey_of_methods.deep_recurrent_q_network.model_setup import QNetwork
import tensorflow as tf


class ExperienceBuffer(object):

    def __init__(self, buffer_size=1000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        n = len(self.buffer) + 1
        if n >= self.buffer_size:
            self.buffer[0:n - self.buffer_size] = []

        self.buffer.append(experience)

    def sample(self, batch_size, trace_length):
        sampled_episodes = random.sample(self.buffer, batch_size)
        sampled_traces = []
        for episode in sampled_episodes:
            point = np.random.randint(0, len(episode) + 1 - trace_length)
            sampled_traces.append(episode[point:point + trace_length])

        return np.reshape(np.array(sampled_traces), [batch_size * trace_length, 5])


def make_gif(images, filename, duration=2, true_image=False, salience=False, sal_images=None):
    """ Enables gifs of the training episode to be saved for use in the Control Center """

    # noinspection PyBroadException
    def make_frame(t):
        try:
            x = images[int(len(images) / duration * t)]
        except Exception:
            x = images[-1]

        if true_image:
            return x.astype(np.uint8)
        else:
            return ((x + 1) / 2 * 255).astype(np.uint8)

    # noinspection PyBroadException
    def make_mask(t):
        try:
            x = sal_images[int(len(sal_images) / duration * t)]
        except Exception:
            x = sal_images[-1]

        return x

    clip = mpy.VideoClip(make_frame, duration=duration)
    if salience:
        mask = mpy.VideoClip(make_mask, ismask=True, duration=duration)
        mask = mask.set_opacity(0.1)
        mask.write_gif(filename, fps=len(images) / duration, verbose=False)
    else:
        clip.write_gif(filename, fps=len(images) / duration, verbose=False)


def process_state(states):
    """ resize game frames """
    return np.reshape(states, [21168])


def save_to_monitor(i, rewards, js, buffer_array, summary_length, n_hidden, sess, main_network, time_per_step):
    """ Record performance metrics and episode logs for the Control Center """
    with open('./monitor/log.csv', 'a') as f:
        state_display = (np.zeros([1, n_hidden]), np.zeros([1, n_hidden]))
        images_s = []
        for i, _ in enumerate(np.vstack(buffer_array[:, 0])):
            img, state_display = sess.run([main_network.salience, main_network.rnn_state], feed_dict={
                main_network.scalar_input: np.reshape(buffer_array[i, 0], [1, 21168]) / 255.,
                main_network.sequence_length: 1,
                main_network.state_in: state_display,
                main_network.batch_size: 1
            })
            images_s.append(img)

        images_s = (images_s - np.min(images_s)) / (np.max(images_s) - np.min(images_s))
        images_s = np.vstack(images_s)
        images_s = np.resize(images_s, [len(images_s), 84, 84, 3])
        luminance = np.max(images_s, 3)
        images_s = np.multiply(np.ones([len(images_s), 84, 84, 3]),
                               np.reshape(luminance, [len(images_s), 84, 84, 1]))
        make_gif(np.ones([len(images_s), 84, 84, 3]), './monitor/frames/sal{}.gif'.format(i),
                 duration=len(images_s) * time_per_step, true_image=False, salience=True, sal_images=luminance)

        images = list(zip(buffer_array[:, 0]))
        images.append(buffer_array[-1, 3])
        images = np.vstack(images)
        images = np.resize(images, [len(images), 84, 84, 3])
        make_gif(images, './monitor/frames/image{}.gif'.format(i), duration=len(images_s) * time_per_step,
                 true_image=True, salience=False)
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow([i, np.mean(js[-100:]), np.mean(rewards[-summary_length:]),
                         './frames/image{}.gif'.format(i),
                         './frames/log{}.csv'.format(i),
                         './frames/sal{}.gif'.format(i)])
        f.close()

    with open('./monitor/frames/log{}.csv'.format(i), 'w') as f:
        state_train = (np.zeros([1, n_hidden]), np.zeros([1, n_hidden]))
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(['ACTION', 'REWARD', 'A0', 'A1', 'A2', 'A3', 'V'])
        a, v = sess.run([main_network.advantage, main_network.value], feed_dict={
            main_network.scalar_input: np.vstack(buffer_array[:, 0]) / 255.,
            main_network.sequence_length: len(buffer_array),
            main_network.state_in: state_train,
            main_network.batch_size: 1
        })
        writer.writerows(zip(buffer_array[:, 1], buffer_array[:, 2], a[:, 0], a[:, 1], a[:, 2], a[:, 3], v[:, 0]))


def test(env, n_hidden, save_path, load_model, n_episodes, max_episode_length,
         summary_length, time_per_step, epsilon):
    tf.reset_default_graph()

    # we define the cells for the primary and target Q-networks
    cell_main = tf.contrib.rnn.BasicLSTMCell(num_units=n_hidden, state_is_tuple=True)
    main_network = QNetwork(n_hidden, cell_main, 'main')
    saver = tf.train.Saver(max_to_keep=2)
    js, rewards = [], []
    n_steps = 0

    # make path to save model, unless path already exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # write the first line of the master log-file for the Control Center
    with open('monitor/log.csv', 'w') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(['Episode', 'Length', 'Reward', 'IMG', 'LOG', 'SAL'])

    with tf.Session() as sess:
        if load_model:
            print('Loading model...')
            ckpt = tf.train.get_checkpoint_state(save_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())

        for i in range(n_episodes):
            episode_buffer = []

            # reset environment and get first new observation
            s_p = env.reset()
            s = process_state(s_p)
            total_reward = 0
            j = 0

            # reset the recurrent layer's hidden state
            state = (np.zeros([1, n_hidden]), np.zeros([1, n_hidden]))

            # Train the Q-network
            # if the agent takes longer to reach either of the blocks, then end the trial
            while j < max_episode_length:
                j += 1

                # choose an action greedily from the Q-network,
                # with epsilon chance of random action
                if np.random.rand(1) < epsilon:
                    state1 = sess.run(main_network.rnn_state, feed_dict={
                        main_network.scalar_input: [s / 255.],
                        main_network.sequence_length: 1,
                        main_network.state_in: state,
                        main_network.batch_size: 1
                    })
                    a = np.random.randint(0, 4)
                else:
                    a, state1 = sess.run([main_network.predict, main_network.rnn_state], feed_dict={
                        main_network.scalar_input: [s / 255.],
                        main_network.sequence_length: 1,
                        main_network.state_in: state,
                        main_network.batch_size: 1
                    })
                    a = a[0]

                s1_p, reward, done = env.step(a)
                s1 = process_state(s1_p)
                n_steps += 1

                # save the experience to our episode buffer
                episode_buffer.append(np.reshape(np.array([s, a, reward, s1, done]), [1, 5]))

                total_reward += reward
                s = s1
                # s_p = s1_p
                state = state1

                if done:
                    break

            js.append(j)
            rewards.append(total_reward)

            # periodically save the model
            if len(rewards) % summary_length == 0 and i != 0:
                print('Number steps:', n_steps, 'mean reward:', np.mean(rewards[-10:]), 'epsilon:', epsilon)

                # record performance metrics and episode logs for the Control Center
                save_to_monitor(i, rewards, js, np.reshape(np.array(episode_buffer), [len(episode_buffer), 5]),
                                summary_length, n_hidden, sess, main_network, time_per_step)

    print('Successful episodes %:', str(sum(rewards) / n_episodes))

    return rewards, js


def train(env, n_hidden, start_epsilon, end_epsilon, annealing_steps, tau, gamma, learning_rate,
          trace_length, save_path, load_model, n_episodes, max_episode_length, n_pretrain_steps,
          batch_size, update_freq, summary_length, time_per_step):
    tf.reset_default_graph()

    # we define the cells for the primary and target Q-networks
    cell_main = tf.contrib.rnn.BasicLSTMCell(num_units=n_hidden, state_is_tuple=True)
    cell_target = tf.contrib.rnn.BasicLSTMCell(num_units=n_hidden, state_is_tuple=True)
    main_network = QNetwork(n_hidden, cell_main, 'main', learning_rate)
    target_network = QNetwork(n_hidden, cell_target, 'target', learning_rate)
    saver = tf.train.Saver()
    trainables = tf.trainable_variables()
    target_ops = update_target_graph(trainables, tau)
    buffer = ExperienceBuffer()
    epsilon = start_epsilon
    step_drop = (start_epsilon - end_epsilon) / annealing_steps
    js, rewards = [], []
    n_steps = 0

    # make path to save model, unless path already exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # write the first line of the master log-file for the Control Center
    with open('monitor/log.csv', 'w') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(['Episode', 'Length', 'Reward', 'IMG', 'LOG', 'SAL'])

    with tf.Session() as sess:
        if load_model:
            print('Loading model...')
            ckpt = tf.train.get_checkpoint_state(save_path)
            saver.restore(sess, ckpt.model_checkpoint_path)

        sess.run(tf.global_variables_initializer())
        i = 0
        for i in range(n_episodes):
            episode_buffer = []

            # reset environment and get first new observation
            s_p = env.reset()
            s = process_state(s_p)
            total_reward = 0
            j = 0

            # reset the recurrent layer's hidden state
            state = (np.zeros([1, n_hidden]), np.zeros([1, n_hidden]))

            # Train the Q-network
            # if the agent takes longer to reach either of the blocks, then end the trial
            while j < max_episode_length:
                j += 1

                # choose an action greedily from the Q-network,
                # with epsilon chance of random action
                if np.random.rand(1) < epsilon or n_steps < n_pretrain_steps:
                    state1 = sess.run(main_network.rnn_state, feed_dict={
                        main_network.scalar_input: [s / 255.],
                        main_network.sequence_length: 1,
                        main_network.state_in: state,
                        main_network.batch_size: 1
                    })
                    a = np.random.randint(0, 4)
                else:
                    a, state1 = sess.run([main_network.predict, main_network.rnn_state], feed_dict={
                        main_network.scalar_input: [s / 255.],
                        main_network.sequence_length: 1,
                        main_network.state_in: state,
                        main_network.batch_size: 1
                    })
                    a = a[0]

                s1_p, reward, done = env.step(a)
                s1 = process_state(s1_p)
                n_steps += 1

                # save the experience to our episode buffer
                episode_buffer.append(np.reshape(np.array([s, a, reward, s1, done]), [1, 5]))

                if n_steps > n_pretrain_steps:
                    if epsilon > end_epsilon:
                        epsilon -= step_drop

                    if n_steps % update_freq == 0:
                        update_target(target_ops, sess)
                        state_train = (np.zeros([batch_size, n_hidden]), np.zeros([batch_size, n_hidden]))

                        # get a random batch of experiences
                        train_batch = buffer.sample(batch_size, trace_length)

                        # perform the Double-DQN update to the target Q-values
                        q1 = sess.run(main_network.predict, feed_dict={
                            main_network.scalar_input: np.vstack(train_batch[:, 3] / 255.),
                            main_network.sequence_length: trace_length,
                            main_network.state_in: state_train,
                            main_network.batch_size: batch_size
                        })
                        q2 = sess.run(target_network.q_out, feed_dict={
                            target_network.scalar_input: np.vstack(train_batch[:, 3] / 255.),
                            target_network.sequence_length: trace_length,
                            target_network.state_in: state_train,
                            target_network.batch_size: batch_size
                        })
                        end_multiplier = -train_batch[:, 4] - 1
                        double_q = q2[range(batch_size * trace_length), q1]
                        target_q = train_batch[:, 2] + gamma * double_q * end_multiplier

                        # update the network with our target values
                        _ = sess.run(main_network.update_op, feed_dict={
                            main_network.scalar_input: np.vstack(train_batch[:, 0] / 255.),
                            main_network.target_q: target_q,
                            main_network.actions: train_batch[:, 1],
                            main_network.sequence_length: trace_length,
                            main_network.state_in: state_train,
                            main_network.batch_size: batch_size
                        })

                total_reward += reward
                s = s1
                # s_p = s1_p
                state = state1

                if done:
                    break

            buffer_array = np.array(episode_buffer)
            episode_buffer = list(zip(buffer_array))
            buffer.add(episode_buffer)
            js.append(j)
            rewards.append(total_reward)

            # periodically save the model
            if i % 1000 == 0:
                saver.save(sess, '{}/model-{}'.format(save_path, i))
                print('Saved model')

            if len(rewards) % 100 == 0 and i != 0:
                print('Number steps:', n_steps, 'mean reward:', np.mean(rewards[-10:]), 'epsilon:', epsilon)

                # record performance metrics and episode logs for the Control Center
                save_to_monitor(i, rewards, js, np.reshape(np.array(episode_buffer), [len(episode_buffer), 5]),
                                summary_length, n_hidden, sess, main_network, time_per_step)

        saver.save(sess, '{}/model-{}'.format(save_path, i))

        print('Successful episodes %:', str(sum(rewards) / n_episodes))

        return rewards, js


def update_target(op_holder, sess):
    for op in op_holder:
        sess.run(op)


def update_target_graph(tf_vars, tau):
    """ update the parameters of our target network with those of the primary network """
    n_vars = len(tf_vars)
    op_holder = []
    for i, var in enumerate(tf_vars[0:n_vars//2]):
        op_holder.append(tf_vars[i + n_vars//2].assign(var.value() * tau +
                                                       (1 - tau) * tf_vars[i + n_vars//2].value()))
    return op_holder
