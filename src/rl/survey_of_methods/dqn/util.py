import numpy as np
import os
import random
from rl.survey_of_methods.dqn.model_setup import QNetwork
import tensorflow as tf


class ExperienceBuffer(object):
    """
    This class lets us store experiences, then sample them randomly to train the network.
    """

    def __init__(self, buffer_size=50000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        n = len(self.buffer) + len(experience)
        if n >= self.buffer_size:
            self.buffer[0:n - self.buffer_size] = []

        self.buffer.extend(experience)

    def sample(self, size):
        return np.reshape(np.array(random.sample(self.buffer, size)), [size, 5])


def process_state(states):
    """ resize game frames """
    return np.reshape(states, [21168])


def train(env, n_hidden, start_epsilon, end_epsilon, annealing_steps, tau, gamma, learning_rate,
          save_path, load_model, n_episodes, batch_size, max_episode_length, n_pretrain_steps, update_freq):
    """
    Learning should occur in a couple hours on a moderately powerful machine (GTX970).
    (Getting Atari games to work will take at least a day of training on a powerful machine.)

    :return: rewards, list of J values
    """
    tf.reset_default_graph()
    n_actions = env.actions
    main_network = QNetwork(n_hidden, n_actions, learning_rate)
    target_network = QNetwork(n_hidden, n_actions, learning_rate)
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

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if load_model:
            print('Loading model...')
            ckpt = tf.train.get_checkpoint_state(save_path)
            saver.restore(sess, ckpt.model_checkpoint_path)

        i = 0
        for i in range(n_episodes):
            episode_buffer = ExperienceBuffer()

            # reset environment and get first new observation
            s = env.reset()
            s = process_state(s)
            total_reward = 0
            j = 0

            # Train the Q-network
            # if the agent takes longer to reach either of the blocks, then end the trial
            while j < max_episode_length:
                j += 1

                # choose an action greedily from the Q-network,
                # with epsilon chance of random action
                if np.random.rand(1) < epsilon or n_steps < n_pretrain_steps:
                    a = np.random.randint(0, 4)
                else:
                    a = sess.run(main_network.predict, feed_dict={main_network.scalar_input: [s]})[0]

                s1, reward, done = env.step(a)
                s1 = process_state(s1)
                n_steps += 1

                # save the experience to our episode buffer
                episode_buffer.add(np.reshape(np.array([s, a, reward, s1, done]), [1, 5]))

                if n_steps > n_pretrain_steps:
                    if epsilon > end_epsilon:
                        epsilon -= step_drop

                    if n_steps % update_freq == 0:
                        # get a random batch of experiences
                        train_batch = buffer.sample(batch_size)

                        # perform the Double-DQN update to the target Q-values
                        q1 = sess.run(main_network.predict, feed_dict={
                            main_network.scalar_input: np.vstack(train_batch[:, 3])
                        })
                        q2 = sess.run(target_network.q_out, feed_dict={
                            target_network.scalar_input: np.vstack(train_batch[:, 3])
                        })
                        end_multiplier = -train_batch[:, 4] - 1
                        double_q = q2[range(batch_size), q1]
                        target_q = train_batch[:, 2] + gamma * double_q * end_multiplier

                        # update the network with our target values
                        _ = sess.run(main_network.update_op, feed_dict={
                            main_network.scalar_input: np.vstack(train_batch[:, 0]),
                            main_network.target_q: target_q,
                            main_network.actions: train_batch[:, 1]
                        })
                        # update the target network toward the primary network
                        update_target(target_ops, sess)

                total_reward += reward
                s = s1

                if done:
                    break

            buffer.add(episode_buffer.buffer)
            js.append(j)
            rewards.append(total_reward)

            # periodically save the model
            if i % 1000 == 0:
                saver.save(sess, '{}/model-{}'.format(save_path, i))
                print('Saved model')

            if len(rewards) % 10 == 0:
                print('Number steps:', n_steps, 'mean reward:', np.mean(rewards[-10:]), 'epsilon:', epsilon)

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
