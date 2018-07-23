import csv
import moviepy.editor as mpy
import numpy as np
import random
import tensorflow as tf


class ExperienceBuffer(object):
    
    def __init__(self, buffer_size=50000):
        self.buffer = []
        self.buffer_size = buffer_size
        
    def add(self, experience):
        n = len(self.buffer) + len(experience)
        if n >= self.buffer_size:
            self.buffer[0:n-self.buffer_size] = []
            
        self.buffer.extend(experience)
        
    def sample(self, size):
        return np.reshape(np.array(random.sample(self.buffer, size)), [size, 5])

    
class ExperienceBuffer2(object):
    
    def __init__(self, buffer_size=1000):
        self.buffer = []
        self.buffer_size = buffer_size
        
    def add(self, experience):
        n = len(self.buffer) + 1
        if n >= self.buffer_size:
            self.buffer[0:n-self.buffer_size] = []
            
        self.buffer.append(experience)
        
    def sample(self, batch_size, trace_length):
        sampled_episodes = random.sample(self.buffer, batch_size)
        sampled_traces = []
        for episode in sampled_episodes:
            point = np.random.randint(0, len(episode) + 1 - trace_length)
            sampled_traces.append(episode[point:point+trace_length])
            
        return np.reshape(np.array(sampled_traces), [batch_size * trace_length, 5])


def process_state(states):
    """ resize game frames """
    return np.reshape(states, [21168])


def update_target_graph(tf_vars, tau):
    """ update the parameters of our target network with those of the primary network """
    n_vars = len(tf_vars)
    op_holder = []
    for i, var in enumerate(tf_vars[0:n_vars//2]):
        op_holder.append(tf_vars[i + n_vars//2].assign(var.value() * tau +
                                                       (1 - tau) * tf_vars[i + n_vars//2].value()))
    return op_holder


def update_target(op_holder, sess):
    for op in op_holder:
        sess.run(op)

    #total_vars = len(tf.trainable_variables())
    #a = tf.trainable_variables()[0].eval(session=sess)
    #b = tf.trainable_variables()[total_vars//2].eval(session=sess)
    #if a.all() == b.all():
    #    print('Target set success')
    #else:
    #    print('Target set failed')


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
