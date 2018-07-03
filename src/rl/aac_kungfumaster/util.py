import gym
from gym.core import Wrapper
from gym.spaces.box import Box
import numpy as np
from scipy.misc import imresize


def evaluate(agent, env, sess, n_games=1):
    """ Plays a game from start till done, returns per game rewards """
    game_rewards = []
    for _ in range(n_games):
        state = env.reset()
        total_reward = 0
        while True:
            action = sample_actions(agent.step(sess, [state]))[0]
            state, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                break

        game_rewards.append(total_reward)

    return game_rewards


def make_env():
    env = gym.make('KungFuMasterDeterministic-v0')
    env = PreprocessAtari(env, height=42, width=42,
                          crop=lambda img: img[60:-30, 5:],
                          dim_order='tensorflow',
                          color=False,
                          n_frames=4,
                          reward_scale=0.01)
    return env


class PreprocessAtari(Wrapper):

    def __init__(self, env, height=42, width=42, color=False,
                 crop=lambda img: img,
                 n_frames=4,
                 dim_order='theano',
                 reward_scale=1.):
        """ A gym wrapper that reshapes, crops and scales images into the desired shapes """
        super().__init__(env)
        assert dim_order in ['theano', 'tensorflow']
        self.img_size = (height, width)
        self.crop = crop
        self.color = color
        self.dim_order = dim_order
        self.reward_scale = reward_scale

        n_channels = (3 * n_frames) if color else n_frames
        obs_shape = [n_channels, height, width] if dim_order == 'theano' else [height, width, n_channels]
        self.observation_space = Box(0.0, 1.0, obs_shape)
        self.framebuffer = np.zeros(obs_shape, 'float32')

    def reset(self):
        """ resets, returns initialized frames """
        self.framebuffer = np.zeros_like(self.framebuffer)
        self.update_buffer(self.env.reset())
        return self.framebuffer

    def step(self, action):
        """ Plays for 1 step, returns frame buffer """
        new_img, reward, done, info = self.env.step(action)
        self.update_buffer(new_img)
        return self.framebuffer, reward * self.reward_scale, done, info

    def update_buffer(self, img):
        img = self.preproc_image(img)
        offset = 3 if self.color else 1
        if self.dim_order == 'theano':
            axis = 0
            cropped_framebuffer = self.framebuffer[:-offset]
        else:
            axis = -1
            cropped_framebuffer = self.framebuffer[:, :, :-offset]

        self.framebuffer = np.concatenate([img, cropped_framebuffer], axis=axis)

    def preproc_image(self, img):
        img = self.crop(img)
        img = imresize(img, self.img_size)
        if not self.color:
            img = img.mean(-1, keepdims=True)

        if self.dim_order == 'theano':
            img = img.transpose([2, 0, 1])  # [h, w, c] to [c, h, w]

        img = img.astype('float32') / 255.
        return img


def sample_actions(agent_outputs):
    """ Picks actions given numeric agent outputs (np arrays) """
    logits, state_values = agent_outputs
    policy = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
    return np.array([np.random.choice(len(p), p=p) for p in policy])
