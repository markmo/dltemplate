import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rl.dqn_breakout.model_setup import PreprocessAtariImage
from rl.util import evaluate, FrameBuffer, load_weights_into_target_network
from rl.util import play_and_record, ReplayBuffer, sample_batch
from tqdm import trange


def make_env():
    env = gym.make('BreakoutDeterministic-v0')
    env = PreprocessAtariImage(env)
    env = FrameBuffer(env, n_frames=4, dim_order='tensorflow')
    return env


def train(agent, target_network, env, sess, train_step, td_loss, n_epochs=10**5, batch_size=64):
    exp_replay = ReplayBuffer(n_epochs)
    mean_rw_history = []
    td_loss_history = []
    for i in trange(n_epochs):
        play_and_record(agent, env, exp_replay, n_steps=10)
        _, loss_t = sess.run([train_step, td_loss], sample_batch(exp_replay, batch_size))
        td_loss_history.append(loss_t)
        if i % 500 == 0:
            load_weights_into_target_network(agent, target_network)
            agent.epsilon = max(agent.epsilon * 0.99, 0.01)
            mean_rw_history.append(evaluate(make_env(), agent, n_games=3))

        if i % 100 == 0:
            print('buffer size = %i, epsilon = %.5f' % (len(exp_replay), agent.epsilon))
            plt.subplot(1, 2, 1)
            plt.title('mean reward per game')
            plt.plot(mean_rw_history)
            plt.grid()

            assert not np.isnan(loss_t)
            plt.figure(figsize=[12, 4])
            plt.subplot(1, 2, 2)
            plt.title('TD loss history (moving average)')
            plt.plot(pd.ewma(np.array(td_loss_history), span=100, min_periods=100))
            plt.grid()
            plt.show()
