from argparse import ArgumentParser
from common.util import merge_dict
import gym
from keras_model.dqn.hyperparams import get_constants
from keras_model.dqn.model_setup import DQNAgent
import numpy as np
import os


OUTPUT_DIR = os.path.expanduser('~/src/DeepLearning/dltemplate/output/model_output/cartpole')


def run(constant_overwrites):
    env = gym.make('CartPole-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    constants = merge_dict(get_constants(), constant_overwrites)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    agent = DQNAgent(state_size, action_size, constants)

    n_episodes = constants['n_episodes']
    batch_size = constants['batch_size']
    for e in range(n_episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for time in range(5000):
            env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print('episode: {}/{}, score: {}, e: {:.2}'.format(e, n_episodes, time, agent.epsilon))
                break

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

            if e % 50 == 0:
                agent.save(OUTPUT_DIR + 'weights_' + '{:04d}'.format(e) + '.hdf5')


if __name__ == "__main__":
    # read args
    parser = ArgumentParser(description='Run CartPole DQN')
    parser.add_argument('--epochs', dest='n_epochs', type=int, help='number epochs')
    parser.add_argument('--learning_rate', dest='learning_rate', type=float, help='learning_rate')
    parser.add_argument('--model-filename', dest='model_filename', help='model filename')
    parser.add_argument('--retrain', dest='retrain', help='retrain flag', action='store_true')
    parser.set_defaults(retrain=False)
    args = parser.parse_args()
    run(vars(args))
