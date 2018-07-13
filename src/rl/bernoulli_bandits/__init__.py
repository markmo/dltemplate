from argparse import ArgumentParser
from common.util import load_hyperparams, merge_dict
import os
from rl.bernoulli_bandits.model_setup import BernoulliBandit
from rl.bernoulli_bandits.model_setup import EpsilonGreedyAgent, ThompsonSamplingAgent, UCBAgent
from rl.bernoulli_bandits.util import get_regret, plot_regret


def run(constant_overwrites):
    config_path = os.path.join(os.path.dirname(__file__), 'hyperparams.yml')
    constants = merge_dict(load_hyperparams(config_path), constant_overwrites)

    agents = [
        EpsilonGreedyAgent(),
        UCBAgent(),
        ThompsonSamplingAgent()
    ]
    n_epochs = constants['n_epochs']
    n_trials = constants['n_trials']
    regret = get_regret(BernoulliBandit(), agents, n_steps=n_epochs, n_trials=n_trials)
    plot_regret(regret, agents)


if __name__ == '__main__':
    # read args
    parser = ArgumentParser(description='Run Bernoulli Bandits')
    parser.add_argument('--epochs', dest='n_epochs', type=int, help='number epochs')
    parser.add_argument('--trials', dest='n_trials', type=int, help='number trials')
    args = parser.parse_args()

    run(vars(args))
