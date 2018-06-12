from argparse import ArgumentParser
from common.util import load_hyperparams, merge_dict
from IPython.display import clear_output
import numpy as np
import os
from rl.frozen_lake.utils import FrozenLakeEnv
from rl.utils import draw_policy, get_action_value, get_optimal_action, MDP
from time import sleep


def get_new_state_value(mdp, state_values, state, gamma):
    """
    Computes next V(s).

    Please do not change state_values in process.
    """
    if mdp.is_terminal(state):
        return 0

    return max([get_action_value(mdp, state_values, state, a, gamma)
                for a in mdp.get_possible_actions(state)])


def value_iteration(mdp, state_values=None, gamma=0.9, num_iter=1000, min_difference=1e-5):
    """
    Performs num_iter value iteration steps starting from state_values.
    """
    # initialize V(s)
    state_values = state_values or {s: 0 for s in mdp.get_all_states()}

    for i in range(num_iter):
        # Compute new state values using the functions defined above.
        # It must be a dict {state: new_V(state)}
        new_state_values = {s: get_new_state_value(mdp, state_values, s, gamma)
                            for s, _ in state_values.items()}
        assert isinstance(new_state_values, dict)

        # Compute difference
        diff = max(abs(new_state_values[s] - state_values[s]) for s in mdp.get_all_states())

        # print('iter %4i   |   diff: %6.5f   |   ' % (i, diff), end='')
        # print('   '.join('V(%s) = %.3f' % (s, v) for s, v in state_values.items()), end='\n\n')
        print('iter %4i   |   diff: %6.5f   |   V(start): %.3f ' % (i, diff, new_state_values[mdp.initial_state]))

        state_values = new_state_values

        if diff < min_difference:
            print('Terminated')
            break

    return state_values


def visualize_frozen_lake_actions(map_name='4x4', slip_chance=0., n_iter=100, gamma=0.9):
    mdp = FrozenLakeEnv(map_name=map_name, slip_chance=slip_chance)
    mdp.render()

    state_values = value_iteration(mdp)

    s = mdp.reset()
    mdp.render()

    for t in range(n_iter):
        a = get_optimal_action(mdp, state_values, s, gamma)
        print(a, end='\n\n')
        s, r, done, _ = mdp.step(a)
        mdp.render()
        if done:
            break


def visualize_frozen_lake_value_iteration(map_name='4x4', slip_chance=0., n_iter=30):
    mdp = FrozenLakeEnv(map_name=map_name, slip_chance=slip_chance)
    mdp.render()

    state_values = value_iteration(mdp)

    for i in range(n_iter):
        clear_output(True)
        # print('after iteration %i' % i)
        state_values = value_iteration(mdp, state_values, num_iter=1)
        draw_policy(mdp, state_values)
        sleep(0.5)


def run_frozen_lake(map_name='4x4', slip_chance=0., n_iter=1000, gamma=0.9):
    print('Frozen Lake Game Play:')
    print('map_name=%s, slip_chance=%.2f, n_iter=%i, gamma=%.2f' % (map_name, slip_chance, n_iter, gamma))
    print('')
    mdp = FrozenLakeEnv(map_name=map_name, slip_chance=slip_chance)
    mdp.render()

    state_values = value_iteration(mdp)

    total_rewards = []
    for game_i in range(n_iter):
        # print('after iteration %i' % game_i)
        s = mdp.reset()
        rewards = []
        for t in range(100):
            s, r, done, _ = mdp.step(get_optimal_action(mdp, state_values, s, gamma))
            rewards.append(r)
            if done:
                break

        total_rewards.append(np.sum(rewards))

    print('average reward: ', np.mean(total_rewards))


def run(constant_overwrites):
    config_path = os.path.join(os.path.dirname(__file__), 'hyperparams.yml')
    constants = merge_dict(load_hyperparams(config_path), constant_overwrites)

    transition_probs = {
        's0': {
            'a0': {'s0': 0.5, 's2': 0.5},
            'a1': {'s2': 1}
        },
        's1': {
            'a0': {'s0': 0.7, 's1': 0.1, 's2': 0.2},
            'a1': {'s1': 0.95, 's2': 0.05}
        },
        's2': {
            'a0': {'s0': 0.4, 's1': 0.6},
            'a1': {'s0': 0.3, 's1': 0.3, 's2': 0.4}
        }
    }
    rewards = {
        's1': {'a0': {'s0': +5}},
        's2': {'a1': {'s0': -1}}
    }
    mdp = MDP(transition_probs, rewards, initial_state='s0')

    print('initial state =', mdp.reset())

    next_state, reward, done, info = mdp.step('a1')

    print('next_state = %s, reward = %s, done = %s' % (next_state, reward, done))

    # See rl.utils for function to calculate action value, and to get optimal action

    gamma = constants['gamma']
    n_iter = constants['n_iter']
    min_difference = constants['min_difference']

    # Run baseline scenario

    # initialize V(s)
    state_values = {s: 0 for s in mdp.get_all_states()}

    value_iteration(mdp, None, gamma, n_iter, min_difference)

    # Measure agent's average reward
    s = mdp.reset()
    rewards = []
    for _ in range(10000):
        s, r, done, _ = mdp.step(get_optimal_action(mdp, state_values, s, gamma))
        rewards.append(r)

    print('average reward: ', np.mean(rewards))

    # Run Frozen Lake Scenarios

    print('\n')
    visualize_frozen_lake_actions(map_name='4x4', slip_chance=0, n_iter=100, gamma=gamma)

    print('\n')
    visualize_frozen_lake_value_iteration(map_name='8x8', slip_chance=0, n_iter=30)

    for slip_chance in [0, 0.1, 0.25]:
        print('\n')
        run_frozen_lake(map_name='4x4', slip_chance=slip_chance, n_iter=1000, gamma=gamma)

    print('\n')
    run_frozen_lake(map_name='8x8', slip_chance=0.2, n_iter=1000, gamma=gamma)


if __name__ == '__main__':
    # read args
    parser = ArgumentParser(description='Run Frozen Lake RL model')
    parser.add_argument('--iters', dest='n_iter', type=int, help='number iterations')
    parser.add_argument('--gamma', dest='gamma', type=float, help='gamma')
    parser.add_argument('--min-difference', dest='min_difference', type=float, help='threshold to stop value iteration')
    args = parser.parse_args()

    run(vars(args))
