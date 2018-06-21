import numpy as np
from rl.util import MDP


# noinspection SpellCheckingInspection
class FrozenLakeEnv(MDP):
    """
    Winter is here. You and your friends were tossing around a
    frisbee at the park when you made a wild throw that left the
    frisbee out in the middle of the lake. The water is mostly
    frozen, but there are a few holes where the ice has melted. If
    you step into one of those holes, you'll fall into the freezing
    water. At this time, there's an international frisbee shortage,
    so it's absolutely imperative that you navigate across the lake
    and retrieve the disc. However, the ice is slippery, so you won't
    always move in the direction you intend.

    The surface is described using a grid like the following:

        SFFF       (S: starting point, safe)
        FHFH       (F: frozen surface, safe)
        FFFH       (H: hole, fall to your doom)
        HFFG       (G: goal, where the frisbee is located)

    The episode ends when you reach the goal or fall in a hole. You
    receive a reward of 1 if you reach the goal, and zero otherwise.
    """
    MAPS = {
        '4x4': [
            'SFFF',
            'FHFH',
            'FFFH',
            'HFFG'
        ],
        '8x8': [
            'SFFFFFFF',
            'FFFFFFFF',
            'FFFHFFFF',
            'FFFFFHFF',
            'FFFHFFFF',
            'FHHFFFHF',
            'FHFFHFHF',
            'FFFHFFFG'
        ]
    }

    def __init__(self, desc=None, map_name='4x4', slip_chance=0.2):
        if desc is None and map_name is None:
            raise ValueError('Must provide either desc or map_name')
        elif desc is None:
            desc = self.MAPS[map_name]

        assert ''.join(desc).count('S') == 1, 'this implementation supports having exactly one initial state'
        assert all(c in 'SFHG' for c in ''.join(desc)), 'all cells must be either of S, F, H or G'

        self.desc = desc = np.asarray(list(map(list, desc)), dtype='str')
        self.last_action = None

        n_row, n_col = desc.shape
        states = [(i, j) for i in range(n_row) for j in range(n_col)]
        actions = ['left', 'down', 'right', 'up']

        initial_state = states[np.array(desc == b'S').ravel().argmax()]

        def move(row_, col_, movement_):
            if movement_ == 'left':
                col_ = max(col_ - 1, 0)
            elif movement_ == 'down':
                row_ = min(row_ + 1, n_row - 1)
            elif movement_ == 'right':
                col_ = min(col_ + 1, n_col - 1)
            elif movement_ == 'up':
                row_ = max(row_ - 1, 0)
            else:
                raise ValueError('Invalid action %s' % movement_)

            return row_, col_

        transition_probs = {s: {} for s in states}
        rewards = {s: {} for s in states}
        for (row, col) in states:
            if desc[row, col] in 'GH':
                continue

            for action_i in range(len(actions)):
                action = actions[action_i]
                transition_probs[(row, col)][action] = {}
                rewards[(row, col)][action] = {}
                for movement_i in [(action_i - 1) % len(actions), action_i, (action_i + 1) % len(actions)]:
                    movement = actions[movement_i]
                    new_row, new_col = move(row, col, movement)
                    prob = (1. - slip_chance) if movement == action else (slip_chance / 2.)
                    if prob == 0:
                        continue

                    if (new_row, new_col) not in transition_probs[row, col][action]:
                        transition_probs[row, col][action][new_row, new_col] = prob
                    else:
                        transition_probs[row, col][action][new_row, new_col] += prob

                    if desc[new_row, new_col] == 'G':
                        rewards[row, col][action][new_row, new_col] = 1.0

        MDP.__init__(self, transition_probs, rewards, initial_state)

    def render(self):
        desc_copy = np.copy(self.desc)
        desc_copy[self._current_state] = '*'
        print('\n'.join(map(''.join, desc_copy)), end='\n\n')
