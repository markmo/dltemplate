import numpy as np
from rl.reinforce_cartpole.util import get_cumulative_rewards


def test_get_cumulative_rewards():
    assert len(get_cumulative_rewards(range(100))) == 100
    assert np.allclose(get_cumulative_rewards([0, 0, 1, 0, 0, 1, 0], gamma=0.9),
                       [1.40049, 1.5561, 1.729, 0.81, 0.9, 1.0, 0.0])
    assert np.allclose(get_cumulative_rewards([0, 0, 1, -2, 3, -4, 0], gamma=0.5),
                       [0.0625, 0.125, 0.25, -1.5, 1.0, -4.0, 0.0])
    assert np.allclose(get_cumulative_rewards([0, 0, 1, 2, 3, 4, 0], gamma=0),
                       [0, 0, 1, 2, 3, 4, 0])
