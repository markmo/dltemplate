Monte-carlo Tree Search (MCTS)
------------------------------

Implement the vanilla MCTS algorithm using UCB1-based node selection.


Extensions
^^^^^^^^^^

UCB1 is a weak bound as it relies on a very general bound. (Hoeffding Inequality, to be exact).

Try playing with alpha. The theoretically optimal alpha for CartPole is 200 (max reward).

Use a different exploration strategy (Bayesian UCB, for example).

Expand not all but several random actions per expand call. See the notes below for details.

The goal is to find out what gives the optimal performance for CartPole-v0 for different time
budgets (ie. different n_iter in plan_mcts.)

Evaluate your results on AcroBot-v1.


Atari
^^^^^

Apply MCTS to play Atari games. In particular, start with 'MsPacman-ramDeterministic-v0'.

This requires two things:

1. Slightly modify the `WithSnapshots` Wrapper to work with Atari.
   * Atari has a special interface for snapshots:
     snapshot = self.env.ale.cloneState()
     ...
     self.env.ale.restoreState(snapshot)
   * Try it on the env above to make sure it does what you told it to.
2. Run MCTS on the game above.
   * Start with a small tree size to speed-up computations
   * You will probably want to rollout for 10-100 steps (t_max) for starters
   * Consider using discounted rewards (see notes at the end)
   * Try a better rollout policy


Integrate learning into planning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Planning on each iteration is a costly thing to do. You can speed things up drastically
if you train a classifier to predict which action will turn out to be best according to
MCTS.

To do so, just record the action the MCTS agent took on each step, and fit something to
[state, mcts_optimal_action].

* You can also use optimal actions from discarded states to get more (dirty) samples.
  Just don't forget to fine-tune without them.
* It's also worth a try to use P(best_action|state) from your model to select best nodes
  in addition to UCB
* If your model is lightweight enough, try using it as a rollout policy.

While CartPole is good to test, try expanding this to 'MsPacmanDeterministic-v0'.

* See previous section on how to wrap Atari
* Also consider what `AlphaGo Zero <https://deepmind.com/blog/alphago-zero-learning-scratch/>`_ did in this area.


Integrate planning into learning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Incorporate planning into the agent architecture.

The goal is to implement `Value Iteration Networks <https://arxiv.org/abs/1602.02867>`_

Try `this example <https://github.com/yandexdataschool/AgentNet/blob/master/examples/Deep%20Kung-Fu%20with%20GRUs%20and%20A2c%20algorithm%20(OpenAI%20Gym).ipynb>`_
for starters.

You will need to switch it into a maze-like game. Consider 'MsPacmanDeterministic-v0'.

You will also need to implement a special layer that performs value iteration-like updates
to a recurrent memory, e.g. using an Attention network.


Assumptions
^^^^^^^^^^^

* Finite actions - we enumerate all actions in expand
* Episodic (finite) MDP - while technically it works for infinite MDP, we rollout for 10^4 steps.
  If you are knowingly infinite, please adjust t_max to something more reasonable.
* No discounted rewards - we assume gamma = 1. If that isn't the case, you only need to change
  two lines in `rollout` and use `my_R = r + gamma * child_R` for propagate.
* pickleable env - won't work if for example your env is connected to a web-browser connected to
  the internet. For custom envs, you may need to modify `get_snapshot`/`load_snapshot` from `WithSnapshots`.

On get_best_leaf and expand functions:

This MCTS implementation only selects leaf nodes for expansion. It doesn't break things down
because `expand` adds all possible actions. Hence, all non-leaf nodes are by design fully
expanded and shouldn't be selected.

If you want to add only a few random actions on each expand, you will also have to modify
`get_best_leaf` to consider returning non-leafs.

Rollout policy:

We use a simple uniform policy for rollouts. This introduces a negative bias to good
situations that can be messed up completely with a random bad action. As a simple example,
if you tend to rollout with uniform policy, better not introduce the agent to sharp knives
or walking near cliffs.

You can improve this by integrating a reinforcement learning algorithm with a computationally
light agent. You can even train this agent on optimal policy found by the tree search.