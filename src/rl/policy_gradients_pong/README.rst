Pong from Pixels
----------------

.. image:: ../../../images/policy_network_pong.png

In ordinary supervised learning we would feed an image to the network and get some
probabilities, e.g. for two classes UP and DOWN. I’m showing log probabilities
(-1.2, -0.36) for UP and DOWN instead of the raw probabilities (30% and 70% in this
case) because we always optimize the log probability of the correct label (this makes
math nicer, and is equivalent to optimizing the raw probability because log is monotonic).

.. image:: ../../../images/policy_gradient_example_sup_learning.png

**Policy Gradients**. Suppose our policy network calculated probability of going UP
as 30% (log_prob -1.2) and DOWN as 70% (log_prob -0.36). We will now sample an action
from this distribution; e.g. suppose we sample DOWN, and we will execute it in the game.
At this point notice one interesting fact: we could immediately fill in a gradient of
1.0 for DOWN as we did in supervised learning, and find the gradient vector that would
encourage the network to be slightly more likely to do the DOWN action in the future.
So we can immediately evaluate this gradient and that’s great, but the problem is that
at least for now we do not yet know if going DOWN is good. But the critical point is
that that’s okay, because we can simply wait a bit and see! For example in Pong, we
could wait until the end of the game, then take the reward we get (either +1 if we won
or -1 if we lost), and enter that scalar as the gradient for the action we have taken
(DOWN in this case). In the example below, going DOWN ended up losing the game (-1 reward).
So if we fill in -1 for log probability of DOWN and do backprop, we will find a gradient
that discourages the network to take the DOWN action for that input in the future (and
rightly so, since taking that action led to us losing the game).

.. image:: ../../../images/policy_gradient_example_rl.png

And that’s it: we have a stochastic policy that samples actions and then actions that
happen to eventually lead to good outcomes get encouraged in the future, and actions taken
that lead to bad outcomes get discouraged. Also, the reward does not even need to be +1
or -1 if we win the game eventually. It can be an arbitrary measure of some kind of
eventual quality. For example if things turn out really well it could be 10.0, which we
would then enter as the gradient instead of -1 to start off backprop. That’s the beauty
of neural nets; Using them can feel like cheating: You’re allowed to have 1 million
parameters embedded in 1 teraflop of compute and you can make it do arbitrary things
with SGD. It shouldn't work, but amusingly we live in a universe where it does.

Reinforcement learning is exactly like supervised learning, but on a continuously changing
dataset (the episodes), scaled by the advantage, and we only want to do one (or very few)
updates based on each sampled dataset.

See `Deep Reinforcement Learning: Pong from Pixels <http://karpathy.github.io/2016/05/31/rl/>`_
by Andrej Karpathy.

To run
^^^^^^

1. Create a Python virtual environment.
2. Install the dependencies in requirements.txt
3. Run the code. Change to the `src/rl/policy_gradients_pong` directory.

::

    export PYTHONPATH=.
    python __init__.py

I've included the model file `model.pkl`, which the code will read and start playing
to a winning level. To use, set the `resume` variable at the top of the file to `True`.
Set the `render` variable to `True` to see the game in action.

You can train a model from scratch by setting the `resume` variable to `False`.
On my MacBook Pro, it took a few days of game play before the agent started to
consistently win. Set the `render` variable to `False` to speed up training time.

I'm still impressed that so little code, with a fairly simple algorithm, can win
at Pong purely through observation of the pixels from the frame-by-frame raw pixel
values.