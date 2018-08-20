Contextual Bandits
------------------

This tutorial contains a simple example of how to build a policy-gradient based agent
that can solve the contextual bandit problem.

There is a set of problems in between the stateless situation and the full RL problem.

.. image:: ../../../../images/context_bandit.png

Contextual Bandits introduce the concept of the state. The state consists of a description
of the environment that the agent can use to take more informed actions. In our problem,
instead of a single bandit, there can now be multiple bandits. The state of the environment
tells us which bandit we are dealing with, and the goal of the agent is to learn the best
action not just for a single bandit, but for any number of them. Since each bandit will
have different reward probabilities for each arm, our agent will need to learn to condition
its action on the state of the environment.

This example solves problems in which there are states, but the states are not determined
by the previous states or actions.


See `Simple Reinforcement Learning with Tensorflow Part 1.5: Contextual Bandits
<https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-1-5-contextual-bandits-bff01d1aad9c>`_.