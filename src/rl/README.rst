Reinforcement Learning
----------------------

Survey of Methods
^^^^^^^^^^^^^^^^^

1. `Q-Table Learning Agent <q_table_learning/>`_
2. `Multi-armed Bandit <multi_armed_bandit/>`_ (TensorFlow)
3. `Contextual Bandits <contextual_bandits/>`_ (TensorFlow)
4. `Vanilla Policy Gradient Agent <policy_gradient/>`_ (TensorFlow)
5. `Model-based example for RL <model_based/>`_ (TensorFlow)
6. `Deep Q-Network <dqn/>`_ (TensorFlow)


Frame Buffer
^^^^^^^^^^^^

Environments which follow a structure where a given state conveys everything
the agent needs to act optimally are called Markov Decision Processes (MDPs).

While MDPs provide a nice formalism, almost all real world problems fail to
meet this standard. Take for example your field of view at this very moment.
Can you see what is behind you? Information outside our view is often essential
to making decisions regarding the world.

In addition to being spatially limited, information available at a given moment
is also often temporally limited. When looking at a photo of a ball being
thrown between two people, the lack of motion may make us unable to determine
the direction and speed of the ball. In games like Pong, not only the position
of the ball, but also it’s direction and speed are essential to making the
correct decisions.

Environments which present themselves in a limited way to the agent are
referred to as Partially Observable Markov Decision Processes (POMDPs).
While they are trickier to solve than their fully observable counterparts,
understanding them is essential to solving most realistic tasks.

How can we build a neural agent which still functions well in a partially
observable world? The key is to give the agent a capacity for temporal
integration of observations.

Within the context of Reinforcement Learning, there are a number of possible
ways to accomplish this temporal integration. The solution taken by `DeepMind
in their original paper <https://www.nature.com/articles/nature14236>`_
on Deep Q-Networks was to stack the frames from the Atari simulator. Instead
of feeding the network a single frame at a time, they used an external frame
buffer which kept the last four frames of the game in memory and fed this to
the neural network.


Recurrent Neural Networks
^^^^^^^^^^^^^^^^^^^^^^^^^

All of these issues can be solved by moving the temporal integration into the
agent itself. This is accomplished by utilizing a recurrent block in our
neural agent.

The class of agents which utilize this recurrent network are referred to as
`Deep Recurrent Q-Networks (DRQN) <https://arxiv.org/abs/1507.06527>`_.

We need to adjust the way our experience buffer stores memories. Since we want
to train our network to understand temporal dependencies, we can’t use random
batches of experience. Instead we need to be able to draw traces of experience
of a given length. In this implementation, our experience buffer will store
entire episodes, and randomly draw traces of 8 steps from a random batch of
episodes. By doing this we both retain our random sampling as well as ensure
each trace of experiences actually follows from one another.

We will be utilizing a technique `developed by a group at Carnegie Mellon who
used a DRQN to train a neural network to play the first person shooter game
Doom <https://arxiv.org/abs/1609.05521>`_. Instead of sending all the
gradients backwards when training their agent, they sent only the last half
of the gradients for a given trace.

See https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-6-partial-observability-and-deep-recurrent-q-68463e9aeefc