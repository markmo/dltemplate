Deep Q-Networks and Beyond
--------------------------

Implement a Deep Q-Network using both Double DQN and Dueling DQN.

The agent learns to solve a navigation task in a basic grid world.

This model makes a few improvements on the ordinary Q-network:

1. Going from a single-layer network to a multi-layer convolutional network.
2. Implementing Experience Replay, which allows our network to train itself
   using stored memories from it’s experience.
3. Utilizing a second “target” network, which we use to compute target Q-values
   during updates.

It was these three innovations that enabled the Google DeepMind team to achieve
superhuman performance on dozens of Atari games using a DQN agent.

Model Architecture
^^^^^^^^^^^^^^^^^^

.. image:: ../../../images/q-network.png

Convolutional layers. Instead of considering each pixel independently, convolutional
layers allow us to consider regions of an image, and maintain spatial relationships
between the objects on the screen as we send information up to higher levels of the
network. In this way, they act similarly to human receptive fields.

Experience Replay. The basic idea is that by storing an agent’s experiences, and
then randomly drawing batches of them to train the network, we can more robustly
learn to perform well in the task. By keeping the experiences we draw random, we
prevent the network from only learning about what it is immediately doing in the
environment, and allow it to learn from a more varied array of past experiences.
Each of these experiences are stored as a tuple of <state,action,reward,next state>.
The Experience Replay buffer stores a fixed number of recent memories, and as new
ones come in, old ones are removed. When the time comes to train, we simply draw
a uniform batch of random memories from the buffer, and train our network with them.

Separate Target Network. This second network is used to generate the target-Q values
that will be used to compute the loss for every action during training. Why not use
just use one network for both estimations? The issue is that at every step of
training, the Q-network’s values shift, and if we are using a constantly shifting
set of values to adjust our network values, then the value estimations can easily
spiral out of control. The network can become destabilized by falling into feedback
loops between the target and estimated Q-values. In order to mitigate that risk,
the target network’s weights are fixed, and only periodically or slowly updated to
the primary Q-networks values. In this way training can proceed in a more stable manner.

Instead of updating the target network periodically and all at once, we will be
updating it frequently, but slowly. This technique was introduced in another
`DeepMind paper <https://arxiv.org/pdf/1509.02971.pdf>`_, where they found that it
stabilized the training process.

Next Steps
^^^^^^^^^^

A number of improvements above and beyond the `DQN architecture described by DeepMind
<http://www.davidqiu.com:8888/research/nature14236.pdf>`_, have allowed for even
greater performance and stability.

Double DQN
^^^^^^^^^^

The main intuition behind Double DQN is that the regular DQN often overestimates the
Q-values of the potential actions to take in a given state. While this would be fine
if all actions were always overestimates equally, there was reason to believe this
wasn't the case. You can easily imagine that if certain suboptimal actions regularly
were given higher Q-values than optimal actions, the agent would have a hard time ever
learning the ideal policy. In order to correct for this, the authors of DDQN paper
propose a simple trick: instead of taking the max over Q-values when computing the
target-Q value for our training step, we use our primary network to chose an action,
and our target network to generate the target Q-value for that action. By decoupling
the action choice from the target Q-value generation, we are able to substantially
reduce the overestimation, and train faster and more reliably. Below is the new DDQN
equation for updating the target value.

::

    Q-Target = r + γQ(s’,argmax(Q(s’,a,ϴ),ϴ’))

Dueling DQN
^^^^^^^^^^^

.. image:: ../../../images/dueling_dqn.png

In order to explain the reasoning behind the architecture changes that Dueling DQN
makes, we need to first explain some a few additional reinforcement learning terms.
The Q-values that we have been discussing so far correspond to how good it is to take
a certain action given a certain state. This can be written as Q(s,a). This action
given state can actually be decomposed into two more fundamental notions of value.
The first is the value function V(s), which says simple how good it is to be in any
given state. The second is the advantage function A(a), which tells how much better
taking a certain action would be compared to the others. We can then think of Q as
being the combination of V and A. More formally:

::

    Q(s,a) =V(s) + A(a)

The goal of Dueling DQN is to have a network that separately computes the advantage
and value functions, and combines them back into a single Q-function only at the
final layer. It may seem somewhat pointless to do this at first glance. Why decompose
a function that we will just put back together? The key to realizing the benefit is
to appreciate that our reinforcement learning agent may not need to care about both
value and advantage at any given time. For example: imagine sitting outside in a park
watching the sunset. It is beautiful, and highly rewarding to be sitting there. No
action needs to be taken, and it doesn’t really make sense to think of the value of
sitting there as being conditioned on anything beyond the environmental state you are
in. We can achieve more robust estimates of state value by decoupling it from the
necessity of being attached to specific actions.

(From the `Deep Q-Networks and Beyond blog post
<https://medium.com/@awjuliani/simple-reinforcement-learning-with-tensorflow-part-4-deep-q-networks-and-beyond-8438a3e2b8df>`_.)
