Asynchronous Actor-Critic Agents (A3C)
--------------------------------------

An implementation of the A3C algorithm to solve a simple 3D Doom challenge using the VizDoom engine.

.. image:: ../../../../images/a3c_network.png

The A3C algorithm was released by Google’s DeepMind group earlier this year, and it made a splash by
essentially obsoleting DQN. It was faster, simpler, more robust, and able to achieve much better
scores on the standard battery of Deep RL tasks. On top of all that it could work in continuous as
well as discrete action spaces. Given this, it has become the go-to Deep RL algorithm for new
challenging problems with complex state and action spaces. In fact, OpenAI just released a version
of A3C as their “universal starter agent” for working with their new (and very diverse) set of
Universe environments.

**Asynchronous**. Unlike DQN, where a single agent represented by a single neural network interacts with
a single environment, A3C utilizes multiple incarnations of the above in order to learn more
efficiently. In A3C there is a global network, and multiple worker agents which each have their own
set of network parameters. Each of these agents interacts with it’s own copy of the environment at
the same time as the other agents are interacting with their environments. The reason this works
better than having a single agent (beyond the speedup of getting more work done), is that the
experience of each agent is independent of the experience of the others. In this way the overall
experience available for training becomes more diverse.

**Actor-Critic**. Actor-Critic combines the benefits of both value-iteration methods such as Q-learning,
and policy-iteration methods such as Policy Gradient. In the case of A3C, our network will estimate
both a value function V(s) (how good a certain state is to be in) and a policy π(s) (a set of action
probability outputs). These will each be separate fully-connected layers sitting at the top of the
network. Critically, the agent uses the value estimate (the critic) to update the policy (the actor)
more intelligently than traditional policy gradient methods.

**Advantage**. The insight of using advantage estimates rather than just discounted returns is to allow
the agent to determine not just how good its actions were, but how much better they turned out to be
than expected. Intuitively, this allows the algorithm to focus on where the network’s predictions
were lacking.

::

    Advantage: A = Q(s,a) - V(s)

Since we won’t be determining the Q values directly in A3C, we can use the discounted returns (R) as
an estimate of Q(s,a) to allow us to generate an estimate of the advantage.

::

    Estimated Advantage: A = R - V(s)

This model utilizes a slightly different version of advantage estimation with lower variance referred
to as Generalized Advantage Estimation.

