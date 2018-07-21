Model-based RL
--------------

Implement a policy and model network, which work in tandem to solve the CartPole
reinforcement learning problem.

.. image:: ../../../images/model-based-rl.png

In this case, a model is going to be a neural network that attempts to learn the
dynamics of the real environment. For example, in the CartPole challenge, we would
like a model to be able to predict the next position of the Cart given the previous
position and an action. By learning an accurate model, we can train our agent using
the model rather than requiring to use the real environment every time. While this
may seem less useful when the real environment is itself a simulation, like in our
CartPole task, it can have huge advantages when attempting to learn policies for
acting in the physical world.

Our training procedure will involve switching between training our model using the
real environment, and training our agent’s policy using the model environment.