Kung-Fu Master with Advantage Actor-Critic
------------------------------------------

Implement a deep reinforcement learning agent for the Atari Kung-Fu Master Game
and train it with Advantage Actor-Critic (AAC).

.. image:: ../../../images/Kung-Fu-Master.jpg

The agent is a convolutional neural network that converts states into action
probabilities π and state values V.


Pre-processing
^^^^^^^^^^^^^^

* Image resized to 42x42 and converted to grayscale to run faster
* Rewards divided by 100 'cuz they are all divisible by 100
* Agent sees last 4 frames of game to account for object velocity


Training on parallel games
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. image:: ../../../images/env_pool.png

To make actor-critic training more stable, you can play several games in parallel.
To do this, initialize several parallel gym environments to which to send the agent's
actions, and `reset` each environment if it terminates.
