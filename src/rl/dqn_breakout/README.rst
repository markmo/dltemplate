Breakout using a Deep Q-Network
-------------------------------

Apply approximate q-learning to an atari game called Breakout.


Processing the game image
^^^^^^^^^^^^^^^^^^^^^^^^^

Raw atari images are large, 210x160x3 by default. However, we don't need
that level of detail in order to learn them.

We can thus save a lot of time by preprocessing game image, including

* Resizing to a smaller shape, 64 x 64
* Converting to grayscale
* Cropping irrelevant image parts (top & bottom)


Network Architecture
^^^^^^^^^^^^^^^^^^^^

A neural network that can map images to state q-values. This network will be called on
every agent's step so it better not be resnet-152 unless you have an array of GPUs.
Instead, you can use strided convolutions with a small number of features to save time
and memory.

You can build any architecture you want, but for reference, here's something that will
more or less work:

.. image:: ../../../images/dqn_arch.png


Experience Replay
^^^^^^^^^^^^^^^^^

.. image:: ../../../images/exp_replay.png

* exp_replay.add(obsvs, actions, rewards, next_obsvs, done_mask) - saves (s, a, r, s', done)
  tuple into the buffer
* exp_replay.sample(batch_size) - returns observations, actions, rewards, next_observations
  and is_done for batch_size random samples.
* len(exp_replay) - returns number of elements stored in replay buffer.