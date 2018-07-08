Breakout using a Deep Q-Network
-------------------------------

Apply approximate q-learning to an atari game called Breakout.


Processing the game image
^^^^^^^^^^^^^^^^^^^^^^^^^

Raw atari images are large, 210x160x3 by default. However, we don't need
that level of detail in order to learn them.

We can thus save a lot of time by pre-processing game image, including

* Resizing to a smaller shape, 64 x 64
* Converting to grayscale
* Cropping irrelevant image parts (top & bottom)


Network Architecture
^^^^^^^^^^^^^^^^^^^^

A neural network that can map images to state q-values. This network will be called on
every agent's step so it better not be resnet-152 unless you have an array of GPUs.
Instead, you can use strided convolutions with a small number of features to save time
and memory.

The architecture is a classical convolutional neural network with three convolutional
layers, followed by two fully connected layers. People familiar with object recognition
networks may notice that there are no pooling layers. But if you really think about that,
then pooling layers buy you a translation invariance – the network becomes insensitive
to the location of an object in the image. That makes perfect sense for a classification
task like ImageNet, but for games, the location of the ball is crucial in determining
the potential reward and we wouldn't want to discard this information!

See `Demystifying Deep Reinforcement Learning <http://neuro.cs.ut.ee/demystifying-deep-reinforcement-learning/>`_

The network architecture that DeepMind used is as follows:

=====  ========  ===========  ======  ===========  ==========  ========
Layer  Input     Filter size  Stride  Num filters  Activation  Output
=====  ========  ===========  ======  ===========  ==========  ========
conv1  84x84x4   8×8          4       32           ReLU        20x20x32
conv2  20x20x32  4×4          2       64           ReLU        9x9x64
conv3  9x9x64    3×3          1       64           ReLU        7x7x64
fc4    7x7x64                         512          ReLU        512
fc5    512                            18           Linear      18
=====  ========  ===========  ======  ===========  ==========  ========

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