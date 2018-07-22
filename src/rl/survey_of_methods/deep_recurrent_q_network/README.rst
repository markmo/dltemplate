Deep Recurrent Q-Network
------------------------

Implement a Deep Recurrent Q-Network, which can solve Partially Observable Markov Decision Processes.

Environments which follow a structure where a given state conveys everything the agent needs to act
optimally are called Markov Decision Processes (MDPs).

While MDPs provide a nice formalism, almost all real world problems fail to meet this standard.
Take for example your field of view at this very moment. Can you see what is behind you? Even if we
were to have 360 degree vision, we may still not know what is on the other side of a wall just
beyond us. Information outside our view is often essential to making decisions regarding the world.

In addition to being spatially limited, information available at a given moment is also often
temporally limited. When looking at a photo of a ball being thrown between two people, the lack of
motion may make us unable to determine the direction and speed of the ball. In games like Pong, not
only the position of the ball, but also it’s direction and speed are essential to making the
correct decisions.

Environments which present themselves in a limited way to the agent are referred to as Partially
Observable Markov Decision Processes (POMDPs). While they are trickier to solve than their fully
observable counterparts, understanding them is essential to solving most realistic tasks.

Making sense of a limited, changing world
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

How can we build a neural agent which still functions well in a partially observable world? The
key is to give the agent a capacity for temporal integration of observations. The intuition behind
this is simple: if information at a single moment isn’t enough to make a good decision, then enough
varying information over time probably is. Revisiting the photo example of the thrown ball A single
image of a ball in motion tells us nothing about its movements, but two images in sequence allows
us to discern the direction of movement. A longer sequence might even allow us to make sense of the
speed of the ball. The same principle can be applied to problems where there is a limited field of
view. If you can’t see behind you, by turning around you can integrate the forward and backward
views over time and get a complete picture of the world with which to act upon.

Within the context of Reinforcement Learning, there are a number of possible ways to accomplish
this temporal integration. The solution taken by DeepMind in their original paper on Deep
Q-Networks was to stack the frames from the Atari simulator. Instead of feeding the network a
single frame at a time, they used an external frame buffer which kept the last four frames of the
game in memory and fed this to the neural network. This approach worked relatively well for the
simple games they employed, but it isn’t ideal for a number of reasons. The first is that it isn’t
necessarily biologically plausible. When light hits our retinas, it does it at a single moment.
There is no way for light to be stored up and passed all at once to an eye. Secondly, by using
blocks of 4 frames as their state, the experience buffer used needed to be much larger to
accommodate the larger stored states. This makes the training process require a larger amount of
potentially unnecessary memory. Lastly, we may simply need to keep things in mind that happened
much earlier than would be feasible to capture with stacking frames. Sometimes an event hundreds of
frames earlier might be essential to deciding what to do at the current moment. We need a way for
our agent to keep events in mind more robustly.

Recurrent Neural Networks
^^^^^^^^^^^^^^^^^^^^^^^^^

All of these issues can be solved by moving the temporal integration into the agent itself. This is
accomplished by utilizing a recurrent block in our neural agent.

By utilizing a recurrent block in our network, we can pass the agent single frames of the
environment, and the network will be able to change its output depending on the temporal pattern of
observations it receives. It does this by maintaining a hidden state that it computes at every
time-step. The recurrent block can feed the hidden state back into itself, thus acting as an
augmentation which tells the network what has come before. The class of agents which utilize this
recurrent network are referred to as Deep Recurrent Q-Networks (DRQN).

Implementation
^^^^^^^^^^^^^^

In order to implement a Deep Recurrent Q-Network (DRQN) architecture in Tensorflow, we need to make
a few modifications to our DQN.

The first change is to the agent itself. We insert an LSTM recurrent cell between the output of the
last convolutional layer and the input into the split between the Value and Advantage streams. We
can do this by utilizing the tf.nn.dynamic_rnn function and defining a tf.nn.rnn_cell.LSTMCell
which is fed to the rnn node. We also need to slightly alter the training process in order to send
an empty hidden state to our recurrent cell at the beginning of each sequence.

The second main change is to adjust the way our experience buffer stores memories. Since we want to
train our network to understand temporal dependencies, we can’t use random batches of experience.
Instead, we need to be able to draw traces of experience of a given length. In this implementation,
our experience buffer will store entire episodes, and randomly draw traces of 8 steps from a random
batch of episodes. By doing this we both retain our random sampling as well as ensure each trace of
experiences actually follows from one another.

Finally, we utilize a technique developed by a group at Carnegie Mellon who used a DRQN to train
a neural network to play the first person shooter game Doom. Instead of sending all the gradients
backwards when training their agent, they sent only the last half of the gradients for a given
trace. We can do this by simply masking the loss for the first half of each trace in a batch. They
found it improved performance by only sending more meaningful information through the network.

From the `Partial Observability and Deep Recurrent Q-Networks <https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-6-partial-observability-and-deep-recurrent-q-68463e9aeefc>`_
blog post.