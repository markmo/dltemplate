Create a Calculator using Machine Learning
------------------------------------------

Use a sequence-to-sequence model to build a calculator for evaluating arithmetic expressions.
Take an equation as input to the neural network and produce an answer as output.

.. image:: ../../../images/encoder-decoder-pic.png

The picture above shows the use of special characters, e.g. see how the start symbol ^ is used.
The transparent parts are ignored. In the decoder, it is masked out in the loss computation.
In the encoder, the green state is considered as final and passed to the decoder.


Data Preparation
^^^^^^^^^^^^^^^^

First of all, you need to understand what is the basic unit of the sequence in your task.
In this case, we operate on symbols, therefore the basic unit is a symbol. The number of
symbols is small, so we don't need to apply filtering or normalization. However, in other
tasks the basic unit is often a word, in which case the mapping would be word to integer.
The number of words might be huge, so it would be reasonable to filter them, for example,
by frequency and keep only the frequent ones. Other strategies that you could consider
include:

* data normalization (lower-casing, tokenization, special treatment of punctuation marks)
* separate vocabulary for input and output (as for machine translation)
* some specifics of the task


Architecture
^^^^^^^^^^^^

The encoder-decoder pattern is a successful architecture for sequence-to-sequence tasks that
have different lengths of input and output sequences. The main idea is to use two recurrent
neural networks, where the first network encodes the input sequence into a real-valued vector
and then the second network decodes this vector into the output sequence.

You should see a loss value of approximately 2.7 at the beginning of the training, and near 1
after the 10th epoch.
