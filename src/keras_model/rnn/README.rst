Generating names with recurrent neural networks
-----------------------------------------------

We can define a recurrent neural network as a consecutive application of dense layer to input xt
and previous rnn state ht.

.. image:: ../../../images/rnn.png

Since we're training a language model, there should also be:

* An embedding layer that converts character id x_t to a vector
* An output layer that predicts probabilities of next character
