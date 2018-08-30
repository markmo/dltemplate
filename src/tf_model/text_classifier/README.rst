Multi-class Text Classification using a CNN and RNN
---------------------------------------------------

This model achieves good classification performance across a range of text classification tasks
(like Sentiment Analysis) and has since become a standard baseline for new text classification
architectures.

.. image:: ../../../images/text_classification_cnn.png

The first layer embeds words into low-dimensional vectors. The next layer performs convolutions
over the embedded word vectors using multiple filter sizes. For example, sliding over 3, 4 or 5
words at a time. Next, we max-pool the result of the convolutional layer into a long feature
vector, add dropout regularization, and classify the result using a softmax layer.

See `Implementing a CNN for Text Classification in TensorFlow <http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/>`_.