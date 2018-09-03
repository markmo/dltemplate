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

Using `Kaggle San Francisco Crime <https://www.kaggle.com/c/sf-crime/data>`_:

Input: 'Descript' column
Output: 'Category' column
Examples:

==========================================  =============
Descript                                    Category
==========================================  =============
GRAND THEFT FROM LOCKED AUTO                LARCENY/THEFT
POSSESSION OF NARCOTICS PARAPHERNALIA       DRUG/NARCOTIC
AIDED CASE, MENTAL DISTURBED                NON-CRIMINAL
AGGRAVATED ASSAULT WITH BODILY FORCE        ASSAULT
ATTEMPTED ROBBERY ON THE STREET WITH A GUN  ROBBERY
==========================================  =============

Train
^^^^^
::

    export PYTHONPATH=.
    python tf_model/text_classifier/__init__.py --train


Predict
^^^^^^^
::

    export PYTHONPATH=.
    python tf_model/text_classifier/__init__.py
