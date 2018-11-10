Word-level CNN (Kim 2014)
-------------------------

See `Convolutional Neural Networks for Sentence Classification <https://arxiv.org/pdf/1408.5882.pdf>`_.

Originally invented for computer vision, CNN models have subsequently been shown to be effective
for NLP and have achieved excellent results in semantic parsing (Yih et al., 2014), search query
retrieval (Shen et al., 2014), sentence modeling (Kalchbrenner et al., 2014), and other traditional
NLP tasks (Collobert et al., 2011).

The first layers embeds words into low-dimensional vectors. The next layer performs convolutions over
the embedded word vectors using multiple filter sizes. For example, sliding over 3, 4 or 5 words at a
time. Next, we max-pool the result of the convolutional layer into a long feature vector, add dropout
regularization, and classify the result using a softmax layer.

We will not enforce L2 norm constraints on the weight vectors. `A Sensitivity Analysis of (and
Practitioners’ Guide to) Convolutional Neural Networks for Sentence Classification
<https://arxiv.org/pdf/1510.03820.pdf>`_ found that the constraints had little effect on the end result.

See also `Implementing a CNN for Text Classification in TensorFlow
<http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/#more-452>`_
by Denny Britz.

To train:

::

    python text_classification_benchmarks/word_cnn/__init__.py

To test:

::

        python text_classification_benchmarks/word_cnn/__init__.py --test --checkpoint-dir=text_classification_benchmarks/word_cnn/runs/<run_id>/checkpoints


Word-level CNN initialised with Word2Vec Embeddings
---------------------------------------------------
::

    python text_classification_benchmarks/word_cnn/__init__.py --word2vec-filename=../../../data/word2vec/GoogleNews-vectors-negative300.bin

Trains the above CNN initialised with word vectors obtained from an unsupervised neural language
model. These vectors were trained by Mikolov et al. (2013) on 100 billion words of Google News.
The vectors have dimensionality of 300 and were trained using the continuous bag-of-words
architecture.

We keep the word vectors static and learn only the other parameters of the model.

.. image:: ../../../images/cnn_classification.png