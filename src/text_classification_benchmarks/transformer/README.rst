Transformer - Attention is all you need
---------------------------------------

See `Attention Is All You Need <https://arxiv.org/pdf/1706.03762.pdf>`_.

Self-attention, sometimes called intra-attention is an attention mechanism relating
different positions of a single sequence in order to compute a representation of the
sequence. Self-attention has been used successfully in a variety of tasks including
reading comprehension, abstractive summarization, textual entailment and learning
task-independent sentence representations.

To the best of our knowledge, the Transformer is the first transduction model relying
entirely on self-attention to compute representations of its input and output without
using sequence aligned RNNs or convolution.

.. image:: ../../../images/transformer.png


Problems with RNNs / Motivation for Transformers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Sequential computation prevents parallelization
* Despite GRUs and LSTMs, RNNs still need attention mechanism to deal with long range
  dependencies – path length for codependent computation between states grows with sequence
* But if attention gives us access to any state... maybe we don’t need the RNN?


Please note!!
^^^^^^^^^^^^^

GoogleNews-vectors-negative300.bin is ISO-8859-1 encoded, not utf-8 encoded.
`word2vec <https://github.com/danielfrg/word2vec>`_
`drops the first character <https://github.com/nicholas-leonard/word2vec/issues/25>`_
of words loaded from GoogleNews-vectors-negative300.bin. In the file `wordvectors.py`
on line 171 they read an extra character after each vector. This just sends the first
letter to nowhere. If you comment this line out then it works i.e.::

    171: #fin.read(1)  # newline

I'm assuming other pretrained word2vec files have a newline after each word, and the
above file does not. Using the gensim package instead.