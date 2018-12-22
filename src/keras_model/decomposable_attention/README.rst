Decomposable Attention to identify question pairs that have the same intent
---------------------------------------------------------------------------

Our approach applies an NLP task known as Textual Entailment. Textual Entailment (TE) models take a pair
of sentences and predict whether the facts in the first necessarily imply the facts in the second one.
In this case, rather than the usual: Entails, Contradicts, Neutral judgments, the model predicts is-similar
or not.

Read `A Decomposable Attention Model for Natural Language Inference <https://arxiv.org/abs/1606.01933>`_.

A large body of work based on neural networks for text similarity tasks including NLI has been published
in recent years (Hu et al., 2014; Rocktaschel et al., 2016; Wang and Jiang, 2016; Yin et al., 2016,
inter alia). The dominating trend in these models is to build complex, deep text representation models,
for example, with convolutional networks (LeCun et al., 1990, CNNs henceforth) or long short-term memory
networks (Hochreiter and Schmidhuber, 1997, LSTMs henceforth) with the goal of deeper sentence comprehension.

This approach uses attention to decompose the problem into sub-problems that can be solved separately, thus
making it trivially parallelizable.

Given two sentences, where each word is represented by an embedding vector, we first create a soft alignment
matrix using neural attention (Bahdanauet al., 2015). We then use the (soft) alignment to decompose the task
into sub-problems that are solved separately. Finally, the results of these sub-problems are merged to produce
the final classification. In addition, we optionally apply intra-sentence attention (Cheng et al., 2016) to
endow the model with a richer encoding of substructures prior to the alignment step.

Let `a = (a1, ..., a_l_a)` and `b = (b1, ..., b_l_b)` be the two input sentences of length `l_a` and `l_b`,
respectively. We assume that each `a_i, b_j ∈ R^d` is a word embedding vector of dimension `d` and that each
sentence is prepended with a "NULL" token. Our training data comes in the form of labeled pairs
`{a^(n), b^(n), y^(n)}^N_(n=1)^N`, where `y^(n) = (y^(n)_1, ..., y^(n)_C)` is an indicator vector encoding
the label and `C` is the number of output classes. At test time, we receive a pair of sentences `(a, b)` and
our goal is to predict the correct label `y`.

.. image:: ../../../images/decomposable_attn.png


Second model - Enhanced Sequential Inference Model (ESIM)
---------------------------------------------------------

Read `Enhanced LSTM for Natural Language Inference <https://arxiv.org/pdf/1609.06038.pdf>`_.

The model takes as input two sentences `a = (a_1, ..., a_(l_a))` and `b = (b_1, ..., b_(l_b))`,
where `a` is a premise and `b` a hypothesis. The `a_i` or `b_j ∈ R^l` is an embedding of
`l`-dimensional vector, which can be initialized with some pre-trained word embeddings and
organized with parse trees. The goal is to predict a label y that indicates the logic relationship
between `a` and `b`.

A Bidirectional-LSTM (BiLSTM) is used to encode the input premise and hypothesis. Later we will also
use BiLSTM to perform inference composition to construct the final prediction, where BiLSTM encodes
local inference information and its interaction.


Create embeddings file
^^^^^^^^^^^^^^^^^^^^^^

::

    ~/src/DeepLearning/fastText/fasttext skipgram -input data.txt -output ft -dim 300
