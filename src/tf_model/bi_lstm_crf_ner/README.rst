Recognize named entities on Twitter using a combined Bidirectional LSTM and CRF Layer
-------------------------------------------------------------------------------------

Using a combined recurrent neural network and conditional random fields (CRFs) to
solve a Named Entity Recognition (NER) problem, as in `this paper <http://www.aclweb.org/anthology/P15-1109>`_.

The benefit of combining a Bi-LSTM and CRF is that it uses the LSTM for feature
engineering, and a CRF to apply constraints, and make use of context, e.g. previous
work and next word, in the tagging decision as well as in determining the tag score.

NER is a common task in natural language processing systems. It serves for extraction
of entities from text, such as persons, organizations, and locations.

For example, we want to extract persons' and organizations' names from the text:

    Ian Goodfellow works for Google Brain

A NER model needs to provide the following sequence of tags:

    B-PER I-PER    O     O   B-ORG  I-ORG

Where B- and I- prefixes stand for the beginning and inside of the entity, while O stands
for out-of-tag or no tag. Markup with this prefix scheme is called BIO markup.

For a given sequence, we look up the embedding of each word and process this vector
with the LSTM layers for the high level representation. After having finished the
whole sequence, we take the representations of all time steps as the input features
for CRF to perform the sequence tagging task. The traditional viterbi decoding is used
for inference. The gradient of the log-likelihood of the tag sequence with respect to
the input of the CRF is calculated and back-propagated to all the LSTM layers to get
the gradient of the parameters.

    On one hand, the LSTM network is capable of capturing long distance dependencies,
    especially in its deep form. On the other hand, traditional feature templates are
    only good at describing the properties in neighborhood, and a small mistake in the
    syntactic tree will result in a large deviation in tagging. Moreover, from the
    analysis of the internal states of the deep network, we see that the model implicitly
    learns to capture some syntactic structure similar to the dependency parsing tree.

| `End-to-end Learning of Semantic Role Labeling Using Recurrent Neural Networks`_,
| Jie Zhou and Wei Xu, Baidu Research

.. _`End-to-end Learning of Semantic Role Labeling Using Recurrent Neural Networks`: http://www.aclweb.org/anthology/P15-1109
