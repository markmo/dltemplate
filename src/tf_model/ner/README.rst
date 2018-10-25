Recognize named entities on Twitter using a Bidirectional LSTM
--------------------------------------------------------------

Using a recurrent neural network to solve a Named Entity Recognition (NER) problem.
NER is a common task in natural language processing systems. It serves for extraction
of entities from text, such as persons, organizations, and locations.

For example, we want to extract persons' and organizations' names from the text:

::

    Ian Goodfellow works for Google Brain

A NER model needs to provide the following sequence of tags:

::

    B-PER I-PER    O     O   B-ORG  I-ORG

Where B- and I- prefixes stand for the beginning and inside of the entity, while O stands
for out-of-tag or no tag. Markup with this prefix scheme is called BIO markup.

Bi-LSTM is one of the state of the art approaches for solving NER problem and it
outperforms other classical methods, despite the modest training corpus. The
implemented model outperforms classical CRFs for this task. However, improvements
might be made by using combination of bidirectional LSTM, CNN and CRF according to
`this paper <https://arxiv.org/abs/1603.01354>`_.