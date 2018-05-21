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
