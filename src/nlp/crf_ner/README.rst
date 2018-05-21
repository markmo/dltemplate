Recognize named entities on Twitter using CRF
---------------------------------------------

Using conditional random fields (CRF) to create a named entity recognition (NER) model.

For example, we want to extract persons' and organizations' names from the text:

    Ian Goodfellow works for Google Brain

A NER model needs to provide the following sequence of tags:

    B-PER I-PER    O     O   B-ORG  I-ORG

Where B- and I- prefixes stand for the beginning and inside of the entity, while O stands
for out-of-tag or no tag. Markup with this prefix scheme is called BIO markup.

When dealing with textual data you need to find a way to convert text to numbers. A common
approach is to use embeddings like word2vec or doc2vec. Another or complementary approach
is to use feature functions; ad-hoc mappings from words to numbers. For example, a feature
function could emphasize adjective and assign the number one when an adjective is found and
zero otherwise. The feature function could look at the previous word or the next-next-word
as well.

For a thorough overview, read `"An introduction to conditional random fields"`_ by Charles
Sutton and Andrew McCallum in Machine Learning Vol. 4, No. 4 (2011) 267–373.

.. _`"An introduction to conditional random fields"`: http://homepages.inf.ed.ac.uk/csutton/publications/crftut-fnt.pdf