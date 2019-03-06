lda2vec
-------

The lda2vec model tries to mix the best parts of word2vec and LDA into a single framework.
Word2vec captures relationships between words, but the resulting vectors are largely
uninterpretable and don't represent documents. LDA on the other hand is quite interpretable,
but doesn't model local word relationships like word2vec.

This model builds both word and document topics, makes them interpretable, makes topics over
features and documents, and makes topics that can be supervised and used to predict another
target.

lda2vec also includes more contexts and features than LDA. LDA dictates that words are generated
by a document vector; but we might have all kinds of 'side-information' that could influence the
topics. Example features might include a comment about a particular item, written at a particular
time and in a particular region.

Adapted from `@nateraw <https://github.com/nateraw/Lda2vec-Tensorflow>`_, which is adapted from
`@meereeum <https://github.com/meereeum/lda2vec-tf>`_, which is adapted from
`@cemoody <https://github.com/cemoody/lda2vec>`_ (code for the `original paper <https://arxiv.org/abs/1605.02019>`_
by Chris Moody). See `Mixing Dirichlet Topic Models and Word Embeddings to Make lda2vec <https://arxiv.org/abs/1605.02019>`_.