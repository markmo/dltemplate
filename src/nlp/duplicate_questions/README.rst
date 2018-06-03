Detect Duplicate Questions on StackOverflow
-------------------------------------------

Calculate similarity of pieces of text.

Usually, we have word-based embeddings, but for the task we need to create a representation
for the whole question. It could be done in different ways.


Method 1 - Word2Vec Embeddings
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The first method will use a mean of all word vectors in the question.

Note that there could be words without the corresponding embeddings. In this case, we'll just
skip these words and not take them into account in calculating the result. If the question
doesn't contain any known words with embedding, the function will return a zero vector.


Method 2 - StarSpace Embeddings
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the second method, we train our own word embeddings for the task of duplicates detection.
The `StarSpace model`_ can be trained specifically for some tasks. In contrast to the word2vec
model, which tries to train similar embeddings for words in similar contexts, StarSpace uses
embeddings for the whole sentence (just as a sum of embeddings of words and phrases). Despite
the fact that in both cases we get word embeddings as a result of the training, StarSpace
embeddings are trained using some supervised data, e.g. a set of similar sentence pairs, and
thus they can better suit the task.

In our case, StarSpace should use two types of sentence pairs for training: "positive" and
"negative". "Positive" examples are extracted from the train sample (duplicates, high
similarity) and the "negative" examples are generated randomly (low similarity assumed).


Evaluation of text similarity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We can imagine that if we use good embeddings, the cosine similarity between the duplicate
sentences should be less than for the random ones. Overall, for each pair of duplicate sentences
we can generate R random negative examples and find out the position of the correct duplicate.

For example, we have the question "Exceptions What really happens" and we are sure that another
question "How does the catch keyword determine the type of exception that was thrown" is a
duplicate. But our model doesn't know it and tries to find the best option also among questions
like "How Can I Make These Links Rotate in PHP", "NSLog array description not memory address"
and "PECL_HTTP not recognised php ubuntu". The goal of the model is to rank all these 4 questions
(1 positive and R = 3 negative) in the way that the correct one is in the first place.

However, it is unnatural to count on that the best candidate will be always in the first place.
So let us consider the place of the best candidate in the sorted list of candidates and formulate
a metric based on it. We can fix some K — a reasonable number of top-ranked elements and N — a
number of queries (size of the sample).

The first simple metric will be a number of correct hits for some K - `hits_count`.

The second one is a simplified `discounted cumulative gain (DCG) metric`_ - `dcg_score`.
According to this metric, the model gets a higher reward for a higher position of the correct
answer. If the answer does not appear in top K at all, the reward is zero.


.. _`StarSpace model`: https://arxiv.org/pdf/1709.03856.pdf
.. _`discounted cumulative gain (DCG) metric`: https://en.wikipedia.org/wiki/Discounted_cumulative_gain