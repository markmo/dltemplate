fastText performance in Multi-class Text Classification
-------------------------------------------------------

Recently, models based on neural networks have become increasingly popular (Kim, 2014;
Zhang and LeCun, 2015; Conneau et al., 2016). While these models achieve very good
performance in practice, they tend to be relatively slow both at train and test time,
limiting their use on very large datasets.

A simple and efficient baseline for sentence classification is to represent sentences
as bag of words (BoW) and train a linear classifier, e.g., a logistic regression or an
SVM (Joachims, 1998; Fan et al., 2008). However, linear classifiers do not share parameters
among features and classes. This possibly limits their generalization in the context of
large output space where some classes have very few examples. Common solutions to this
problem are to factorize the linear classifier into low rank matrices (Schutze, 1992;
Mikolov et al., 2013) or to use multilayer neural networks (Collobert and Weston, 2008;
Zhang et al., 2015).

Hierarchical Softmax

When the number of classes is large, computing the linear classifier is computationally
expensive. More precisely, the computational complexity is O(kh) where k is the number of
classes and h the dimension of the text representation. In order to improve our running
time, we use a hierarchical softmax (Goodman, 2001) based on the Huffman coding tree
(Mikolov et al., 2013). During training, the computational complexity drops to O(h log2(k)).

.. image:: ../../../images/fastText.jpg

Model architecture of a fastText for a sentence with `N` ngram features `x1, ...xn`. The
features are embedded and averaged to calculate the hidden variable.

This architecture is similar to the CBOW model of Mikolov et al. (2013), where the middle
word is replaced by a label. We use the softmax function f to compute the probability
distribution over the predefined classes. Cross entropy is used to compute loss. As bag-of-word
representation does not consider word order, n-gram features is used to capture some partial
information about the local word order.

See `Bag of Tricks for Efficient Text Classification <https://arxiv.org/abs/1607.01759>`_.