NB-SVM - SVM with NB features
-----------------------------

A simple but novel SVM variant using NB log-count ratios as feature values.

By combining generative and discriminative classifiers, we present a simple model variant where an SVM
is built over NB log-count ratios as feature values, and show that it is a strong and robust performer
over all the presented tasks.

Otherwise identical to the SVM, except we use ``x(k) = ˜f^(k)``, where ``˜f(k) = ˆr ◦ ˆf(k)`` is the
elementwise product. While this does very well for long documents, we find that an interpolation between
MNB and SVM performs excellently for all documents and we report results using this model:

::

    w' = (1 − β)w¯ + βw

where ``w¯ = ||w||1 / |V|`` is the mean magnitude of ``w``, and ``β ∈ [0, 1]`` is the interpolation
parameter. This interpolation can be seen as a form of regularization: trust NB unless the SVM is
very confident.

While (Ng and Jordan, 2002) showed that NB is better than SVM/logistic regression (LR) with few training
cases, we show that MNB is also better with short documents. In contrast to their result that an SVM
usually beats NB when it has more than 30–50 training cases, we show that MNB is still better on snippets
even with relatively large training sets (9k cases).

Based on the paper from Sida Wang and Christopher D. Manning: `Baselines and Bigrams: Simple, Good Sentiment
and Topic Classification; ACL 2012 <http://nlp.stanford.edu/pubs/sidaw12_simple_sentiment.pdf>`_.
