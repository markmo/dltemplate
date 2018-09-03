Universal Language Model Fine-tuning for Text Classification
------------------------------------------------------------

Using transfer learning to improve performance on small labelled datasets given a pretrained
language model. This approach enables robust inductive transfer learning for any NLP task,
akin to fine-tuning ImageNet models.

In particular, the following techniques are used to retain previous knowledge and avoid
catastrophic forgetting during fine-tuning:

* **Discriminative fine-tuning**. As different layers capture different types of information
  (Yosinski et al., 2014), they should be fine-tuned to different extents. Instead of using the
  same learning rate for all layers of the model, discriminative fine-tuning allows us to tune
  each layer with different learning rates. We empirically found it to work well to first choose
  the learning rate η^L of the last layer by fine-tuning only the last layer and using
  η^(l−1) = η^l/2.6 as the learning rate for lower layers.
* **Slanted triangular learning rates**. For adapting its parameters to task-specific features,
  we would like the model to quickly converge to a suitable region of the parameter space in the
  beginning of training and then refine its parameters. Using the same learning rate (LR) or an
  annealed learning rate throughout training is not the best way to achieve this behaviour.
  The slanted triangular learning rates (STLR) technique first linearly increases the learning rate
  and then linearly decays it as a function of the number of training iterations. STLR modifies
  triangular learning rates (Smith, 2017) with a short increase and a long decay period, which we
  found key for good performance.
* Gradual unfreezing.

The method is universal in the sense that it meets these practical criteria:

1. It works across tasks varying in document size, number, and label type;
2. it uses a single architecture and training process;
3. it requires no custom feature engineering or preprocessing; and
4. it does not require additional in-domain documents or labels.

The Language Model (LM) in this implementation uses `AWD-LSTM`_ (Averaged SGD Weight-Dropped LSTM,
Merity et al., 2017a), and a regular LSTM (with no attention, short-cut connections, or other
sophisticated additions) with various tuned dropout hyperparameters. Analogous to computer vision,
we expect that downstream performance can be improved by using higher performance language models
in the future.

The general-domain LM is pretrained on Wikitext-103 (Merity et al., 2017b) consisting of 28,595
preprocessed Wikipedia articles and 103 million words. Pretraining is most beneficial for tasks
with small datasets and enables generalization even with 100 labeled examples.

Finally, for fine-tuning the classifier, we augment the pretrained language model with two additional
linear blocks. Following standard practice for CV classifiers, each block uses batch normalization
(Ioffe and Szegedy, 2015) and dropout, with ReLU activations for the intermediate layer and a softmax
activation that outputs a probability distribution over target classes at the last layer. Note that
the parameters in these task-specific classifier layers are the only ones that are learned from
scratch. The first linear layer takes as the input the pooled last hidden layer states.

Read `Universal Language Model Fine-tuning for Text Classification <https://arxiv.org/pdf/1801.06146.pdf>`_.

.. _`AWD-LSTM`: https://github.com/salesforce/awd-lstm-lm