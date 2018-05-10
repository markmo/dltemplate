Generate doodle images using a GAN
----------------------------------

One major advance in unsupervised learning has been the advent of generative adversarial networks (GANs),
introduced by Ian Goodfellow and his fellow researchers at the University of Montreal in 2014.

In GANs, we have two neural networks. One network—known as the "generator" — generates data based on a model
data distribution it has created using samples of real data it has received. The other network — known as
the "discriminator" — discriminates between the data created by the generator and data from the true data
distribution.

As a simple analogy, the generator is the counterfeiter, and the discriminator is the police trying to
identify the forgery. The two networks are locked in a zero-sum game. The generator is trying to fool
the discriminator into thinking the synthetic data comes from the true data distribution, and the
discriminator is trying to call out the synthetic data as fake.

GANs are unsupervised learning algorithms because the generator can learn the underlying structure of
the true data distribution even when there are no labels. It learns the underlying structure by using
a number of parameters significantly smaller than the amount of data it has trained on. This constraint
forces the generator to efficiently capture the most salient aspects of the true data distribution.

See `"An applied introduction to generative adversarial networks"`_.

Train an adversarial model on images from the `Quick, Draw! dataset`_.

The Quick Draw Dataset is a collection of 50 million drawings across `345 categories`_,
contributed by players of the game `Quick, Draw!`_. The drawings were captured as
timestamped vectors, tagged with metadata including what the player was asked to draw
and in which country the player was located.


.. _`"An applied introduction to generative adversarial networks"`:
    https://www.oreilly.com/ideas/an-applied-introduction-to-generative-adversarial-networks
.. _`Quick, Draw! dataset`: https://quickdraw.withgoogle.com/data
.. _`345 categories`: https://github.com/googlecreativelab/quickdraw-dataset/blob/master/categories.txt
.. _`Quick, Draw!`: https://quickdraw.withgoogle.com/