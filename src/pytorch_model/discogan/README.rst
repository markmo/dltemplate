DiscoGAN
--------

Because the authors call their method DiscoGAN!

.. image:: ../../../images/discogan.png

PyTorch implementation of `Learning to Discover Cross-Domain Relations with Generative Adversarial Networks <https://arxiv.org/abs/1703.05192>`_.

.. image:: ../../../images/discogan_model.png

Generates an image in one domain given another image in another domain, without explicitly
paired training data.

Cross-domain relations are often natural to humans. For example, we recognize the relationship
between an English sentence and its translated sentence in French. We also choose a suit jacket
with pants or shoes in the same style to wear. Can machines also achieve a similar ability to
relate two different image domains?

This GAN-based model trains with two independently collected sets of images and learns how to
map two domains without any extra label. For example, the model takes a handbag image as an
input, and generates its corresponding shoe image in the same style.

The core of the model is based on two different GANs coupled together – each of them ensures
our generative functions can map each domain to its counterpart domain. A key intuition is to
constraint all images in one domain to be representable by images in the other domain. For
example, when learning to generate a shoe image based on each handbag image, we force this
generated image to be an image-based representation of the handbag image (and hence reconstruct
the handbag image) through a reconstruction loss, and to be as close to images in the shoe
domain as possible through a GAN loss.