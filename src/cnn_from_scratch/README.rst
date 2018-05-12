CNN from Scratch
----------------

Implementing the building blocks of a convolutional neural network using just numpy.

Convolution functions, including:

* Zero Padding
* Convolve window
* Convolution forward
* Convolution backward (optional)

Pooling functions, including:

* Pooling forward
* Create mask
* Distribute value
* Pooling backward (optional)

A convolution layer transforms an input volume into an output volume of different size, as shown below.

.. image:: ../../../images/conv_nn.png

Zero-padding adds zeros around the border of an image:

.. image:: ../../../images/padding.png

The main benefits of padding are:

* It allows you to use a convolutional layer without necessarily shrinking the height
  and width of the volumes. This is important for building deeper networks, since otherwise
  the height/width would shrink as you go to deeper layers. An important special case is
  the "same" convolution, in which the height/width is exactly preserved after one layer.
* It helps us keep more of the information at the border of an image. Without padding, very
  few values at the next layer would be affected by pixels as the edges of an image.
