Boilerplate for Deep Learning Projects
======================================

Model Templates
---------------

1. `Multi-layer Perceptron <src/homemade/__init__.py>`_ - MNIST (Homemade framework)
2. `Logistic Regression <src/tf_model/logreg/__init__.py>`_ - MNIST (TensorFlow)
3. `Simple Multi-layer Perceptron <src/tf_model/simple/__init__.py>`_ - MNIST (TensorFlow)
4. `Enhanced Multi-layer Perceptron using Batch Normalization <src/tf_model/enhanced/__init__.py>`_ - MNIST (TensorFlow)
5. `Enhanced Multi-layer Perceptron using TensorFlow Estimator API <src/tf_model/with_estimator/__init__.py>`_ for training - MNIST
6. `Simple CNN <src/tf_model/simple_cnn/__init__.py>`_ - MNIST (TensorFlow)
7. `Enhanced CNN <src/keras_model/cnn/__init__.py>`_ - Image Classifier (Keras)
8. `Image classifier <src/keras_model/image_classifier/__init__.py>`_ (Keras)
9. `Autoencoder <src/keras_model/autoencoder/__init__.py>`_ - Denoising images, Facial Recognition, Face Generation (Keras)
10. `RNN <src/keras_model/rnn/__init__.py>`_ - Name Generator (Keras)
11. `Part of speech (POS) tagging <src/keras_model/pos_tagger/__init__.py>`_ using an RNN (Keras)
12. `Image Captioning <src/keras_model/image_captioning/__init__.py>`_ (Keras)
13. `Image Classifier using Fast.ai and resnet <src/pytorch_model/cnn/__init__.py>`_ (PyTorch)

Demonstrates
^^^^^^^^^^^^

1. Basic principles of a neural net framework with methods for forward and backward steps
2. Basics of TensorFlow
3. Basic setup for a deep network
4. More complex network using batch normalization
5. Training with the TensorFlow Estimator API
6. Basic principles of a convolutional neural network
7. CNN using Keras
8. Fine-tuning InceptionV3 for image classification
9. Autoencoders
10. Basic principles of a recurrent neural network for character-level text generation
11. Using an RNN for POS tagging, using the high-level Keras API for building an RNN,
    creating a bidirectional RNN
12. Combining a CNN (encoder) and RNN (decoder) to caption images
13. A higher level framework (3 lines of code for an image classifier)


Datasets
--------

1. MNIST - handwritten digits (Keras)
2. CIFAR-10 - labelled images with 10 classes
3. `Flowers classification dataset`_
4. LFW (Labeled Faces in the Wild) - photographs of faces from the web
5. Names - list of human names
6. Captioned Images
7. Tagged sentences from the NLTK Brown Corpus


Naming conventions
------------------

Hyperparameters
^^^^^^^^^^^^^^^

* n_epochs
* learning_rate, lr
* epsilon


Parameters
^^^^^^^^^^

* features, inp, x, x_train, x_val, x_test
* labels, y, y_train, y_val, y_test
* weights, w, w1, w2, w3
* bias, b, b1, b2, b3
* z, z1, z2, z3
* a, a1, a2, a3


Common tests
------------

1. Check gradients against a calculated finite-difference approximation
2. Check shapes
3. Logits range. If your model has a specific output range rather than linear, you can test
   to make sure that the range stays consistent. For example, if logits has a tanh output,
   all of our values should fall between 0 and 1.
4. Input dependencies. Makes sure all of the variables in feed_dict affect the train_op.
5. Variable change. Check variables you expect to train with each training op.

Good practices for tests:

1. Keep them deterministic. If you really want randomized input, make sure to seed the
   random number so you can rerun the test easily.
2. Keep the tests short. Don’t have a unit test that trains to convergence and checks
   against a validation set. You are wasting your own time if you do this.
3. Make sure you reset the graph between each test.


Useful references
^^^^^^^^^^^^^^^^^

1. `How to test gradient implementations`_

.. _`Flowers classification dataset`: http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html
.. _`How to test gradient implementations`: https://timvieira.github.io/blog/post/2017/04/21/how-to-test-gradient-implementations/
