Boilerplate for Deep Learning Projects
======================================

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

1. how-to-test-gradient-implementations_

.. _how-to-test-gradient-implementations: https://timvieira.github.io/blog/post/2017/04/21/how-to-test-gradient-implementations/
