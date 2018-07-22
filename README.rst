Boilerplate for Deep Learning Projects
======================================

Model Templates
---------------

1. `Multi-layer Perceptron <src/homemade/>`_ - MNIST (Homemade framework)
2. `CNN from scratch <src/cnn_from_scratch/>`_ (Homemade framework)
3. `Logistic Regression <src/tf_model/logreg/>`_ - MNIST (TensorFlow)
4. `Simple Multi-layer Perceptron <src/tf_model/simple/>`_ - MNIST (TensorFlow)
5. `Enhanced Multi-layer Perceptron using Batch Normalization <src/tf_model/enhanced/>`_ - MNIST (TensorFlow)
6. `Enhanced Multi-layer Perceptron using TensorFlow Estimator API <src/tf_model/with_estimator/>`_ - MNIST
7. `Simple CNN <src/tf_model/simple_cnn/>`_ - MNIST (TensorFlow)
8. `Enhanced CNN <src/keras_model/cnn/>`_ - Image Classifier (Keras)
9. `Image classifier <src/keras_model/image_classifier/>`_ (Keras)
10. `Autoencoder <src/keras_model/autoencoder/>`_ - Denoising images, Facial Recognition, Face Generation (Keras)
11. `RNN <src/keras_model/rnn/>`_ - Name Generator (Keras)
12. `Part of speech (POS) tagging <src/keras_model/pos_tagger/>`_ using an RNN (Keras)
13. `Image Captioning <src/keras_model/image_captioning/>`_ (Keras)
14. `Image Classifier using ResNet and Fast.ai <src/pytorch_model/cnn/>`_ (PyTorch)
15. `Deep Q Network <src/keras_model/dqn/>`_ (Keras)
16. `Generative Adversarial Network (GAN) <src/keras_model/gan/>`_ (Keras)
17. `Predicting StackOverflow Tags using Classical NLP <src/nlp/multilabel_classification/>`_
18. `CNN using Sonnet <src/sonnet_model/cnn>`_ - Signs dataset
19. `Recognize named entities on Twitter using a Bidirectional LSTM <src/tf_model/ner/>`_ (TensorFlow)
20. `Recognize named entities on Twitter using CRF <src/nlp/crf_ner/>`_ (sklearn-crfsuite)
21. `Recognize named entities on Twitter using Bi-LSTM + CRF <src/tf_model/bi_lstm_crf_ner/>`_ (TensorFlow)
22. `Detect Duplicate Questions on StackOverflow using Embeddings <src/nlp/duplicate_questions/>`_
23. `Building a Simple Calculator using a Sequence-to-Sequence Model <src/tf_model/seq2seq/>`_ (TensorFlow)
24. `Reinforcement Learning using crossentropy method <src/rl/crossentropy/>`_
25. `Reinforcement Learning using a neural net <src/rl/crossentropy/>`_ (sklearn)
26. `Navigate a Frozen Lake using a Markov Decision Process (MDP) <src/rl/frozen_lake/>`_
27. `A Sequence-to-Sequence Chatbot <src/tf_model/chatbot1/>`_ (TensorFlow)
28. `Solve the Taxi Challenge using Q-Learning <src/rl/qlearning/>`_
29. `Training a Deep Q-Learning Network to play Atari Breakout <src/rl/dqn_breakout/>`_ (Keras)
30. `Playing CartPole using REINFORCE <src/rl/reinforce_cartpole/>`_ (Keras)
31. `Playing Kung Fu Master using Advantage Actor Critic (AAC) <src/rl/aac_kungfumaster/>`_ (Keras)
32. `Playing CartPole using Monte Carlo Tree Search <src/rl/mcts_planning/>`_
33. `Translating Hebrew to English using RL for Seq2Seq Models <src/rl/rl_for_seq2seq/>`_ (TensorFlow)
34. `Bernoulli Bandits - Survey of Model-free RL Algorithms <src/rl/bernoulli_bandits/>`_
35. `Q-Table Learning Agent <src/rl/survey_of_methods/q_table_learning/>`_
36. `Multi-armed Bandit <src/rl/survey_of_methods/multi_armed_bandit/>`_ (TensorFlow)
37. `Contextual Bandits <src/rl/survey_of_methods/contextual_bandits/>`_ (TensorFlow)
38. `Vanilla Policy Gradient Agent <src/rl/survey_of_methods/policy_gradient/>`_ (TensorFlow)
39. `Model-based example for RL <src/rl/survey_of_methods/model_based/>`_ (TensorFlow)
40. `Deep Q-Network <src/rl/survey_of_methods/dqn/>`_ (TensorFlow)
41. `Wake-word Detection <src/keras_model/wake_word_detection/>`_ (Keras)
42. `Neural Turing Machine <src/tf_model/neural_turing_machine/>`_ (TensorFlow)

`Reinforcement Learning - Survey of Methods <src/rl/survey_of_methods/>`_.


Demonstrates
^^^^^^^^^^^^

1. Basic principles of a neural net framework with methods for forward and backward steps
2. Basic principles of convolutional neural network
3. Basics of TensorFlow
4. Basic setup for a deep network
5. More complex network using batch normalization
6. Training with the TensorFlow Estimator API
7. Basic principles of a convolutional neural network
8. CNN using Keras
9. Fine-tuning InceptionV3 for image classification
10. Autoencoders
11. Basic principles of a recurrent neural network for character-level text generation
12. Using an RNN for POS tagging, using the high-level Keras API for building an RNN,
    creating a bidirectional RNN
13. Combining a CNN (encoder) and RNN (decoder) to caption images
14. A higher level framework (3 lines of code for an image classifier)
15. Deep Reinforcement Learning using CartPole environment in the OpenAI Gym
16. Basic principles of a GAN to generate doodle images trained on the 'Quick, Draw!' dataset.
17. Exploring classical NLP techniques for multi-label classification.
18. Basic usage of Sonnet to organize a TensorFlow model
19. Basic principles of a Bidirectional LSTM for named entity recognition
20. Basic principles of Conditional Random Fields (CRF) and comparison with Bi-LSTM on the same task
21. Combining a Bi-LSTM with CRF to get learned features + constraints
22. Use of embeddings at a sentence level, testing `StarSpace`_ from Facebook Research.
23. Solving sequence-to-sequence prediction tasks.
24. Basic principles of reinforcement learning
25. Approximating crossentropy with neural nets in an RL model
26. Using a Markov Decision Process to solve an RL problem.
27. Building a chatbot using a sequence-to-sequence model approach.
28. Basic principles of Q-Learning
29. Tips and tricks to train a Deep Q-Learning Network - Frame Buffer, Experience Replay
30. Basic principles of using the REINFORCE algorithm
31. Basic principles of using the Advantage Actor Critic (AAC) algorithm
32. Introduction to Planning Algorithms using Monte Carlo Tree Search.
33. Reinforcement learning for sequence-to-sequence models.
34. Survey of Model-free RL algorithms - Epsilon-greedy, UCB1, and Thompson Sampling.
35. Introduction to Q-Table Learning.
36. Building a simple policy-gradient based agent that can solve the multi-armed bandit problem.
37. Building a simple policy-gradient based agent where the environment has state, but state is
    not determined by the previous state or action.
38. Introduction to Policy Gradient methods in RL.
39. Introduction to model-based RL networks.
40. Implement a Deep Q-Network using Experience Replay.
41. Processing audio using an RNN to detect wake-words.
42. Introduction to Neural Turing Machines.


Datasets
--------

1. MNIST - handwritten digits (Keras)
2. CIFAR-10 - labelled images with 10 classes
3. `Flowers classification dataset`_
4. LFW (Labeled Faces in the Wild) - photographs of faces from the web
5. Names - list of human names
6. Captioned Images
7. Tagged sentences from the NLTK Brown Corpus
8. `Quick, Draw! dataset`_
9. StackOverflow posts and corresponding tags
10. Sign language - numbers 0 - 5
11. Tweets tagged with named entities
12. Duplicate questions set, with positive and negative examples, from StackOverflow
13. Cornell movie dialog corpus.
14. Open Subtitles movie dialog corpus.
15. Hebrew to English words.


Notation
--------

* Superscript :math:`[l]` denotes an object of the :math:`l^{th}` layer.
    * Example: :math:`a^{[4]}` is the :math:`4^{th}` layer activation. :math:`W^{[5]}` and :math:`b^{[5]}`
      are the :math:`5^{th}` layer parameters.
* Superscript :math:`(i)` denotes an object from the :math:`i^{th}` example.
    * Example: :math:`x^{(i)}` is the :math:`i^{th}` training example input.
* Subscript :math:`i` denotes the :math:`i^{th}` entry of a vector.
    * Example: :math:`a^{[l]}_i` denotes the :math:`i^{th}` entry of the activations in layer :math:`l`,
      assuming this is a fully connected (FC) layer.
* :math:`n_H`, :math:`n_W` and :math:`n_C` denote respectively the height, width and number of channels
  of a given layer. If you want to reference a specific layer :math:`l`, you can also write
  :math:`n_H^{[l]}`, :math:`n_W^{[l]}`, :math:`n_C^{[l]}`.
* :math:`n_{H_{prev}}`, :math:`n_{W_{prev}}` and :math:`n_{C_{prev}}` denote respectively the height,
  width and number of channels of the previous layer. If referencing a specific layer
  :math:`l`, this could also be denoted :math:`n_H^{[l-1]}`, :math:`n_W^{[l-1]}`, :math:`n_C^{[l-1]}`.


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
.. _`Quick, Draw! dataset`: https://github.com/googlecreativelab/quickdraw-dataset
.. _`How to test gradient implementations`: https://timvieira.github.io/blog/post/2017/04/21/how-to-test-gradient-implementations/


Ideas
-----

* Turn trainers into generators, one epoch at a time


.. _`StarSpace`: https://github.com/facebookresearch/StarSpace