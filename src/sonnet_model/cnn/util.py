from common.util import random_minibatches
import matplotlib.pyplot as plt
import numpy as np
from sonnet_model.cnn.model_setup import model_builder
import tensorflow as tf
from tensorflow.python.framework import ops


def plot_costs(costs, learning_rate):
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title('Learning rate =' + str(learning_rate))
    plt.show()


def train(x_train, y_train, x_test, y_test, learning_rate=0.009,
          n_epochs=100, batch_size=64, seed=3, print_cost=True):
    (m, h, w, c) = x_train.shape
    n_classes = y_train.shape[1]
    costs = []

    # print('(m, h, w, c):', x_train.shape)
    print('learning_rate:', learning_rate)
    print('n_epochs:', n_epochs)

    ops.reset_default_graph()
    tf.set_random_seed(1)

    x_placeholder = tf.placeholder(tf.float32, [None, h, w, c])
    y_placeholder = tf.placeholder(tf.float32, [None, n_classes])

    cost_op, optimizer, z3 = model_builder(x_placeholder, y_placeholder, learning_rate)

    # initialize all variables globally
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(n_epochs):
            minibatch_cost = 0.
            n_minibatches = int(m / batch_size)
            seed = seed + 1
            minibatches = random_minibatches(x_train, y_train, batch_size, seed)
            for minibatch in minibatches:
                (minibatch_x, minibatch_y) = minibatch
                _, temp_cost = sess.run([optimizer, cost_op],
                                        feed_dict={x_placeholder: minibatch_x, y_placeholder: minibatch_y})
                minibatch_cost += temp_cost / n_minibatches

            if print_cost and epoch % 5 == 0:
                print('Cost after epoch %i: %f' % (epoch, minibatch_cost))

            costs.append(minibatch_cost)

        # Plot costs
        plot_costs(costs, learning_rate)

        # Calculate the correct predictions
        predict_op = tf.argmax(z3, axis=1)
        correct_prediction = tf.equal(predict_op, tf.argmax(y_placeholder, axis=1))

        # Calculate accuracy on test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
        train_accuracy = accuracy.eval({x_placeholder: x_train, y_placeholder: y_train})
        test_accuracy = accuracy.eval({x_placeholder: x_test, y_placeholder: y_test})

        print('Train accuracy:', train_accuracy)
        print('Test accuracy:', test_accuracy)

        return train_accuracy, test_accuracy
