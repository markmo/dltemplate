import tensorflow as tf


def get_parameters(constants):
    w1 = tf.Variable(tf.random_normal([constants['n_input'], constants['n_hidden']]), name='W1')
    b1 = tf.Variable(tf.random_normal([constants['n_hidden']]), name='b1')
    w2 = tf.Variable(tf.random_normal([constants['n_hidden'], constants['n_classes']]), name='W2')
    b2 = tf.Variable(tf.random_normal([constants['n_classes']]), name='b2')
    return {
        'W1': w1,
        'b1': b1,
        'W2': w2,
        'b2': b2,
    }


def network_builder(input_x, params):
    z1 = tf.matmul(input_x, params['W1']) + params['b1']
    a1 = tf.nn.sigmoid(z1)
    z2 = tf.matmul(a1, params['W2']) + params['b2']
    return z2


def model_builder(network, input_x, input_y, params, constants):
    model = network(input_x, params)
    y_ = tf.nn.softmax(model)
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=model, labels=input_y
        ))
    predict_op = tf.argmax(y_, 1)
    optimizer = tf.train.AdamOptimizer(constants['learning_rate']).minimize(loss_op)
    return optimizer, loss_op, predict_op, model, y_
