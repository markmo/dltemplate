import tensorflow as tf


def get_parameters(constants):
    w1 = tf.get_variable('W1', [constants['n_input'], constants['n_hidden']],
                         initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable('b1', [constants['n_hidden']], initializer=tf.zeros_initializer())
    w2 = tf.get_variable('W2', [constants['n_hidden'], constants['n_classes']],
                         initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable('b2', [constants['n_classes']], initializer=tf.zeros_initializer())
    beta1 = tf.Variable(tf.ones([constants['n_hidden']]))
    scale1 = tf.Variable(tf.ones([constants['n_hidden']]))
    return {
        'W1': w1,
        'b1': b1,
        'W2': w2,
        'b2': b2,
        'beta1': beta1,
        'scale1': scale1
    }


def network_builder(input_x, params, constants):
    z1 = tf.matmul(input_x, params['W1']) + params['b1']
    batch_mean1, batch_var1 = tf.nn.moments(z1, [0])
    bn1 = tf.nn.batch_normalization(z1, batch_mean1, batch_var1,
                                    params['beta1'], params['scale1'],
                                    constants['epsilon'])
    a1 = tf.nn.relu(bn1)
    z2 = tf.matmul(a1, params['W2']) + params['b2']
    return z2


def model_builder(network, input_x, input_y, params, constants):
    model = network(input_x, params, constants)
    y_ = tf.nn.softmax(model)
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=model, labels=input_y
        ))
    optimizer = tf.train.AdamOptimizer(constants['learning_rate']).minimize(loss_op)
    return optimizer, loss_op, model, y_
