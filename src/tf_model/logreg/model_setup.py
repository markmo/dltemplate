import tensorflow as tf


def get_parameters(constants):
    w = tf.Variable(tf.zeros([constants['n_input'], constants['n_classes']]), name='W')
    b = tf.Variable(tf.zeros([constants['n_classes']]), name='b')
    return {
        'W': w,
        'b': b
    }


def model_builder(input_x, input_y, params, constants):
    model = tf.matmul(input_x, params['W']) + params['b']
    y_ = tf.nn.softmax(model)
    loss_op = tf.reduce_mean(input_y * -tf.log(y_) - (1 - input_y) * tf.log(1 - y_))
    optimizer = tf.train.GradientDescentOptimizer(constants['learning_rate']).minimize(loss_op)
    return optimizer, loss_op, model, y_
