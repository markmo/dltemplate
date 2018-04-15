import tensorflow as tf
from tf_model.with_estimator.hyperparams import get_constants


def get_parameters(constants):
    c = constants
    w1 = tf.get_variable('W1', [c['n_input'], c['n_hidden1']],
                         initializer=tf.contrib.layers.xavier_initializer()
                         )
    b1 = tf.get_variable('b1', [c['n_hidden1']], initializer=tf.zeros_initializer())
    w2 = tf.get_variable('W2', [c['n_hidden1'], c['n_hidden2']],
                         initializer=tf.contrib.layers.xavier_initializer()
                         )
    b2 = tf.get_variable('b2', [c['n_hidden2']], initializer=tf.zeros_initializer())
    w3 = tf.get_variable('W3', [c['n_hidden2'], c['n_classes']],
                         initializer=tf.contrib.layers.xavier_initializer()
                         )
    b3 = tf.get_variable('b3', [c['n_classes']], initializer=tf.zeros_initializer())
    beta2 = tf.Variable(tf.ones([c['n_hidden2']]))
    scale2 = tf.Variable(tf.ones([c['n_hidden2']]))
    return {
        'W1': w1,
        'b1': b1,
        'W2': w2,
        'b2': b2,
        'W3': w3,
        'b3': b3,
        'beta2': beta2,
        'scale2': scale2
    }


def network_builder(x_dict, params, constants, scope, reuse, is_training):
    with tf.variable_scope(scope, reuse=reuse):
        x = x_dict['X']

        # Example Dense / linear layer
        z1 = tf.matmul(x, params['W1']) + params['b1']

        # Example activation
        a1 = tf.nn.relu(z1)

        z2 = tf.matmul(a1, params['W2']) + params['b2']

        # Example batch normalization
        batch_mean2, batch_var2 = tf.nn.moments(z2, [0])
        bn2 = tf.nn.batch_normalization(z2, batch_mean2, batch_var2,
                                        params['beta2'], params['scale2'],
                                        constants['epsilon'])
        a2 = tf.nn.relu(bn2)

        z3 = tf.matmul(a2, params['W3']) + params['b3']
        return z3


def model_builder(features, labels, mode):
    constants = get_constants()
    params = get_parameters(constants)

    logits_train = network_builder(features, params, constants, 'Model', reuse=False, is_training=True)
    logits_test = network_builder(features, params, constants, 'Model', reuse=True, is_training=False)

    pred_classes = tf.argmax(logits_test, axis=1)
    # pred_probs = tf.nn.softmax(logits_test)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits_train, labels=tf.cast(labels, dtype=tf.int32)
        ))
    optimizer = tf.train.AdamOptimizer(learning_rate=constants['learning_rate'])
    train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())
    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)
    est_specs = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred_classes,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops={'accuracy': acc_op}
        )

    return est_specs
