import tensorflow as tf
from tf_model.simple_cnn.hyperparams import get_constants


def network_builder(x_dict, constants, scope, reuse, is_training):
    with tf.variable_scope(scope, reuse=reuse):
        x = x_dict['X']
        height, width, channels = constants['img_shape']
        filter1, filter2 = constants['filters']
        kernel1, kernel2 = constants['kernel_sizes']
        input_layer = tf.reshape(x, shape=[-1, height, width, channels])
        conv1 = tf.layers.conv2d(input_layer, filter1, kernel1, activation=tf.nn.relu)
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)
        conv2 = tf.layers.conv2d(conv1, filter2, kernel2, activation=tf.nn.relu)
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)
        fc1 = tf.contrib.layers.flatten(conv2)
        fc1 = tf.layers.dense(fc1, constants['n_hidden'])
        fc1 = tf.layers.dropout(fc1, rate=constants['keep_prob'], training=is_training)
        out = tf.layers.dense(fc1, constants['n_classes'])
        return out


def model_builder(features, labels, mode):
    constants = get_constants()

    logits_train = network_builder(features, constants, 'Model', reuse=False, is_training=True)
    logits_test = network_builder(features, constants, 'Model', reuse=True, is_training=False)

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
