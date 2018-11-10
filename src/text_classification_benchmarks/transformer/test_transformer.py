import copy
import numpy as np
import os
import random
import tensorflow as tf
from text_classification_benchmarks.transformer.model_setup import Transformer


def test_training():
    """
    This is a unit test. If you use this for text classification, the input sentence
    must be transformed to vocabulary indices first.
    :return:
    """
    n_classes = 11  # additional two classes: '__GO__', '__END__'
    learning_rate = 0.00001
    batch_size = 1
    decay_steps = 1000
    decay_rate = 0.9
    seq_len = 6
    vocab_size = 300
    is_training = True
    keep_prob = 0.9
    decoder_sent_len = 6
    l2_lambda = 0.0001
    d_model = 512
    d_k = 64
    d_v = 64
    h = 8
    n_layers = 1
    embed_size = d_model
    tf.reset_default_graph()
    model = Transformer(n_classes, learning_rate, batch_size, decay_steps, decay_rate,
                        seq_len, vocab_size, embed_size, d_model, d_k, d_v, h, n_layers,
                        is_training, decoder_sent_len=decoder_sent_len, l2_lambda=l2_lambda)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        checkpoint_dir = './checkpoint_transformer/sequence_reverse/'
        if os.path.exists(checkpoint_dir + 'checkpoint'):
            saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))

        for i in range(150000):
            label_list = get_unique_labels()
            input_x = np.array([label_list + [9]], dtype=np.int32)
            label_list_original = copy.deepcopy(label_list)
            label_list.reverse()
            decoder_input = np.array([[0] + label_list], dtype=np.int32)
            input_y_label = np.array([label_list + [1]], dtype=np.int32)
            loss, acc, pred, w_projection, _ = \
                sess.run([model.loss_val, model.accuracy, model.predictions, model.w_projection, model.train_op],
                         {
                             model.input_x: input_x,
                             model.decoder_input: decoder_input,
                             model.input_y_label: input_y_label,
                             model.keep_prob: keep_prob
                         })
            print(i, 'loss:', loss, 'accuracy:', acc, 'input_x:', label_list_original,
                  'input_y_label:', input_y_label, 'prediction:', pred)
            if i % 1500 == 0:
                save_path = checkpoint_dir + 'model'
                saver.save(sess, save_path, global_step=i)


def test_predict():
    """
    This is a unit test. If you use this for text classification, the input sentence
    must be transformed to vocabulary indices first.
    :return:
    """
    n_classes = 11  # additional two classes: '__GO__', '__END__'
    learning_rate = 0.0001
    batch_size = 1
    decay_steps = 1000
    decay_rate = 0.9
    seq_len = 6
    vocab_size = 300
    is_training = False
    keep_prob = 1
    decoder_sent_len = 6
    l2_lambda = 0.0001
    d_model = 512
    d_k = 64
    d_v = 64
    h = 8
    n_layers = 1
    embed_size = d_model
    model = Transformer(n_classes, learning_rate, batch_size, decay_steps, decay_rate,
                        seq_len, vocab_size, embed_size, d_model, d_k, d_v, h, n_layers,
                        is_training, decoder_sent_len=decoder_sent_len, l2_lambda=l2_lambda)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        checkpoint_dir = './checkpoint_transformer/sequence_reverse/'
        if os.path.exists(checkpoint_dir + 'checkpoint'):
            saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))
            print('================= model restored')

        for i in range(150000):
            label_list = get_unique_labels()
            input_x = np.array([label_list + [9]], dtype=np.int32)
            label_list_original = copy.deepcopy(label_list)
            label_list.reverse()
            decoder_input = np.array([[0] * decoder_sent_len], dtype=np.int32)
            pred, w_projection = \
                sess.run([model.predictions, model.w_projection],
                         {
                             model.input_x: input_x,
                             model.decoder_input: decoder_input,
                             model.keep_prob: keep_prob
                         })
            print(i, 'input_x:', label_list_original, 'prediction:', pred)


def test_training_batch():
    """
    This is a unit test. If you use this for text classification, the input sentence
    must be transformed to vocabulary indices first.
    :return:
    """
    n_classes = 11  # additional two classes: '__GO__', '__END__'
    learning_rate = 0.001
    batch_size = 16
    decay_steps = 1000
    decay_rate = 0.9
    seq_len = 5
    vocab_size = 300
    is_training = True
    keep_prob = 1
    decoder_sent_len = 6
    l2_lambda = 0.0001
    d_model = 512
    d_k = 64
    d_v = 64
    h = 8
    n_layers = 1
    embed_size = d_model
    checkpoint_dir = './checkpoint_transformer/sequence_reverse/'
    model = Transformer(n_classes, learning_rate, batch_size, decay_steps, decay_rate,
                        seq_len + 1, vocab_size, embed_size, d_model, d_k, d_v, h, n_layers,
                        is_training, decoder_sent_len=decoder_sent_len, l2_lambda=l2_lambda)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(150000):
            label_list = get_unique_labels_batch(batch_size, length=seq_len)
            input_x = np.array([ll + [9] for ll in label_list], dtype=np.int32)
            label_list_original = copy.deepcopy(label_list)

            decoder_input_list = []
            input_y_label_list = []
            for _, sub_label_list in enumerate(label_list):
                sub_label_list.reverse()
                decoder_input_list.append([0] + sub_label_list)
                input_y_label_list.append(sub_label_list + [1])

            decoder_input = np.array(decoder_input_list, dtype=np.int32)
            input_y_label = np.array(input_y_label_list, dtype=np.int32)
            loss, acc, pred, w_projection, _ = \
                sess.run([model.loss_val, model.accuracy, model.predictions, model.w_projection, model.train_op],
                         {
                             model.input_x: input_x,
                             model.decoder_input: decoder_input,
                             model.input_y_label: input_y_label,
                             model.keep_prob: keep_prob
                         })
            print(i, 'loss:', loss, 'accuracy:', acc)
            if i % 100 == 0:
                print('input_x:', label_list_original, 'input_y_label:', input_y_label, 'prediction:', pred)

            if i % int(1500 / batch_size) == 0:
                save_path = checkpoint_dir + 'model'
                saver.save(sess, save_path, global_step=i * batch_size)


def get_unique_labels(length=5):
    x = [i for i in range(2, 2 + length)]
    random.shuffle(x)
    return x


def get_unique_labels_batch(batch_size, length=None):
    x = []
    for i in range(batch_size):
        labels = get_unique_labels(length)
        x.append(labels)

    return x
