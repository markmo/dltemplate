from common.util import one_hot_encode
import csv
import datetime
import json
import numpy as np
import os
import pandas as pd
import re
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.contrib import learn
from text_classification_benchmarks.word_cnn.model_setup import TextCNN
import time


# noinspection SpellCheckingInspection
def load_data(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
        x = []
        y = []
        for item in data:
            x.append(item['question'])
            y.append(1)
            if len(item['nbestanswers']) > 0:
                x.append(item['nbestanswers'][0])
                y.append(0)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, shuffle=True)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, shuffle=True)
        train_df = pd.DataFrame({'x': x_train, 'y': y_train})
        val_df = pd.DataFrame({'x': x_val, 'y': y_val})
        test_df = pd.DataFrame({'x': x_test, 'y': y_test})
        classes = np.array([0, 1])
        print('Lengths Train: {}, Val: {}, Test: {}, Classes: {}'
              .format(len(train_df), len(val_df), len(test_df), len(classes)))
        return train_df, val_df, test_df, classes


def batch_iter(data, batch_size, n_epochs, shuffle=True):
    """ Generates a batch iterator for a dataset. """
    data = np.array(data)
    data_size = len(data)
    n_batches_per_epoch = int((data_size - 1) / batch_size) + 1
    for epoch in range(n_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data

        for batch_i in range(n_batches_per_epoch):
            start_index = batch_i * batch_size
            end_index = min((batch_i + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?'`]", " ", string)
    string = re.sub(r"'s", " 's", string)
    string = re.sub(r"'ve", " 've", string)
    string = re.sub(r"n't", " n't", string)
    string = re.sub(r"'re", " 're", string)
    string = re.sub(r"'d", " 'd", string)
    string = re.sub(r"'ll", " 'll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(df, n_classes):
    x = [clean_str(sent) for sent in df.x]
    y = one_hot_encode(df.y.values, n_classes)
    return x, y


def preprocess(train_df, val_df, n_classes):
    x_train, y_train = load_data_and_labels(train_df, n_classes)
    x_val, y_val = load_data_and_labels(val_df, n_classes)
    x_all = np.concatenate([x_train, x_val])

    # Build vocabulary
    max_length = max([len(x.split(' ')) for x in x_all])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_length)
    vocab_processor.fit(x_all)
    x_train = np.array(list(vocab_processor.transform(x_train)))
    x_val = np.array(list(vocab_processor.transform(x_val)))

    # Shuffle data
    np.random.seed(42)
    shuffle_indices_train = np.random.permutation(np.arange(len(y_train)))
    x_train_shuffled = x_train[shuffle_indices_train]
    y_train_shuffled = y_train[shuffle_indices_train]
    shuffle_indices_val = np.random.permutation(np.arange(len(y_val)))
    x_val_shuffled = x_val[shuffle_indices_val]
    y_val_shuffled = y_val[shuffle_indices_val]

    print('Vocab size: {:d}'.format(len(vocab_processor.vocabulary_)))

    return x_train_shuffled, y_train_shuffled, x_val_shuffled, y_val_shuffled, vocab_processor


def test(x_test, batch_size, checkpoint_dir, allow_soft_placement, log_device_placement, y_test=None):
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=allow_soft_placement,
            log_device_placement=log_device_placement
        )
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph('{}.meta'.format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_x = graph.get_operation_by_name('input_x').outputs[0]
            keep_prob = graph.get_operation_by_name('keep_prob').outputs[0]

            # Tensors we want to evaluate
            preds = graph.get_operation_by_name('output/predictions').outputs[0]

            # Generate batches for one epoch
            batches = batch_iter(list(x_test), batch_size=batch_size, n_epochs=1, shuffle=False)

            # Collect predictions
            all_preds = []
            for x_test_batch in batches:
                batch_preds = sess.run(preds, {input_x: x_test_batch, keep_prob: 1.0})
                all_preds = np.concatenate([all_preds, batch_preds])

    # Print accuracy if y_test is defined
    if y_test is not None:
        correct_preds = float(sum(all_preds == y_test))
        n_examples = len(y_test)
        print('Total number of test examples:', n_examples)
        print('Accuracy: {:g}'.format(correct_preds / float(n_examples)))

    return all_preds.astype(int)


def save_eval_to_csv(x_raw, preds, checkpoint_dir):
    # Save the evaluation to a CSV
    human_readable_preds = np.column_stack((np.array(x_raw), preds))
    out_path = os.path.join(checkpoint_dir, '..', 'predictions.csv')
    print('Saving evaluation to', out_path)
    with open(out_path, 'w') as f:
        csv.writer(f).writerows(human_readable_preds)


def train(x_train, y_train, x_val, y_val, vocab_processor, model, learning_rate,
          n_checkpoints, keep_prob, batch_size, n_epochs, evaluate_every, checkpoint_every,
          allow_soft_placement, log_device_placement, constants):
    vocab_size = len(vocab_processor.vocabulary_)
    embed_size = constants['embed_size']
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=allow_soft_placement,
            log_device_placement=log_device_placement
        )
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            model = TextCNN(seq_len=x_train.shape[1], n_classes=y_train.shape[1],
                            vocab_size=vocab_size,
                            embed_size=embed_size,
                            filter_sizes=constants['filter_sizes'],
                            n_filters=constants['n_filters'],
                            l2_reg_lambda=constants['l2_reg_lambda'])
            # Define training procedure
            global_step = tf.Variable(0, name='global_step', trainable=False)
            optimizer = tf.train.AdamOptimizer(learning_rate)
            grads_and_vars = optimizer.compute_gradients(model.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram('{}/grad/hist'.format(v.name), g)
                    sparsity_summary = tf.summary.scalar('{}/grad/sparsity'.format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)

            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            ts = str(int(time.time()))
            outdir = os.path.abspath(os.path.join(os.path.curdir, 'runs', ts))
            print('Writing to {}\n'.format(outdir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar('loss', model.loss)
            acc_summary = tf.summary.scalar('accuracy', model.accuracy)

            # Train summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(outdir, 'summaries', 'train')
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Validate summaries
            val_summary_op = tf.summary.merge([loss_summary, acc_summary])
            val_summary_dir = os.path.join(outdir, 'summaries', 'val')
            val_summary_writer = tf.summary.FileWriter(val_summary_dir, sess.graph)

            # Checkpoint directory. TensorFlow assumes this directory already exists,
            # so we need to first create it.
            checkpoint_dir = os.path.abspath(os.path.join(outdir, 'checkpoints'))
            checkpoint_prefix = os.path.join(checkpoint_dir, 'model')
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)

            saver = tf.train.Saver(tf.global_variables(), max_to_keep=n_checkpoints)

            # Write vocabulary
            vocab_processor.save(os.path.join(outdir, 'vocab'))

            # Initialise all variables
            sess.run(tf.global_variables_initializer())

            word2vec_filename = constants.get('word2vec_filename', None)
            if word2vec_filename:
                print('Loading word2vec embeddings...')
                init_w = np.random.uniform(-0.25, 0.25, (vocab_size, embed_size))
                print('Loading word2vec file', word2vec_filename)
                with open(word2vec_filename, 'rb') as f:
                    header = f.readline()
                    vocab_size, layer1_size = map(int, header.split())
                    bin_length = np.dtype('float32').itemsize * layer1_size
                    for line in range(vocab_size):
                        word = []
                        while True:
                            ch = f.read(1).decode('latin-1')
                            if ch == ' ':
                                word = ''.join(word)
                                break

                            if ch != '\n':
                                word.append(ch)

                        idx = vocab_processor.vocabulary_.get(word)
                        if idx != 0:
                            # noinspection PyTypeChecker
                            init_w[idx] = np.fromstring(f.read(bin_length), dtype='float32')
                        else:
                            f.read(bin_length)

                sess.run(model.w.assign(init_w))

            def train_step(x_batch_, y_batch_):
                """ A single training step. """
                feed_dict = {
                    model.input_x: x_batch_,
                    model.input_y: y_batch_,
                    model.keep_prob: keep_prob
                }
                _, step, summaries, loss, accuracy = sess.run([train_op, global_step, train_summary_op,
                                                               model.loss, model.accuracy], feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print('{}: step {}, loss {:g}, acc {:g}'.format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)

            def val_step(x_batch_, y_batch_, writer=None):
                """ Evaluates model on a val set. """
                feed_dict = {
                    model.input_x: x_batch_,
                    model.input_y: y_batch_,
                    model.keep_prob: 1.0
                }
                step, summaries, loss, accuracy = sess.run([global_step, val_summary_op,
                                                            model.loss, model.accuracy], feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print('{}: step {}, loss {:g}, acc {:g}'.format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)

            # Generate batches
            batches = batch_iter(list(zip(x_train, y_train)), batch_size, n_epochs)

            # Training loop:
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % evaluate_every == 0:
                    print('\nEvaluation:')
                    val_step(x_val, y_val, writer=val_summary_writer)
                    print('')

                if current_step % checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print('Saved model checkpoint to {}\n'.format(path))
