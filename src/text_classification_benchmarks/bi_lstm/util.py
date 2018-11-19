from datetime import datetime
import numpy as np
import os
import re
import tensorflow as tf
from tensorflow.contrib import learn
from text_classification_benchmarks.bi_lstm.model_setup import BiLSTM
from text_classification_benchmarks.data_loader import clean_data, load_data, remove_classes_with_too_few_examples


def batch_iter(data, labels, lengths, batch_size, n_epochs):
    """
    Generates mini-batches for training.

    :param data: list of sentences. Each sentence is a vector of integers.
    :param labels: list of labels
    :param lengths:
    :param batch_size: size of mini-batch
    :param n_epochs: number of epochs
    :return: a mini-batch iterator
    """
    assert len(data) == len(labels) == len(lengths)
    data_size = len(data)
    epoch_len = data_size // batch_size
    end_index = 0
    for _ in range(n_epochs):
        for i in range(epoch_len):
            start_index = i * batch_size
            end_index = start_index + batch_size
            x = data[start_index:end_index]
            y = labels[start_index:end_index]
            seq_len = lengths[start_index:end_index]
            yield x, y, seq_len

    if end_index < data_size:
        start_index = end_index
        end_index = data_size
        x = data[start_index:end_index]
        y = labels[start_index:end_index]
        seq_len = lengths[start_index:end_index]
        yield x, y, seq_len


def load_dataset(outdir, min_frequency=0, dirname='.'):
    train_df, val_df, test_df, classes = load_data(dirname=dirname)
    train_df = remove_classes_with_too_few_examples(clean_data(train_df))
    val_df = remove_classes_with_too_few_examples(clean_data(val_df))
    train_labels, train_utterances, train_lengths = prepare_data(train_df)
    val_labels, val_utterances, val_lengths = prepare_data(val_df)
    (train_data, train_labels, train_lengths,
     val_data, val_labels, val_lengths,
     max_length, vocab_size, vocab_processor) = \
        process_data(train_labels, train_utterances, train_lengths,
                     val_labels, val_utterances, val_lengths, min_frequency)
    vocab_processor.save(os.path.join(outdir, 'vocab'))
    return (train_data, train_labels, train_lengths,
            val_data, val_labels, val_lengths,
            max_length, vocab_size, classes)


def prepare_data(df):
    labels = df.label.values
    utterances = df.utterance.apply(lambda x: _clean_data(x.lower())).values
    lengths = np.array(list(map(len, [x.strip().split(' ') for x in utterances])))
    return labels, utterances, lengths


def process_data(train_labels, train_utterances, train_lengths, val_labels, val_utterances, val_lengths, min_frequency):
    all_utterances = np.concatenate([train_utterances, val_utterances])
    max_length = max(np.concatenate([train_lengths, val_lengths]))
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_length, min_frequency=min_frequency)
    vocab_processor.fit(all_utterances)
    train_data = np.array(list(vocab_processor.transform(train_utterances)))
    val_data = np.array(list(vocab_processor.transform(val_utterances)))
    max_length = vocab_processor.max_document_length
    vocab_size = len(vocab_processor.vocabulary_)
    return (train_data, train_labels, train_lengths,
            val_data, val_labels, val_lengths,
            max_length, vocab_size, vocab_processor)


# noinspection PyUnusedLocal
def train(train_data, x_val, y_val, val_lengths, n_classes, vocab_size, n_hidden, n_layers,
          l2_reg_lambda, learning_rate, decay_steps, decay_rate, keep_prob, outdir, num_checkpoint,
          evaluate_every_steps, save_every_steps):
    with tf.Graph().as_default():
        with tf.Session() as sess:
            model = BiLSTM(n_classes, vocab_size, n_hidden, n_layers, l2_reg_lambda)
            global_step = tf.Variable(0, name='global_step', trainable=False)
            learning_rate = tf.train.exponential_decay(learning_rate, global_step,
                                                       decay_steps, decay_rate, staircase=True)
            optimizer = tf.train.AdamOptimizer(learning_rate)
            grads_and_vars = optimizer.compute_gradients(model.cost)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            loss_summary = tf.summary.scalar('Loss', model.cost)
            accuracy_summary = tf.summary.scalar('Accuracy', model.accuracy)

            train_summary_op = tf.summary.merge_all()
            train_summary_dir = os.path.join(outdir, 'summaries_bal', 'train')
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            val_summary_op = tf.summary.merge_all()
            val_summary_dir = os.path.join(outdir, 'summaries_bal', 'valid')
            val_summary_writer = tf.summary.FileWriter(val_summary_dir, sess.graph)

            saver = tf.train.Saver(max_to_keep=num_checkpoint)

            sess.run(tf.global_variables_initializer())

            def run_step(input_data, is_training=True):
                """ Run one step of the training process. """
                input_x, input_y, seq_len = input_data
                fetches = {
                    'step': global_step,
                    'cost': model.cost,
                    'accuracy': model.accuracy,
                    'learning_rate': learning_rate,
                    'final_state': model.final_state
                }
                feed_dict = {
                    model.input_x: input_x,
                    model.input_y: input_y,
                    model.batch_size: len(input_x),
                    model.seq_len: seq_len
                }
                if is_training:
                    fetches['train_op'] = train_op
                    fetches['summaries'] = train_summary_op
                    feed_dict[model.keep_prob] = keep_prob
                else:
                    fetches['summaries'] = val_summary_op
                    feed_dict[model.keep_prob] = 1.0

                tvars = sess.run(fetches, feed_dict)
                step = tvars['step']
                cost = tvars['cost']
                accuracy = tvars['accuracy']
                summaries = tvars['summaries']

                # Write summaries to file
                if is_training:
                    train_summary_writer.add_summary(summaries, step)
                else:
                    val_summary_writer.add_summary(summaries, step)

                time_str = datetime.now().isoformat()
                print('{}: step: {}, loss: {:g}, accuracy: {:g}'.format(time_str, step, cost, accuracy))

                return accuracy

            print('Start training...')
            for train_input in train_data:
                run_step(train_input, is_training=True)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % evaluate_every_steps == 0:
                    print('\nValidation')
                    run_step((x_val, y_val, val_lengths), is_training=False)
                    print('')

                if current_step % save_every_steps == 0:
                    saver.save(sess, os.path.join(outdir, 'model_bal/clf'), current_step)

            print('\nAll files have been saved to {}\n'.format(outdir))


def _clean_data(text):
    """ Remove special characters """
    text = re.sub(r"[^A-Za-z0-9(),!?\'`]", " ", text)
    text = re.sub(r"\'s", " \'s", text)
    text = re.sub(r"\'ve", " \'ve", text)
    text = re.sub(r"n\'t", " n\'t", text)
    text = re.sub(r"\'re", " \'re", text)
    text = re.sub(r"\'d", " \'d", text)
    text = re.sub(r"\'ll", " \'ll", text)
    text = re.sub(r",", " , ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\(", " \( ", text)
    text = re.sub(r"\)", " \) ", text)
    text = re.sub(r"\?", " \? ", text)
    text = re.sub(r"\s{2,}", " ", text)

    return text


def test(data, labels, lengths, batch_size, run_dir, checkpoint):
    # Restore graph
    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session()

        # Restore metagraph
        saver = tf.train.import_meta_graph('{}.meta'.format(os.path.join(run_dir, 'model_bal', checkpoint)))

        # Restore weights
        saver.restore(sess, os.path.join(run_dir, 'model_bal', checkpoint))

        # Get tensors
        input_x = graph.get_tensor_by_name('input_x:0')
        input_y = graph.get_tensor_by_name('input_y:0')
        keep_prob = graph.get_tensor_by_name('keep_prob:0')
        preds = graph.get_tensor_by_name('softmax/preds:0')
        accuracy = graph.get_tensor_by_name('accuracy/accuracy:0')

        # Generate batches
        batches = batch_iter(data, labels, lengths, batch_size, 1)
        n_batches = int(len(data) / batch_size)
        all_preds = []
        sum_accuracy = 0

        for batch in batches:
            x_test, y_test, x_lengths = batch
            batch_sz = graph.get_tensor_by_name('batch_size:0')
            seq_len = graph.get_tensor_by_name('seq_len:0')
            feed_dict = {input_x: x_test, input_y: y_test, batch_sz: len(x_test), seq_len: x_lengths, keep_prob: 1.0}
            batch_preds, batch_accuracy = sess.run([preds, accuracy], feed_dict)

            sum_accuracy += batch_accuracy
            all_preds = np.concatenate([all_preds, batch_preds])

        final_accuracy = (sum_accuracy / n_batches) if n_batches > 0 else 0

    print('Test accuracy:', final_accuracy)

    return all_preds.astype(int)
