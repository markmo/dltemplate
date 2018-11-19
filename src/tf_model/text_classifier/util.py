from common.load_data import pad_sentences
import json
import logging
import numpy as np
import os
import pickle
import shutil
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tf_model.text_classifier.model_setup import TextModel
import time

logging.getLogger().setLevel(logging.INFO)


def batch_iter(data, batch_size, n_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    n_batches_per_epoch = int(data_size / batch_size) + 1
    for epoch in range(n_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data

        for batch_num in range(n_batches_per_epoch):
            start_idx = batch_num * batch_size
            end_idx = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_idx:end_idx]


def load_embeddings(vocab):
    embeddings = {}
    for word in vocab:
        embeddings[word] = np.random.uniform(-0.25, 0.25, 300)

    return embeddings


def load_trained_params(trained_dir):
    params = json.loads(open(trained_dir + 'trained_params.json').read())
    words_index = json.loads(open(trained_dir + 'words_index.json').read())
    labels = json.loads(open(trained_dir + 'labels.json').read())
    with open(trained_dir + 'embeddings.pkl', 'rb') as f:
        embeddings = pickle.load(f)

    embedding_mat = np.array(embeddings, dtype=np.float32)
    return params, words_index, labels, embedding_mat


def map_words_to_index(examples, words_index):
    x = []
    for ex in examples:
        temp = []
        for word in ex:
            if word in words_index:
                temp.append(words_index[word])
            else:
                temp.append(0)

        x.append(temp)

    return x


def predict(x, y, df, params, words_index, labels, embedding_mat, trained_dir):
    print('seq_len:', params['seq_len'])
    print('non_static:', params['non_static'])
    print('n_hidden:', params['n_hidden'])
    print('filter_sizes:', params['filter_sizes'])
    print('n_filters:', params['n_filters'])
    print('emb_dim:', params['emb_dim'])
    print('max_pool_size:', params['max_pool_size'])
    x = pad_sentences(x, forced_seq_len=params['seq_len'])
    x = map_words_to_index(x, words_index)
    x_test, y_test = np.asarray(x), None
    if y is not None:
        y_test = np.asarray(y)

    ts = trained_dir.split('/')[-2].split('_')[-1]
    pred_dir = 'predicted_results_{}/'.format(ts)
    if os.path.exists(pred_dir):
        shutil.rmtree(pred_dir)

    os.makedirs(pred_dir)
    emb_dim = params['emb_dim']
    max_pool_size = params['max_pool_size']
    with tf.Graph().as_default():
        sess_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session(config=sess_conf)
        with sess.as_default():
            model = TextModel(embedding_mat=embedding_mat,
                              non_static=params['non_static'],
                              n_hidden=params['n_hidden'],
                              seq_len=len(x_test[0]),
                              max_pool_size=max_pool_size,
                              filter_sizes=params['filter_sizes'],
                              n_filters=params['n_filters'],
                              n_classes=len(labels),
                              emb_dim=emb_dim,
                              l2_reg_lambda=params['l2_reg_lambda'])

            def real_len(batches_):
                return [np.ceil(np.argmin(batch + [0]) * 1.0 / max_pool_size) for batch in batches_]

            def predict_step(x_batch_):
                feed_dict = {
                    model.input_x: x_batch_,
                    model.keep_prob: 1.0,
                    model.batch_size: len(x_batch_),
                    model.pad: np.zeros([len(x_batch_), 1, emb_dim, 1]),
                    model.real_len: real_len(x_batch_)
                }
                preds_ = sess.run([model.preds], feed_dict)
                return preds_

            checkpoint_file = trained_dir + 'best_model'
            # noinspection PyUnusedLocal
            saver = tf.train.Saver(tf.global_variables())
            saver = tf.train.import_meta_graph(checkpoint_file + '.meta')
            saver.restore(sess, checkpoint_file)
            logging.critical('{} has been loaded'.format(checkpoint_file))
            batches = batch_iter(list(x_test), params['batch_size'], 1, shuffle=False)
            preds, pred_labels = [], []
            for x_batch in batches:
                batch_preds = predict_step(x_batch)[0]
                for batch_pred in batch_preds:
                    preds.append(batch_pred)
                    pred_labels.append(labels[batch_pred])

            # Save predictions back to file
            df['NEW_PREDICTED'] = pred_labels
            columns = sorted(df.columns, reverse=True)
            df.to_csv(pred_dir + 'predictions_all.csv', index=False, columns=columns, sep='|')
            if y_test is not None:
                y_test = np.array(np.argmax(y_test, axis=1))
                accuracy = sum(np.array(preds) == y_test) / float(len(y_test))
                logging.critical('Prediction accuracy: {}'.format(accuracy))

            logging.critical('Prediction complete! Saved to {}.'.format(pred_dir))

            return preds, pred_labels, df


def train(x, y, vocab, vocab_inv, labels, constants):
    embeddings = load_embeddings(vocab)
    embedding_mat = [embeddings[word] for _, word in enumerate(vocab_inv)]
    embedding_mat = np.array(embedding_mat, dtype=np.float32)

    # Split the original dataset into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

    # Split the train set into train and val sets
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)
    max_pool_size = constants['max_pool_size']
    emb_dim = constants['emb_dim']
    batch_size = constants['batch_size']
    n_epochs = constants['n_epochs']

    logging.info('x_train: {}, x_val: {}, x_test: {}'.format(len(x_train), len(x_val), len(x_test)))
    logging.info('y_train: {}, y_val: {}, y_test: {}'.format(len(y_train), len(y_val), len(y_test)))

    # Create a directory, everything related to the training will be saved in this directory
    ts = str(int(time.time()))
    trained_dir = 'trained_results_{}/'.format(ts)
    print('trained_dir:', trained_dir)
    if os.path.exists(trained_dir):
        shutil.rmtree(trained_dir)

    os.makedirs(trained_dir)
    graph = tf.Graph()
    with graph.as_default():
        sess_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session(config=sess_conf)
        with sess.as_default():
            model = TextModel(embedding_mat=embedding_mat,
                              seq_len=x_train.shape[1],
                              n_classes=y_train.shape[1],
                              non_static=constants['non_static'],
                              n_hidden=constants['n_hidden'],
                              max_pool_size=max_pool_size,
                              filter_sizes=constants['filter_sizes'],
                              n_filters=constants['n_filters'],
                              emb_dim=emb_dim,
                              l2_reg_lambda=constants['l2_reg_lambda'])
            global_step = tf.Variable(0, name='global_step', trainable=False)
            optimizer = tf.train.RMSPropOptimizer(constants['learning_rate'], decay=constants['decay'])
            grads_and_vars = optimizer.compute_gradients(model.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Checkpoint files will be saved in this directory during training
            checkpoint_dir = 'checkpoints_{}/'.format(ts)
            if os.path.exists(checkpoint_dir):
                shutil.rmtree(checkpoint_dir)

            os.makedirs(checkpoint_dir)
            checkpoint_prefix = os.path.join(checkpoint_dir, 'model')

            def real_len(batches):
                return [np.ceil(np.argmin(batch + [0]) * 1.0 / max_pool_size) for batch in batches]

            def train_step(x_batch, y_batch):
                feed_dict = {
                    model.input_x: x_batch,
                    model.input_y: y_batch,
                    model.keep_prob: constants['keep_prob'],
                    model.batch_size: len(x_batch),
                    model.pad: np.zeros([len(x_batch), 1, emb_dim, 1]),
                    model.real_len: real_len(x_batch)
                }
                _, step, loss_, accuracy_ = sess.run([train_op, global_step, model.loss, model.accuracy], feed_dict)

            def val_step(x_batch, y_batch):
                feed_dict = {
                    model.input_x: x_batch,
                    model.input_y: y_batch,
                    model.keep_prob: 1.0,
                    model.batch_size: len(x_batch),
                    model.pad: np.zeros([len(x_batch), 1, emb_dim, 1]),
                    model.real_len: real_len(x_batch)
                }
                step, loss_, accuracy_, num_correct, preds_ = sess.run([global_step, model.loss, model.accuracy,
                                                                        model.num_correct, model.preds], feed_dict)
                return accuracy_, loss_, num_correct, preds_

            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())

            # Train
            train_batches = batch_iter(list(zip(x_train, y_train)), batch_size, n_epochs)
            best_accuracy, best_at_step = 0, 0

            # Train the model with x_train and y_train
            for train_batch in train_batches:
                x_train_batch, y_train_batch = zip(*train_batch)
                train_step(x_train_batch, y_train_batch)
                current_step = tf.train.global_step(sess, global_step)

                # Evaluate the model with x_dev and y_dev
                if current_step % constants['eval_every'] == 0:
                    val_batches = batch_iter(list(zip(x_val, y_val)), batch_size, 1)
                    total_val_correct = 0
                    for val_batch in val_batches:
                        x_val_batch, y_val_batch = zip(*val_batch)
                        acc, loss, num_val_correct, preds = val_step(x_val_batch, y_val_batch)
                        total_val_correct += num_val_correct

                    accuracy = float(total_val_correct) / len(y_val)
                    logging.info('Accuracy on val set: {}'.format(accuracy))
                    if accuracy >= best_accuracy:
                        best_accuracy, best_at_step = accuracy, current_step
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        logging.critical('Saved model {} at step {}'.format(path, best_at_step))
                        logging.critical('Best accuracy {} at step {}'.format(best_accuracy, best_at_step))

            logging.critical('Training complete! Testing the best model on x_test, y_test...')

            # Save model files to trained_dir, predict needs trained model files
            saver.save(sess, trained_dir + 'best_model')

            # Evaluate x_test, y_test
            saver.restore(sess, checkpoint_prefix + '-' + str(best_at_step))
            test_batches = batch_iter(list(zip(x_test, y_test)), batch_size, 1, shuffle=False)
            total_test_correct = 0
            for test_batch in test_batches:
                x_test_batch, y_test_batch = zip(*test_batch)
                acc, loss, num_test_correct, preds = val_step(x_test_batch, y_test_batch)
                total_test_correct += int(num_test_correct)

            logging.critical('Accuracy on test set: {}'.format(float(total_test_correct) / len(y_test)))

        # Save trained parameters and files since predict needs them
        with open(trained_dir + 'words_index.json', 'w') as f:
            json.dump(vocab, f, indent=4, ensure_ascii=False)

        with open(trained_dir + 'embeddings.pkl', 'wb') as f:
            pickle.dump(embedding_mat, f, pickle.HIGHEST_PROTOCOL)

        with open(trained_dir + 'labels.json', 'w') as f:
            json.dump(labels, f, indent=4, ensure_ascii=False)

        constants['seq_len'] = x_train.shape[1]
        with open(trained_dir + 'trained_params.json', 'w') as f:
            json.dump(constants, f, indent=4, sort_keys=True, ensure_ascii=False)

    return trained_dir
