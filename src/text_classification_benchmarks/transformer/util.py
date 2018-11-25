import codecs
import gensim
import numpy as np
import os
import pickle
import tensorflow as tf
from text_classification_benchmarks.transformer.model_setup import Transformer
from tflearn.data_utils import pad_sequences

PAD = '__PAD__'
GO = '__GO__'
END = '__END__'
ROOT = os.path.dirname(os.path.realpath(__file__))


def train(training_data_path, n_classes, learning_rate, batch_size, n_epochs, decay_steps, decay_rate,
          seq_len, embed_size, d_model, d_k, d_v, h, n_layers, l2_lambda, keep_prob, checkpoint_dir,
          use_embedding, vocab_labels_filename, word2vec_filename, validate_step, is_multilabel,
          name_scope='transformer_classification'):
    word2idx, idx2word = create_vocab(word2vec_filename, name_scope=name_scope)
    vocab_size = len(word2idx)
    print('vocab_size:', vocab_size)
    word2idx_label, idx2word_label = create_vocab_labels(vocab_labels_filename, name_scope=name_scope)
    (x_train, y_train), (x_test, y_test), _ = \
        load_data_multilabel_new(training_data_path, word2idx, word2idx_label, is_multilabel=is_multilabel)
    x_train = pad_sequences(x_train, maxlen=seq_len, value=0.)
    x_test = pad_sequences(x_test, maxlen=seq_len, value=0.)
    print('x_test length:', len(x_test))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        model = Transformer(n_classes, learning_rate, batch_size, decay_steps, decay_rate, seq_len,
                            vocab_size, embed_size, d_model, d_k, d_v, h, n_layers, l2_lambda=l2_lambda,
                            decoder_sent_len=seq_len, is_training=True)
        saver = tf.train.Saver()
        if os.path.exists(os.path.join(checkpoint_dir, 'checkpoint')):
            print('Restoring variables from checkpoint')
            saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))
        else:
            print('Initializing variables')
            sess.run(tf.global_variables_initializer())
            if use_embedding:
                # Load pretrained word embeddings
                assign_pretrained_word_embeddings(sess, idx2word, vocab_size, model, embed_size,
                                                  word2vec_filename=word2vec_filename)

        curr_epoch = sess.run(model.epoch_step)
        input_size = len(x_train)
        print('input_size:', input_size)
        prev_eval_loss = 10000
        best_eval_loss = 10000
        for epoch in range(curr_epoch, n_epochs):
            loss, acc, counter = 0.0, 0.0, 0
            for start, end in zip(range(0, input_size, batch_size), range(batch_size, input_size, batch_size)):
                if epoch == 0 and counter == 0:
                    print('x_train[start:end]:', x_train[start:end])

                feed_dict = {
                    model.input_x: x_train[start:end],
                    model.input_y_label: y_train[start:end],
                    model.keep_prob: keep_prob
                }
                curr_loss, curr_acc, _ = sess.run([model.loss_val, model.accuracy, model.train_op], feed_dict)
                loss, counter, acc = loss + curr_loss, counter + 1, acc + curr_acc
                if counter % 50 == 0:
                    print('Epoch: %d\tBatch: %d\tTrain Loss: %.3f\tTrain Accuracy: %.3f' %
                          (epoch + 1, counter, loss / float(counter), acc / float(counter)))

                if batch_size != 0 and start % (validate_step * batch_size) == 0:
                    eval_loss, eval_acc = do_eval(sess, model, x_test, y_test, batch_size)
                    print('Validation prev_eval_loss:', prev_eval_loss, 'curr_eval_loss:', eval_loss)
                    if eval_loss > prev_eval_loss:  # if loss is not decreasing
                        # reduce the learning rate by a factor of 0.5
                        print('reducing learning rate by half')
                        lr1 = sess.run(model.learning_rate)
                        sess.run([model.learning_rate_decay_half_op])
                        lr2 = sess.run(model.learning_rate)
                        print('lr1:', lr1, 'lr2:', lr2)
                    else:  # loss is decreasing
                        if eval_loss < best_eval_loss:
                            print('Saving the model... eval_loss:', eval_loss, 'best_eval_loss:', best_eval_loss)
                            # Save model to checkpoint
                            print('checkpoint_dir:', os.path.abspath(checkpoint_dir))
                            save_path = checkpoint_dir + 'model'
                            saver.save(sess, save_path, global_step=epoch)
                            best_eval_loss = eval_loss

                    prev_eval_loss = eval_loss

            print('Incrementing epoch counter')
            sess.run(model.epoch_increment)

        test_loss, test_acc = do_eval(sess, model, x_test, y_test, batch_size)

    return test_loss, test_acc


def assign_pretrained_word_embeddings(sess, idx2word, vocab_size, model, embed_size, word2vec_filename=None):
    print('Using pretrained word embeddings:', os.path.abspath(word2vec_filename))
    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_filename, binary=True, limit=500000)
    word2vec_dict = {}
    for word, vector in zip(word2vec_model.vocab, word2vec_model.vectors):
        word2vec_dict[word] = vector

    word_embedding_2d_list = [[]] * vocab_size  # create an empty word embedding list
    # noinspection PyTypeChecker
    word_embedding_2d_list[0] = np.zeros(embed_size)  # assign empty for first word: 'PAD'
    bound = np.sqrt(6.0) / np.sqrt(vocab_size)  # bound for random variables
    count_exist = 0
    count_not_exist = 0
    for i in range(1, vocab_size):  # loop each word
        word = idx2word[i]
        embedding = word2vec_dict.get(word, None)
        if embedding is not None:
            assert len(embedding) == embed_size, 'embedding size: {}'.format(len(embedding))
            word_embedding_2d_list[i] = embedding
            count_exist += 1
        else:
            # noinspection PyTypeChecker
            word_embedding_2d_list[i] = np.random.uniform(-bound, bound, embed_size)
            count_not_exist += 1

    word_embedding_final = np.array(word_embedding_2d_list)  # convert to 2d array
    word_embedding = tf.constant(word_embedding_final, dtype=tf.float32)  # convert to tensor
    t_assign_embedding = tf.assign(model.embedding, word_embedding)
    sess.run(t_assign_embedding)
    print('count_exist:', count_exist, 'count_not_exist:', count_not_exist)


# noinspection PyUnusedLocal
def do_eval(sess, model, x_val, y_val, batch_size, eval_decoder_input=None, is_multi_label=False):
    n_examples = len(x_val)
    print('n_examples:', n_examples)
    eval_loss, eval_acc, eval_counter = 0.0, 0.0, 0
    for start, end in zip(range(0, n_examples, batch_size), range(batch_size, n_examples, batch_size)):
        feed_dict = {
            model.input_x: x_val[start:end],
            model.keep_prob: 1.0,
            model.input_y_label: y_val[start:end]
        }
        # if is_multi_label:
        #     feed_dict[model.input_y_label] = y_val[start:end]
        #     feed_dict[model.decoder_input] = eval_decoder_input[start:end]
        # else:
        #     feed_dict[model.input_y] = y_val[start:end]

        curr_eval_loss, logits, curr_eval_acc, pred = sess.run([model.loss_val, model.logits, model.accuracy,
                                                                model.predictions], feed_dict)
        eval_loss, eval_acc, eval_counter = eval_loss + curr_eval_loss, eval_acc + curr_eval_acc, eval_counter + 1

    return eval_loss / float(eval_counter), eval_acc / float(eval_counter)


def create_vocab(word2vec_filename, name_scope=None):
    name_scope = '' if name_scope is None else name_scope + '_'
    cache_path = '{}/vocab_cache/{}vocab.pkl'.format(ROOT, name_scope)
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            word2idx, idx2word = pickle.load(f)
    else:
        word2idx, idx2word = {}, {}
        print('Building vocabulary, word2vec_filename:', word2vec_filename)

        # doesn't work. see https://github.com/nicholas-leonard/word2vec/issues/25
        # model = word2vec.load(word2vec_filename, kind='bin', encoding='ISO-8859-1')
        model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_filename, binary=True, limit=500000)
        word2idx[PAD] = 0
        idx2word[0] = PAD
        special_idx = 0
        if 'biLstmTextRelation' in name_scope:
            word2idx['EOS'] = 1
            idx2word[1] = 'EOS'
            special_idx = 1

        for i, word in enumerate(model.vocab):
            word2idx[word] = i + 1 + special_idx
            idx2word[i + 1 + special_idx] = word

        if not os.path.exists(cache_path):
            with open(cache_path, 'ab') as f:
                pickle.dump((word2idx, idx2word), f)

    return word2idx, idx2word


def create_vocab_labels_file(vocab_labels_filename, classes):
    with codecs.open(vocab_labels_filename, 'w') as f:
        for cls in classes:
            f.write('__label__{}\n'.format(cls))


def create_vocab_labels(vocab_labels_filename, name_scope=None, use_seq2seq=False):
    name_scope = '' if name_scope is None else name_scope + '_'
    cache_path = '{}/vocab_labels_cache/{}vocab_labels.pkl'.format(ROOT, name_scope)
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            word2idx_labels, idx2word_labels = pickle.load(f)
    else:
        with codecs.open(vocab_labels_filename, 'r', 'utf8') as f:
            word2idx_labels, idx2word_labels, label_count_dict = {}, {}, {}
            for i, line in enumerate(f.readlines()):
                if '__label__' in line:
                    label = line[line.index('__label__') + len('__label__'):].strip().replace('\n', '')
                    k = label_count_dict.get(label, 0)
                    label_count_dict[label] = k + 1

        label_list = sort_by_value(label_count_dict)
        count = 0
        if use_seq2seq:  # if seq2seq model, insert two special labels ['__GO__', '__END__']
            for i, label in enumerate([GO, END, PAD]):
                word2idx_labels[label] = i
                idx2word_labels[i] = label

        for i, label in enumerate(label_list):
            if i < 10:
                count += label_count_dict[label]

            idx = i + 3 if use_seq2seq else i
            word2idx_labels[label] = idx
            idx2word_labels[idx] = label

        if not os.path.exists(cache_path):
            with open(cache_path, 'ab') as f:
                pickle.dump((word2idx_labels, idx2word_labels), f)

    return word2idx_labels, idx2word_labels


def sort_by_value(d):
    back_items = [[v[1], v[0]] for v in d.items()]
    back_items.sort(reverse=True)
    return [back_items[i][1] for i in range(len(back_items))]


def predict(test_file, n_classes, learning_rate, batch_size, decay_steps, decay_rate, seq_len,
            embed_size, d_model, d_k, d_v, h, n_layers, l2_lambda, checkpoint_dir,
            vocab_labels_filename, word2vec_filename, name_scope='transformer_classification'):
    word2idx, idx2word = create_vocab(word2vec_filename, name_scope=name_scope)
    vocab_size = len(word2idx)
    print('Transformer Classification Vocab size:', vocab_size)
    word2idx_label, idx2word_label = create_vocab_labels(vocab_labels_filename, name_scope=name_scope)
    test_raw = load_final_test_data(test_file)
    test_data = load_data_predict(word2idx, test_raw)
    x = []
    labels = []
    for label, indices in test_data:
        labels.append(label)
        x.append(indices)

    x_test = pad_sequences(x, seq_len, value=0)  # padding to max length

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        model = Transformer(n_classes, learning_rate, batch_size, decay_steps, decay_rate, seq_len,
                            vocab_size, embed_size, d_model, d_k, d_v, h, n_layers, l2_lambda=l2_lambda,
                            is_training=False)
        saver = tf.train.Saver()
        if os.path.exists(checkpoint_dir):
            print('Restoring variables from checkpoint')
            saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))
        else:
            raise Exception("Can't find checkpoint, stopping.")

        data_len = len(x_test)
        print('data length:', data_len)
        result = []
        for start, end in zip(range(0, data_len, batch_size), range(batch_size, data_len + 1, batch_size)):
            # [batch_size, n_classes]
            logits = sess.run(model.logits, feed_dict={model.input_x: x_test[start:end], model.keep_prob: 1.0})
            indices_batch = x[start:end]
            result.extend(get_label_using_logits_batch(indices_batch, logits, idx2word_label, top=1))

        return result, labels[:len(result)]


def get_label_using_logits_batch(indices_batch, logits_batch, idx2word_label, top=5):
    # print('logits_batch shape:', np.array(logits_batch).shape)
    result = []
    for i, logits in enumerate(logits_batch):
        labels = get_label_using_logits(logits, idx2word_label, top=top)
        result.append((indices_batch[i], labels))

    return result


def get_label_using_logits(logits, idx2word_label, top=5):
    indices = np.argsort(logits)[-top:]
    indices = indices[::-1]  # reverse
    labels = []
    for i in indices:
        label = idx2word_label[i]
        labels.append(label)

    return labels


def load_final_test_data(filename):
    print('Loading data...')
    with codecs.open(filename, 'r', 'utf8') as f:
        lines = f.readlines()

    data = []
    for i, line in enumerate(lines):
        x, y = line.split('__label__')
        y = y.strip().replace('\n', '')
        x = x.strip().split(' ')
        data.append((x, y))

    print('test data length:', len(data))
    return data


def load_final_test_data_tsv(filename):
    data = []
    with codecs.open(filename, 'r') as f:
        for line in f.readlines():
            label, text = line.split('\t')
            text = text.strip().replace('\n', '')
            data.append((label, text))

    print('test data length:', len(data))
    return data


def load_data_predict(word2idx, test_raw, do_gen_ngrams=False):
    data = []
    for text, label in test_raw:
        if do_gen_ngrams:
            words = generate_ngrams(text).split(' ')
        else:
            words = text.split(' ') if type(text) == str else text

        indices = [word2idx.get(w, 0) for w in words]  # unknown words set to 0 = '__PAD__'
        data.append((label, indices))

    print('n_examples:', len(data))
    return data


def generate_ngrams(text, ngram=3):
    """

    :param text: (str) example: 'w17314 w5521 w7729 w767 w10147 w111'
    :param ngram: (int) range of ngrams to generate
    :return: (str) example: 'w17314 w17314w5521 w17314w5521w7729 w5521 w5521w7729 w5521w7729w767 w7729 w7729w767
                             w7729w767w10147 w767 w767w10147 w767w10147w111 w10147 w10147w111 w111'
    """
    result = []
    words = text.split(' ') if type(text) == str else text
    sent_len = len(words)
    for i, word in enumerate(words):
        words_i = word  # unigram
        if ngram > 1 and i + 1 < sent_len:  # bigram
            words_i += ' ' + ''.join(words[i:i + 2])

        if ngram > 2 and i + 2 < sent_len:  # trigram
            words_i += ' ' + ''.join(words[i:i + 3])

        if ngram > 3 and i + 3 < sent_len:  # four-gram
            words_i += ' ' + ''.join(words[i:i + 4])

        if ngram > 4 and i + 4 < sent_len:  # five-gram
            words_i += ' ' + ''.join(words[i:i + 5])

        result.append(words_i)

    return ' '.join(result)


# noinspection PyPep8Naming
def load_data_multilabel_new(training_data_path, word2idx, word2idx_label, val_portion=0.05,
                             is_multilabel=False, use_seq2seq=False, seq2seq_label_len=6):
    print('Loading data...')
    with codecs.open(training_data_path, 'r', 'utf8') as f:
        lines = f.readlines()

    X, Y, Y_decoder_input = [], [], []
    for i, line in enumerate(lines):
        x, y = line.split('__label__')
        y = y.strip().replace('\n', '')
        x = x.strip().split(' ')
        x = [word2idx.get(w, 0) for w in x]  # for unknowns set index to 0 ('__PAD__')
        if use_seq2seq:
            # prepare label for seq2seq format: add '__GO__', '__END__', '__PAD__'
            ys = y.split(' ')
            pad_idx = word2idx_label[PAD]
            ys_multihot = [pad_idx] * seq2seq_label_len
            ys_decoder_input = [pad_idx] * seq2seq_label_len
            for j, y in enumerate(ys):
                if j < seq2seq_label_len - 1:
                    ys_multihot[j] = word2idx_label[y]

            ys_len = len(ys)
            if ys_len > seq2seq_label_len - 1:
                ys_multihot[seq2seq_label_len - 1] = word2idx_label[END]
            else:
                ys_multihot[ys_len] = word2idx_label[END]

            ys_decoder_input[0] = word2idx_label[GO]
            for j, y in enumerate(ys):
                if j < seq2seq_label_len - 1:
                    ys_decoder_input[j + 1] = word2idx_label[y]

        else:
            if is_multilabel:
                # prepare multi-label format for classification
                ys = y.split(' ')
                y_indices = []
                for y in ys:
                    y_idx = word2idx_label[y]
                    y_indices.append(y_idx)

                ys_multihot = transform_multilabel_as_multihot(y_indices)
            else:
                # prepare single label format for classification
                ys_multihot = word2idx_label[y]

        X.append(x)
        Y.append(ys_multihot)
        if use_seq2seq:
            # noinspection PyUnboundLocalVariable
            Y_decoder_input.append(ys_decoder_input)

    n_examples = len(X)
    print('n_examples:', n_examples)
    # print('val_portion:', val_portion)
    # print('train ratio:', 1.0 - val_portion)
    print('n_examples_train:', int((1 - val_portion) * n_examples))
    train_ = (X[0:int((1 - val_portion) * n_examples)], Y[0:int((1 - val_portion) * n_examples)])
    test_ = (X[int((1 - val_portion) * n_examples) + 1:], Y[int((1 - val_portion) * n_examples) + 1:])
    if use_seq2seq:
        train_ += (Y_decoder_input[0:int((1 - val_portion) * n_examples)],)
        test_ += (Y_decoder_input[int((1 - val_portion) * n_examples) + 1:],)

    return train_, test_, test_


def transform_multilabel_as_multihot(labels, label_size=1999):
    multihot = np.zeros(label_size)
    multihot[labels] = 1
    return multihot
