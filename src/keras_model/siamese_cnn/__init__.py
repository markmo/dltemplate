from argparse import ArgumentParser
# from common.load_data import load_question_pairs_dataset
from common.model_util import load_hyperparams, merge_dict
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.models import load_model
from keras_model.siamese_cnn.model_setup import build_siamese_cnn_model
from keras_model.siamese_cnn.util import BucketedBatchGenerator, bucket_cases, create_vocab, encode, fully_padded_batch
from keras_model.siamese_cnn.util import load_bucketed_data, load_embeddings, prepare_training_data, write_bucket
from keras_model.siamese_cnn.util import timed
import numpy as np
import os
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

DATA_DIR = os.path.join(os.path.dirname(__file__), '../../../data/')
ROOT_DIR = os.path.dirname(__file__)


# noinspection PyUnusedLocal
def load_question_pairs_dataset(test_size=1000):
    train_df = pd.read_csv(DATA_DIR + 'question_pairs/train_full.csv', header=0)
    # test_df = pd.read_csv(DATA_DIR + 'question_pairs/test.csv', header=0)
    return (train_df[['qid1', 'qid2', 'question1', 'question2', 'is_duplicate']],
            None)  # test_df[['question1', 'question2']][:test_size])


def load_data():
    with timed('loading dataset'):
        train_df, test_df = load_question_pairs_dataset(test_size=1000)

    train_df = prepare_training_data(train_df)
    return train_df, test_df


def run(constant_overwrites):
    print('Running')
    config_path = os.path.join(os.path.dirname(__file__), 'hyperparams.yml')
    constants = merge_dict(load_hyperparams(config_path), constant_overwrites)

    model_path = os.path.join(ROOT_DIR, constants['model_path'])
    output_dir = os.path.join(ROOT_DIR, constants['output_dir'])
    output_dataset_name = constants['output_dataset_name']
    max_doc1_length = 60
    max_doc2_length = 60
    batch_size = constants['batch_size']
    if not output_dir.endswith('/'):
        output_dir += '/'

    if constants['train']:
        embed_size = constants['embed_size']
        word2vec_filename = os.path.join(ROOT_DIR, constants['word2vec_filename'])
        has_shared_embedding = constants['has_shared_embedding']
        has_shared_filters = constants['has_shared_filters']
        filter_sizes = constants['filter_sizes']
        n_filters_per_size = constants['n_filters_per_size']
        is_normalized = constants['is_normalized']
        n_fc_layers = constants['n_fc_layers']
        dropout_prob = constants['dropout_prob']
        merged_layer_size = 4 * len(filter_sizes) * n_filters_per_size
        fc_layer_dims = (int(merged_layer_size / 2), int(merged_layer_size / 4))
        assert len(fc_layer_dims) == n_fc_layers
        n_gpus = constants['n_gpus']
        n_epochs = constants['n_epochs']
        cache_dir = os.path.join(ROOT_DIR, constants['cache_dir'])
        idx2word_path = '{}/idx2word.pkl'.format(cache_dir)
        doc1_encoded_path = '{}/doc1_encoded.npy'.format(cache_dir)
        doc2_encoded_path = '{}/doc2_encoded.npy'.format(cache_dir)
        train_df_path = '{}/train_df.pkl'.format(cache_dir)
        embeddings_path = '{}/embeddings.npy'.format(cache_dir)
        train_df_save_columns = ['qid1', 'qid2', 'question1', 'question2', 'q1_length', 'q2_length',
                                 'q1_encoded', 'q2_encoded', 'is_duplicate']

        train_df = None
        if os.path.exists(embeddings_path):
            with timed('restoring embeddings'):
                embeddings = np.load(embeddings_path)
        else:
            if os.path.exists(idx2word_path):
                with timed('restoring idx2word'):
                    with open(idx2word_path, 'rb') as f:
                        idx2word = pickle.load(f)
            else:
                train_df, _ = load_data()
                word2idx, idx2word = create_vocab(train_df)
                if os.path.exists(doc1_encoded_path):
                    with timed('loading encoding'):
                        q1_encoded = np.load(doc1_encoded_path)
                        q2_encoded = np.load(doc2_encoded_path)
                        train_df = train_df.assign(q1_encoded=q1_encoded, q2_encoded=q2_encoded)
                else:
                    train_df = encode(train_df, word2idx)
                    np.save(doc1_encoded_path, train_df.q1_encoded.values)
                    np.save(doc2_encoded_path, train_df.q2_encoded.values)

                with timed('saving train_df'):
                    save_df = train_df[train_df_save_columns]
                    save_df.to_pickle(train_df_path)

                with timed('saving idx2word'):
                    with open(idx2word_path, 'wb') as f:
                        pickle.dump(idx2word, f)

            embeddings = load_embeddings(idx2word, embed_size, word2vec_filename)
            with timed('saving embeddings'):
                np.save(embeddings_path, embeddings)

        vocab_size = embeddings.shape[0]
        model = build_siamese_cnn_model(vocab_size, embed_size, embeddings, has_shared_embedding, has_shared_filters,
                                        filter_sizes, n_filters_per_size, is_normalized, n_fc_layers, fc_layer_dims,
                                        dropout_prob)
        print('\nModel summary:')
        print(model.summary())

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        if not os.path.exists(output_dir + 'train'):
            if train_df is None:
                if os.path.exists(train_df_path):
                    with timed('restoring train_df'):
                        train_df = pd.read_pickle(train_df_path)
                else:
                    train_df, _ = load_data()
                    if os.path.exists(doc1_encoded_path):
                        with timed('loading encoding'):
                            q1_encoded = np.load(doc1_encoded_path)
                            q2_encoded = np.load(doc2_encoded_path)
                            train_df = train_df.assign(q1_encoded=q1_encoded, q2_encoded=q2_encoded)
                    else:
                        if os.path.exists(idx2word_path):
                            with timed('restoring idx2word'):
                                with open(idx2word_path, 'rb') as f:
                                    idx2word = pickle.load(f)
                                    word2idx = {word: i for i, word in idx2word.items()}
                        else:
                            word2idx, _ = create_vocab(train_df)

                        train_df = encode(train_df, word2idx)
                        np.save(doc1_encoded_path, train_df.q1_encoded.values)
                        np.save(doc2_encoded_path, train_df.q2_encoded.values)

                    with timed('saving train_df'):
                        save_df = train_df[train_df_save_columns]
                        save_df.to_pickle(train_df_path)

            # temp_df = train_df.assign(not_duplicate=train_df.is_duplicate.apply(lambda x: not x))
            # summary = temp_df.groupby('qid2').agg({'qid1': 'count', 'is_duplicate': 'sum', 'not_duplicate': 'sum'})
            # summary = summary.assign(balance=(summary.is_duplicate / summary.qid1))
            # summary.to_csv(os.path.join(ROOT_DIR, 'summary.csv'), header=True, index=True)

            # filtered_q2 = summary[(summary.qid1 > 4) &
            #                       (summary.balance >= 0.1) &
            #                       (summary.balance <= 0.9)][['qid1', 'balance']]
            # filtered_q2.columns = ['total_q2', 'balance']
            # filtered_q2 = filtered_q2.reset_index()
            # filtered_cases = pd.merge(train_df, filtered_q2, on='qid2', how='inner')
            # print('\n{} cases filtered down to {} by removing q2s with less than 4 q1s '
            #       'or with is_duplicate less than 10% or greater than 90%'
            #       .format(train_df.shape[0], filtered_q2.shape[0]))

            # train_cases, dev_cases = train_test_split(filtered_cases, test_size=0.1, random_state=42)

            train_cases, dev_cases = train_test_split(train_df, test_size=0.1, random_state=42)
            train_cases = bucket_cases(train_cases, n_doc1_quantile=3, n_doc2_quantile=3)
            dev_cases = bucket_cases(dev_cases, n_doc1_quantile=3, n_doc2_quantile=3)

            # write_bucket(dev_cases, output_dir + output_dataset_name + '/dev/')

            buckets = train_cases['bucket'].unique()
            for b in buckets:
                write_bucket(train_cases[train_cases.bucket == b], output_dir + output_dataset_name + '/train/', b)

            buckets = dev_cases['bucket'].unique()
            for b in buckets:
                write_bucket(dev_cases[dev_cases.bucket == b], output_dir + output_dataset_name + '/dev/', b)

            trainval_cases: pd.DataFrame = pd.concat([train_cases, dev_cases])

            distinct_q1 = trainval_cases.set_index('qid1').to_dict()['q1_encoded']
            with open(output_dir + output_dataset_name + '/docs1.pkl', 'wb') as f:
                pickle.dump(distinct_q1, f)

            distinct_q2 = trainval_cases.set_index('qid2').to_dict()['q2_encoded']
            with open(output_dir + output_dataset_name + '/docs2.pkl', 'wb') as f:
                pickle.dump(distinct_q2, f)

        input_dir = output_dir + output_dataset_name

        print('')
        with timed('loading bucketed data'):
            bucketed_data = load_bucketed_data(input_dir)

        with timed('preparing validation data'):
            validation_data = fully_padded_batch(bucketed_data, 'dev', max_doc1_length, max_doc2_length)

        with timed('creating training generator'):
            training_generator = BucketedBatchGenerator(bucketed_data, batch_size, split='train',
                                                        max_doc1_length=max_doc1_length,
                                                        max_doc2_length=max_doc2_length,
                                                        gpus=n_gpus)

        checkpoint = ModelCheckpoint(model_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        tb = TensorBoard(log_dir='{}/logs/{}'.format(ROOT_DIR, constants['model_path']), histogram_freq=2,
                         write_grads=True, batch_size=batch_size, write_graph=True)

        with timed('fitting model'):
            # noinspection PyUnusedLocal
            train_history = model.fit_generator(training_generator, steps_per_epoch=len(training_generator),
                                                epochs=n_epochs, use_multiprocessing=True, workers=6,
                                                validation_data=validation_data,
                                                validation_steps=len(validation_data) / batch_size,
                                                callbacks=[tb, checkpoint],
                                                verbose=1,
                                                initial_epoch=0  # use when restarting training (*zero* based)
                                                )
    else:
        with timed('loading model'):
            model = load_model(model_path)

        input_dir = output_dir + output_dataset_name

        print('')
        with timed('loading bucketed data'):
            bucketed_data = load_bucketed_data(input_dir)

        with timed('preparing validation data'):
            validation_data = fully_padded_batch(bucketed_data, 'dev', max_doc1_length, max_doc2_length)

        with timed('scoring'):
            dev_scores = model.predict(validation_data[0], batch_size=batch_size, verbose=1)

        dev_auc = roc_auc_score(validation_data[1], dev_scores)
        dev_acc = accuracy_score(validation_data[1], (dev_scores > 0.5).astype(int))

        print('\nValidation set AUC: {:.6f}, accuracy: {:.6f}'.format(dev_auc, dev_acc))
        print('Validation majority class baseline accuracy: {:.6f}'
              .format(1 - sum(validation_data[1]) / len(validation_data[1])))


if __name__ == '__main__':
    # read args
    parser = ArgumentParser(description='Run Siamese CNN')
    parser.add_argument('--epochs', dest='n_epochs', type=int, help='number epochs')
    parser.add_argument('--cache-dir', dest='cache_dir', type=str, help='path to cache dir')
    parser.add_argument('--train', dest='train', help='run training', action='store_true')
    parser.set_defaults(train=False)
    args = parser.parse_args()

    run(vars(args))
