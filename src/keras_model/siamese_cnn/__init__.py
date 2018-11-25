from argparse import ArgumentParser
from common.load_data import load_question_pairs_dataset
from common.util import load_hyperparams, merge_dict
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras_model.siamese_cnn.model_setup import build_siamese_cnn_model
from keras_model.siamese_cnn.util import BucketedBatchGenerator, bucket_cases, create_vocab, fully_padded_batch
from keras_model.siamese_cnn.util import load_bucketed_data, load_embeddings, write_bucket
import os
import pandas as pd
from sklearn.model_selection import train_test_split


def run(constant_overwrites):
    config_path = os.path.join(os.path.dirname(__file__), 'hyperparams.yml')
    constants = merge_dict(load_hyperparams(config_path), constant_overwrites)
    embed_size = constants['embed_size']
    word2vec_filename = constants['word2vec_filename']
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
    model_path = constants['model_path']
    batch_size = constants['batch_size']
    n_gpus = constants['n_gpus']
    output_dir = constants['output_dir']
    output_dataset_name = constants['output_dataset_name']
    input_dir = output_dir
    n_epochs = constants['n_epochs']
    max_doc1_length = 50
    max_doc2_length = 50

    train_df, test_df = load_question_pairs_dataset(test_size=1000)
    docs1_train, docs2_train, y_train, word2idx, idx2word, train_df = create_vocab(train_df)
    # docs1_train, docs1_val, docs2_train, docs2_val, y_train, y_val = \
    #     train_test_split(docs1_train, docs2_train, y_train, test_size=0.1, random_state=42)
    vocab_size = len(word2idx)
    embeddings = load_embeddings(idx2word, embed_size, word2vec_filename)

    model = build_siamese_cnn_model(vocab_size, embed_size, embeddings, has_shared_embedding, has_shared_filters,
                                    filter_sizes, n_filters_per_size, is_normalized, n_fc_layers, fc_layer_dims,
                                    dropout_prob)
    print(model.summary())

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    train_df = train_df.assign('not_duplicate', train_df.is_duplicate.apply(lambda x: not x))
    summary = train_df.groupby('q2id').agg({'q1id': 'count', 'is_duplicate': 'sum', 'not_duplicate': 'sum'})
    summary = summary.assign('balance', summary.is_duplicate / summary.q1id)
    print(summary)

    filtered_q2 = summary[(summary.q1id > 4) & (summary.balance >= 0.1) & (summary.balance <= 0.0)][['q1id', 'balance']]
    filtered_q2.columns = ['total_q2', 'balance']
    filtered_q2 = filtered_q2.reset_index()
    filtered_cases = pd.merge(train_df, filtered_q2, on='q2id', how='inner')
    print('{} cases filtered down to {} by removing q2s with less than 4 q1s '
          'or with is_duplicate % less than 10% or greater than 90%'.format(train_df.shape[0], filtered_q2.shape[0]))
    train_cases, dev_cases = train_test_split(filtered_cases, test_size=0.1, random_state=42)
    train_cases = bucket_cases(train_cases, n_doc1_quantile=10, n_doc2_quantile=10)

    if not output_dir.endswith('/'):
        output_dir += '/'

    write_bucket(dev_cases, output_dir + output_dataset_name + '/dev/')
    buckets = train_cases['bucket'].unique()
    for b in buckets:
        write_bucket(train_cases[train_cases.bucket == b], output_dir + output_dataset_name + '/train/', b)

    bucketed_data = load_bucketed_data(input_dir)
    validation_data = fully_padded_batch(bucketed_data, 'dev', max_doc1_length, max_doc2_length)

    training_generator = BucketedBatchGenerator(bucketed_data, batch_size, split='train',
                                                max_doc1_length=max_doc1_length, max_doc2_length=max_doc2_length,
                                                gpus=n_gpus)

    checkpoint = ModelCheckpoint(model_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    tb = TensorBoard(log_dir='./logs/' + model_path, histogram_freq=2, write_grads=True,
                     batch_size=batch_size, write_graph=True)

    train_history = model.fit_generator(training_generator, steps_per_epoch=len(training_generator),
                                        epochs=n_epochs, use_multiprocessing=True, workers=6,
                                        validation_data=validation_data, callbacks=[tb, checkpoint],
                                        verbose=1,
                                        initial_epoch=0  # use when restarting training (*zero* based)
                                        )


if __name__ == '__main__':
    # read args
    parser = ArgumentParser(description='Run Siamese CNN')
    parser.add_argument('--epochs', dest='n_epochs', type=int, help='number epochs')
    args = parser.parse_args()

    run(vars(args))
