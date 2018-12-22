from argparse import ArgumentParser
from common.model_util import load_hyperparams, merge_dict
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras_model.decomposable_attention.model_setup import decomposable_attention, esim
from keras_model.decomposable_attention.util import BatchGenerator, BucketedBatchGenerator, bucket_cases
from keras_model.decomposable_attention.util import load_bucketed_data, load_embeddings
from keras_model.decomposable_attention.util import load_question_pairs_dataset, preprocess
from keras_model.layers import MaskedGlobalAveragePooling1D, MaskedGlobalMaxPooling1D
import numpy as np
import os
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split


def run(constant_overwrites):
    print('Running')
    root_dir = os.path.dirname(__file__)
    config_path = os.path.join(root_dir, 'hyperparams.yml')
    constants = merge_dict(load_hyperparams(config_path), constant_overwrites)
    n_epochs = constants['n_epochs']
    batch_size = constants['batch_size']
    max_len = constants['max_len']

    embeddings, word2idx, _, _ = load_embeddings(os.path.join(root_dir, 'ft.vec'))
    print('embeddings length:', len(embeddings))
    print('word2idx length:', len(word2idx))
    ft_matrix_filename = os.path.join(root_dir, 'fasttext_matrix.npy')
    if not os.path.exists(ft_matrix_filename):
        np.save(ft_matrix_filename, embeddings)

    train_df, _ = load_question_pairs_dataset()
    print('train_df length:', len(train_df))
    train_df = preprocess(train_df, word2idx, max_len)
    train_df = bucket_cases(train_df, n_doc1_quantile=5, n_doc2_quantile=5)
    train_df, test_df = train_test_split(train_df, test_size=0.1, shuffle=True)
    train_df, val_df = train_test_split(train_df, test_size=0.1, shuffle=True)

    if constants['model_type'] == 'decom_attn':
        model = decomposable_attention(pretrained_embedding=ft_matrix_filename, max_len=max_len)
        checkpoint_dir = os.path.join(root_dir, 'decom_attn_checkpoint')
    else:
        model = esim(pretrained_embedding=ft_matrix_filename, max_len=max_len)
        checkpoint_dir = os.path.join(root_dir, 'esim_checkpoint')

    # print('\nModel summary:')
    # print(model.summary())

    checkpoint = ModelCheckpoint(checkpoint_dir, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    if constants['train']:
        bucketed_train = load_bucketed_data(train_df)
        # training_generator = BatchGenerator(np.asarray(train_df.q1_encoded.tolist()),
        #                                     np.asarray(train_df.q2_encoded.tolist()),
        #                                     train_df.is_duplicate.values, batch_size)
        training_generator = BucketedBatchGenerator(bucketed_train, batch_size,
                                                    max_doc1_length=max_len, max_doc2_length=max_len, shuffle=True)

        bucketed_val = load_bucketed_data(val_df)
        # validation_data = BatchGenerator(np.asarray(val_df.q1_encoded.tolist()),
        #                                  np.asarray(val_df.q2_encoded.tolist()),
        #                                  val_df.is_duplicate.values, batch_size)
        validation_data = BucketedBatchGenerator(bucketed_val, batch_size,
                                                 max_doc1_length=max_len, max_doc2_length=max_len, shuffle=True)

        # noinspection PyUnusedLocal
        train_history = model.fit_generator(training_generator, steps_per_epoch=len(training_generator),
                                            epochs=n_epochs, use_multiprocessing=True, workers=6,
                                            validation_data=validation_data,
                                            validation_steps=len(validation_data) / batch_size,
                                            callbacks=[checkpoint],
                                            verbose=1,
                                            initial_epoch=0  # use when restarting training (*zero* based)
                                            )
    else:
        model = load_model(checkpoint_dir, custom_objects={
            'MaskedGlobalAveragePooling1D': MaskedGlobalAveragePooling1D(),
            'MaskedGlobalMaxPooling1D': MaskedGlobalMaxPooling1D()
        })
        preds = model.predict([np.asarray(test_df.q1_encoded.tolist()),
                               np.asarray(test_df.q2_encoded.tolist())],
                              batch_size=batch_size, verbose=1)
        y_test = test_df.is_duplicate.values
        dev_auc = roc_auc_score(y_test, preds)
        dev_acc = accuracy_score(y_test, (preds > 0.5).astype(int))
        print('\nTest set AUC: {:.6f}, accuracy: {:.6f}'.format(dev_auc, dev_acc))
        print('Test majority class baseline accuracy: {:.6f}'.format(1 - sum(y_test) / len(y_test)))


if __name__ == '__main__':
    # read args
    parser = ArgumentParser(description='Run Decomposable Attention Model')
    parser.add_argument('--epochs', dest='n_epochs', type=int, help='number epochs')
    parser.add_argument('--batch-size', dest='batch_size', type=int, help='batch size')
    parser.add_argument('--length', dest='max_len', type=int, help='maximum sequence length')
    parser.add_argument('--model', dest='model_type', type=int, help='model type: "decom_attn" (default), "esim"')
    parser.add_argument('--train', dest='train', help='run training', action='store_true')
    parser.set_defaults(train=False)
    args = parser.parse_args()

    run(vars(args))
