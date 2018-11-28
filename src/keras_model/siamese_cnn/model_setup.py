# noinspection PyPep8Naming
from keras import backend as K
from keras.layers import Activation, BatchNormalization, Concatenate, Conv1D, Dense, Dropout, Embedding
from keras.layers import GlobalMaxPooling1D, Input, Lambda, Multiply, Subtract
from keras.models import Model


# noinspection PyPep8Naming
def build_siamese_cnn_model(vocab_size, embed_dim, embeddings, has_shared_embedding, has_shared_filters,
                            filter_sizes, n_filters_per_size, is_normalized, n_fc_layers, fc_layer_dims, dropout_prob):
    # doc1 = resume
    # doc2 = ad
    if has_shared_embedding:
        shared_embedding_layer = Embedding(input_dim=vocab_size, output_dim=embed_dim, weights=[embeddings],
                                           input_length=None, trainable=True, name='shared_embedding')
        doc1_embedding = shared_embedding_layer
        doc2_embedding = shared_embedding_layer
    else:
        doc1_embedding = Embedding(input_dim=vocab_size, output_dim=embed_dim, weights=[embeddings],
                                   input_length=None, trainable=True, name='doc1_embedding')
        doc2_embedding = Embedding(input_dim=vocab_size, output_dim=embed_dim, weights=[embeddings],
                                   input_length=None, trainable=True, name='doc2_embedding')

    if has_shared_filters:
        shared_filter_layers = {}
        for sz in filter_sizes:
            shared_filter_layers[sz] = Conv1D(filters=n_filters_per_size, kernel_size=sz, padding='valid',
                                              activation='relu', strides=1, kernel_initializer='glorot_uniform',
                                              name='conv_{}_doc1'.format(sz))

        doc1_filters = shared_filter_layers
        doc2_filters = shared_filter_layers
    else:
        doc1_filters = {}
        for sz in filter_sizes:
            doc1_filters[sz] = Conv1D(filters=n_filters_per_size, kernel_size=sz, padding='valid',
                                      activation='relu', strides=1, kernel_initializer='glorot_uniform',
                                      name='conv_{}_doc1')

        doc2_filters = {}
        for sz in filter_sizes:
            doc2_filters[sz] = Conv1D(filters=n_filters_per_size, kernel_size=sz, padding='valid',
                                      activation='relu', strides=1, kernel_initializer='glorot_uniform',
                                      name='conv_{}_doc2')

    doc1_input = Input(shape=(None, ), name='doc1_input')
    doc1_embed = doc1_embedding(doc1_input)
    doc1_dropout = Dropout(dropout_prob, name='doc1_dropout')(doc1_embed)

    doc1_conv_blocks = []
    for sz in filter_sizes:
        doc1_conv = doc1_filters[sz](doc1_dropout)
        doc1_max_pool = GlobalMaxPooling1D(name='max_pool_{}_doc1'.format(sz))(doc1_conv)
        doc1_conv_blocks.append(doc1_max_pool)

    if len(doc1_conv_blocks) > 1:
        doc1_output = Concatenate(name='doc1_concat')(doc1_conv_blocks)
    else:
        doc1_output = doc1_conv_blocks[0]

    if is_normalized:
        doc1_output = Lambda(lambda x: K.l2_normalize(x, axis=1), name='doc1_l2')(doc1_output)

    doc2_input = Input(shape=(None, ), name='doc2_input')
    doc2_embed = doc2_embedding(doc2_input)
    doc2_dropout = Dropout(dropout_prob, name='doc2_dropout')(doc2_embed)

    doc2_conv_blocks = []
    for sz in filter_sizes:
        doc2_conv = doc2_filters[sz](doc2_dropout)
        doc2_max_pool = GlobalMaxPooling1D(name='max_pool_{}_doc2'.format(sz))(doc2_conv)
        doc2_conv_blocks.append(doc2_max_pool)

    if len(doc2_conv_blocks) > 1:
        doc2_output = Concatenate(name='doc2_concat')(doc2_conv_blocks)
    else:
        doc2_output = doc2_conv_blocks[0]

    if is_normalized:
        doc2_output = Lambda(lambda x: K.l2_normalize(x, axis=1), name='doc2_l2')(doc2_output)

    # Merged layers

    # Mix
    t_prod = Multiply(name='product')([doc1_output, doc2_output])
    # Keras 2.0.6 doesn't have keras.layers.Subtract, add a negative instead
    # Negative sign causes "attributeError: 'Tensor' object has no attribute '_keras_history'"!!
    # t_diff = Add(name='diff')([doc1_output, -doc2_output])
    t_diff = Subtract(name='diff')([doc1_output, doc2_output])

    # Merge
    X = Concatenate(name='merged')([t_diff, t_prod, doc1_output, doc2_output])
    X = BatchNormalization(name='bn_merged')(X)
    X = Dropout(dropout_prob, name='dropout_merged')(X)

    # Fully connected
    for fc in range(n_fc_layers):
        X = Dense(fc_layer_dims[fc], activation=None, name='fc_{}'.format(fc + 1))(X)
        X = BatchNormalization(name='bn_fc_{}'.format(fc + 1))(X)
        X = Activation('relu', name='relu_{}'.format(fc + 1))(X)

    # Output
    output = Dense(1, activation='sigmoid', name='output')(X)

    return Model(inputs=[doc1_input, doc2_input], outputs=output)
