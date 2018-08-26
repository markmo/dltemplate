import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.python.ops import array_ops, math_ops, nn_ops, variable_scope
import time


# noinspection PyPep8Naming,PyUnusedLocal
def attention_decoder(decoder_inputs, initial_state, encoder_states, enc_padding_mask, cell,
                      initial_state_attention=False, pointer_gen=True, use_coverage=False, prev_coverage=None):
    """
    This function is based on tf.contrib.legacy_seq2seq_attention_decoder, which is now outdated.

    In the future, it would make more sense to write variants on the attention mechanism using
    the new seq2seq library for tensorflow 1.0: https://www.tensorflow.org/api_guides/python/contrib.seq2seq#Attention

    Note that this attention decoder passes each decoder input through a linear layer with
    the previous step's context vector to get a modified version of the input. If
    initial_state_attention is False, on the first decoder step the "previous context vector"
    is just a zero vector. If initial_state_attention is True, we use initial_state to
    (re)calculate the previous step's context vector. We set this to False for train/eval mode
    (because we call attention_decoder once for all decoder steps) and True for decode mode
    (because we call attention_decoder once for each decoder step).

    :param decoder_inputs: list of 2D Tensors [batch_size x input_size]
    :param initial_state: 2D Tensor [batch_size x cell.state_size]
    :param encoder_states: 3D Tensor [batch_size x attn_length x attn_size]
    :param enc_padding_mask: 2D Tensor [batch_size x attn_length] containing 1s and 0s;
           indicates which of the encoder locations are padding (0) or a real token (1).
    :param cell:rnn_cell.RNNCell defining the cell function and size
    :param initial_state_attention:
    :param pointer_gen: (bool) If True, calculate the generation probability p_gen for each decoder step.
    :param use_coverage: (bool) If True, use coverage mechanism.
    :param prev_coverage: If not None, a tensor with shape (batch_size, attn_length). The previous
           step's coverage vector. This is only not None in decode mode when using coverage.
    :return:
        outputs: A list of the same length as decoder_inputs of 2D Tensors of
                 shape [batch_size x cell.output_size]. The output vectors.
    state: The final state of the decoder. A tensor shape [batch_size x cell.state_size].
    attn_dists: A list containing tensors of shape (batch_size,attn_length).
                The attention distributions for each decoder step.
    p_gens: List of scalars. The values of p_gen for each decoder step. Empty list if pointer_gen=False.
    coverage: Coverage vector on the last step computed. None if use_coverage=False.
    """
    with variable_scope.variable_scope('attention_decoder') as scope:
        # if this line fails, it's because the batch size isn't defined
        batch_size = encoder_states.get_shape()[0].value

        # if this line fails, it's because the attention length isn't defined
        attn_size = encoder_states.get_shape()[2].value

        # Reshape encoder_states (need to insert a dim)
        encoder_states = tf.expand_dims(encoder_states, axis=2)  # now is shape (batch_size, attn_len, 1, attn_size)

        # To calculate attention:
        #   v^T tanh(W_h h_i + W_s s_t + b_attn)
        # where h_i is an encoder state, and s_t a decoder state.
        # attn_vec_size is the length of the vectors v, b_attn, (W_h h_i) and (W_s s_t)
        # We set it to be equal to the size of the encoder states.
        attention_vec_size = attn_size

        # Get the weight matrix W_h and apply it to each encoder state
        # to get (W_h h_i), the encoder features
        W_h = variable_scope.get_variable('W_h', [1, 1, attn_size, attention_vec_size])

        # shape (batch_size, attn_length, 1, attention_vec_size)
        encoder_features = nn_ops.conv2d(encoder_states, W_h, [1, 1, 1, 1], padding='SAME')

        # Get the weight vectors v and w_c (w_c is for coverage)
        v = variable_scope.get_variable('v', [attention_vec_size])

        if use_coverage:
            with variable_scope.variable_scope('coverage'):
                w_c = variable_scope.get_variable('w_c', [1, 1, 1, attention_vec_size])

        if prev_coverage is not None:  # for beam search mode with coverage
            # reshape from (batch_size, attn_length) to (batch_size, attn_len, 1, 1)
            prev_coverage = tf.expand_dims(tf.expand_dims(prev_coverage, 2), 3)

    def attention(decoder_state, coverage_=None):
        """
        Calculate the context vector and attention distribution from the decoder state.

        :param decoder_state: state of the decoder
        :param coverage_: (optional) previous timestep's coverage vector, shape (batch_size, attn_len, 1, 1)
        :return:
            context_vector: weighted sum of encoder_states
            attn_dist: attention distribution
            coverage: new coverage vector, shape (batch_size, attn_len, 1, 1)
        """
        with variable_scope.variable_scope('Attention'):
            # Pass the decoder state through a linear layer
            # (this is W_s s_t + b_attn in the paper)
            # shape (batch_size, attention_vec_size)
            decoder_features = linear(decoder_state, attention_vec_size, bias=True)

            # reshape to (batch_size, 1, 1, attention_vec_size)
            decoder_features = tf.expand_dims(tf.expand_dims(decoder_features, 1), 1)

            def masked_attention(e_):
                """ Take softmax of e then apply enc_padding_mask and re-normalize """
                attn_dist_ = nn_ops.softmax(e_)  # take softmax, shape (batch_size, attn_length)
                attn_dist_ *= enc_padding_mask  # apply mask
                masked_sums = tf.reduce_sum(attn_dist_, axis=1)  # shape (batch_size)
                return attn_dist_ / tf.reshape(masked_sums, [-1, 1])  # re-normalize

            if use_coverage and coverage_ is not None:  # non-first step of coverage
                # Multiply coverage vector by w_c to get coverage_features
                # c has shape (batch_size, attn_length, 1, attention_vec_size)
                coverage_features = nn_ops.conv2d(coverage_, w_c, [1, 1, 1, 1], padding='SAME')

                # Calculate v^T tanh(W_h h_i + W_s s_t + w_c c_i^t + b_attn)
                # shape (batch_size, attn_length)
                e = math_ops.reduce_sum(v * math_ops.tanh(encoder_features + decoder_features + coverage_features),
                                        [2, 3])

                # Calculate attention distribution
                attn_dist = masked_attention(e)

                # Update coverage vector
                coverage_ += array_ops.reshape(attn_dist, [batch_size, -1, 1, 1])
            else:
                # Calculate v^T tanh(W_h h_i + W_s s_t + b_attn)
                e = math_ops.reduce_sum(v * math_ops.tanh(encoder_features + decoder_features), [2, 3])

                # Calculate attention distribution
                attn_dist = masked_attention(e)

                if use_coverage:  # first step of training
                    coverage_ = tf.expand_dims(tf.expand_dims(attn_dist, 2), 2)  # initialize coverage

            # Calculate the context vector from attn_dist and encoder_states
            # shape (batch_size, attn_size)
            context_vector_ = math_ops.reduce_sum(array_ops.reshape(attn_dist, [batch_size, -1, 1, 1]) *
                                                  encoder_states, [1, 2])
            context_vector_ = array_ops.reshape(context_vector_, [-1, attn_size])

        return context_vector_, attn_dist, coverage_

    outputs = []
    attn_dists = []
    p_gens = []
    state = initial_state
    coverage = prev_coverage  # initialize coverage to None or whatever was passed in
    context_vector = array_ops.zeros([batch_size, attn_size])
    context_vector.set_shape([None, attn_size])  # ensure the second shape of attention vectors is set

    if initial_state_attention:  # true in decode mode
        # Re-calculate the context vector from the previous step so that we can pass
        # it through a linear layer with this step's input to get a modified version
        # of the input.
        # In decode mode, this is what updates the coverage vector
        context_vector, _, coverage = attention(initial_state, coverage)

    for i, inp in enumerate(decoder_inputs):
        tf.logging.info('Adding attention_decoder timestep %i of %i', i + 1, len(decoder_inputs))
        if i > 0:
            variable_scope.get_variable_scope().reuse_variables()

        # Merge input and previous attentions into one vector x of the same size as inp
        input_size = inp.get_shape().with_rank(2)[1]
        if input_size.value is None:
            raise ValueError('Could not infer input size from input: %s' % inp.name)

        x = linear([inp] + [context_vector], input_size, bias=True)

        # Run the decoder RNN cell. cell_output = decoder state
        cell_output, state = cell(x, state)

        # Run the attention mechanism.
        if i == 0 and initial_state_attention:  # always true in decode mode
            # you need this because you've already run the initial attention(...) call
            with variable_scope.variable_scope(variable_scope.get_variable_scope(), reuse=True):
                context_vector, attention_dist, _ = attention(state, coverage)  # don't allow coverage to update
        else:
            context_vector, attention_dist, coverage = attention(state, coverage)

        attn_dists.append(attention_dist)

        if pointer_gen:
            with tf.variable_scope('calculate_pgen'):
                p_gen = linear([context_vector, state.c, state.h, x], 1, bias=True)  # a scalar
                p_gen = tf.sigmoid(p_gen)
                p_gens.append(p_gen)

        # Concatenate the cell_output (= decoder state) and the context vector,
        # and pass them through a linear layer
        # This is V[s_t, h*_t] + b in the paper
        with variable_scope.variable_scope('AttnOutputProjection'):
            output = linear([cell_output] + [context_vector], cell.output_size, bias=True)

        outputs.append(output)

    # If using coverage, reshape it
    if coverage is not None:
        coverage = array_ops.reshape(coverage, [batch_size, -1])

    return outputs, state, attn_dists, p_gens, coverage


def linear(args, output_size, bias, bias_start=0.0, scope=None):
    """
    Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

    :param args: a 2D Tensor or a list of 2D, batch x n, Tensors
    :param output_size: (int) second dimension of W[i]
    :param bias: (bool) whether to add a bias term or not
    :param bias_start: starting value to initialize the bias; 0 by default.
    :param scope: VariableScope for the created subgraph; defaults to "Linear".
    :return:
        A 2D Tensor with shape [batch x output_size] equal to
        sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
    :raises ValueError: if some of the arguments have unspecified or wrong shape.
    """
    if args is None or (isinstance(args, (list, tuple)) and not args):
        raise ValueError("`args` must be specified")

    if not isinstance(args, (list, tuple)):
        args = [args]

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        if len(shape) != 2:
            raise ValueError('Linear is expecting 2D arguments: %s' % str(shapes))

        if not shape[1]:
            raise ValueError('Linear expects shape[1] of arguments: %s' % str(shapes))
        else:
            total_arg_size += shape[1]

    with tf.variable_scope(scope or 'Linear'):
        matrix = tf.get_variable('Matrix', [total_arg_size, output_size])
        if len(args) == 1:
            res = tf.matmul(args[0], matrix)
        else:
            res = tf.matmul(tf.concat(axis=1, values=args), matrix)

        if not bias:
            return res

        bias_term = tf.get_variable('Bias', [output_size], initializer=tf.constant_initializer(bias_start))

    return res + bias_term


class SummarizationModel(object):
    """
    A class to represent a sequence-to-sequence model for text summarization.
    Supports baseline mode, pointer-generator mode, and coverage.
    """

    def __init__(self, config, vocab):
        self._config = config
        self._vocab = vocab
        self.global_step = None
        self._summaries = None

    def _add_placeholders(self):
        """ Add placeholders to the graph. These are entry points for any input data. """
        config = self._config

        # encoder part
        self._enc_batch = tf.placeholder(tf.int32, [config.batch_size, None], name='enc_batch')
        self._enc_lens = tf.placeholder(tf.int32, [config.batch_size], name='enc_lens')
        self._enc_padding_mask = tf.placeholder(tf.float32, [config.batch_size, None], name='enc_padding_mask')
        if config.pointer_gen:
            self._enc_batch_extend_vocab = tf.placeholder(tf.int32, [config.batch_size, None],
                                                          name='enc_batch_extend_vocab')
            self._max_art_oovs = tf.placeholder(tf.int32, [], name='max_art_oovs')

        # decoder part
        self._dec_batch = tf.placeholder(tf.int32, [config.batch_size, config.max_dec_steps], name='dec_batch')
        self._target_batch = tf.placeholder(tf.int32, [config.batch_size, config.max_dec_steps], name='target_batch')
        self._dec_padding_mask = tf.placeholder(tf.float32, [config.batch_size, config.max_dec_steps],
                                                name='dec_padding_mask')
        if config.mode == 'decode' and config.coverage:
            self.prev_coverage = tf.placeholder(tf.float32, [config.batch_size, None], name='prev_coverage')

    def _make_feed_dict(self, batch, just_enc=False):
        """
        Make a feed dictionary mapping parts of the batch to the appropriate placeholders.

        :param batch: Batch object
        :param just_enc: (bool) If True, only feed the parts needed for the encoder.
        :return:
        """
        feed_dict = {
            self._enc_batch: batch.enc_batch,
            self._enc_lens: batch.enc_lens,
            self._enc_padding_mask: batch.enc_padding_mask
        }
        if self._config.pointer_gen:
            feed_dict[self._enc_batch_extend_vocab] = batch.enc_batch_extend_vocab
            feed_dict[self._max_art_oovs] = batch.max_art_oovs

        if not just_enc:
            feed_dict[self._dec_batch] = batch.dec_batch
            feed_dict[self._target_batch] = batch.target_batch
            feed_dict[self._dec_padding_mask] = batch.dec_padding_mask

        return feed_dict

    def _add_encoder(self, encoder_inputs, seq_len):
        """
        Add a single-layer bidirectional LSTM encoder to the graph.

        :param encoder_inputs: tensor of shape [batch_size, <=max_enc_steps, emb_size]
        :param seq_len: lengths of encoder_inputs (before padding). A tensor of shape [batch_size].
        :return:
            encoder_outputs: tensor of shape [batch_size, <=max_enc_steps, 2*hidden_dim].
                             It's 2*hidden_dim because it's the concatenation of the forwards and
                             backwards states.
            fw_state, bw_state: each are LSTMStateTuples of shape
                                ([batch_size, hidden_dim], [batch_size, hidden_dim])
        """
        with tf.variable_scope('encoder'):
            cell_fw = tf.contrib.rnn.LSTMCell(self._config.hidden_dim,
                                              initializer=self.rand_unif_init,
                                              state_is_tuple=True)
            cell_bw = tf.contrib.rnn.LSTMCell(self._config.hidden_dim,
                                              initializer=self.rand_unif_init,
                                              state_is_tuple=True)
            encoder_outputs, (fw_st, bw_st) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, encoder_inputs,
                                                                              dtype=tf.float32,
                                                                              sequence_length=seq_len,
                                                                              swap_memory=True)
            # concatenate the forwards and backwards states
            encoder_outputs = tf.concat(axis=2, values=encoder_outputs)

        return encoder_outputs, fw_st, bw_st

    def _reduce_states(self, fw_st, bw_st):
        """
        Add to the graph a linear layer to reduce the encoder's final FW and BW state
        into a single initial state for the decoder. This is needed because the encoder
        is bidirectional but the decoder is not.

        :param fw_st: LSTMStateTuple with hidden_dim units
        :param bw_st: LSTMStateTuple with hidden_dim units
        :return:
            state: LSTMStateTuple with hidden_dim units
        """
        hidden_dim = self._config.hidden_dim
        with tf.variable_scope('reduce_final_st'):
            # Define weights and biases to reduce the cell and state
            w_reduce_c = tf.get_variable('w_reduce_c', [hidden_dim * 2, hidden_dim],
                                         dtype=tf.float32, initializer=self.trunc_norm_init)
            w_reduce_h = tf.get_variable('w_reduce_h', [hidden_dim * 2, hidden_dim],
                                         dtype=tf.float32, initializer=self.trunc_norm_init)
            bias_reduce_c = tf.get_variable('bias_reduce_c', [hidden_dim],
                                            dtype=tf.float32, initializer=self.trunc_norm_init)
            bias_reduce_h = tf.get_variable('bias_reduce_h', [hidden_dim],
                                            dtype=tf.float32, initializer=self.trunc_norm_init)

            # Apply linear layer
            old_c = tf.concat(axis=1, values=[fw_st.c, bw_st.c])  # Concatenation of fw and bw cell
            old_h = tf.concat(axis=1, values=[fw_st.h, bw_st.h])  # Concatenation of fw and bw state
            new_c = tf.nn.relu(tf.matmul(old_c, w_reduce_c) + bias_reduce_c)  # Get new cell from old cell
            new_h = tf.nn.relu(tf.matmul(old_h, w_reduce_h) + bias_reduce_h)  # Get new state from old state

            return tf.contrib.rnn.LSTMStateTuple(new_c, new_h)  # Return new cell and state

    def _add_decoder(self, inputs):
        """
        Add attention decoder to the graph. In train or eval mode, you call this once
        to get output on ALL steps. In decode (beam search) mode, you call this once
        for EACH decoder step.

        :param inputs: inputs to the decoder (word embeddings).
               A list of tensors, shape (batch_size, emb_dim).
        :return:
            outputs: list of tensors; the outputs of the decoder
            out_state: final state of the decoder
            attn_dists: list of tensors; the attention distributions
            p_gens: list of scalar tensors; the generation probabilities
            coverage: tensor, the current coverage vector
        """
        config = self._config
        cell = tf.contrib.rnn.LSTMCell(config.hidden_dim, state_is_tuple=True, initializer=self.rand_unif_init)

        # In decode mode, we run attention_decoder one step at a time and
        # so need to pass in the previous step's coverage vector each time
        prev_coverage = self.prev_coverage if config.mode == 'decode' and config.coverage else None
        outputs, out_state, attn_dists, p_gens, coverage = \
            attention_decoder(inputs, self._dec_in_state, self._enc_states, self._enc_padding_mask, cell,
                              initial_state_attention=(config.mode == 'decode'),
                              pointer_gen=config.pointer_gen,
                              use_coverage=config.coverage,
                              prev_coverage=prev_coverage)

        return outputs, out_state, attn_dists, p_gens, coverage

    def _calc_final_dist(self, vocab_dists, attn_dists):
        """
        Calculate the final distribution for the pointer-generator model.

        :param vocab_dists: The vocabulary distributions. List length max_dec_steps of
               (batch_size, vocab_size) arrays. The words are in the order they appear in
               the vocabulary file.
        :param attn_dists: The attention distributions. List length max_dec_steps of
               (batch_size, attn_len) arrays
        :return:
            final_dists: The final distributions. List length max_dec_steps of
                         (batch_size, extended_vocab_size) arrays.
        """
        with tf.variable_scope('final_distribution'):
            # Multiply vocab dists by p_gen and attention dists by (1 - p_gen)
            vocab_dists = [p_gen * dist for p_gen, dist in zip(self.p_gens, vocab_dists)]
            attn_dists = [(1 - p_gen) * dist for p_gen, dist in zip(self.p_gens, attn_dists)]

            # Concatenate some zeros to each vocabulary dist to hold the
            # probabilities for in-article OOV words

            # the maximum (over the batch) size of the extended vocabulary
            extended_vocab_size = self._vocab.size() + self._max_art_oovs
            extra_zeros = tf.zeros((self._config.batch_size, self._max_art_oovs))

            # list length max_dec_steps of shape (batch_size, extended_vocab_size)
            vocab_dists_extended = [tf.concat(axis=1, values=[dist, extra_zeros]) for dist in vocab_dists]

            # Project the values in the attention distributions onto the appropriate
            # entries in the final distributions. This means that if a_i = 0.1 and
            # the ith encoder word is w, and w has index 500 in the vocabulary, then
            # we add 0.1 onto the 500th entry of the final distribution. This is done
            # for each decoder timestep. This is fiddly; we use tf.scatter_nd to do
            # the projection.
            batch_nums = tf.range(0, limit=self._config.batch_size)  # shape (batch_size)
            batch_nums = tf.expand_dims(batch_nums, 1)  # shape (batch_size, 1)
            attn_len = tf.shape(self._enc_batch_extend_vocab)[1]  # number of states we attend over
            batch_nums = tf.tile(batch_nums, [1, attn_len])  # shape (batch_size, attn_len)
            indices = tf.stack((batch_nums, self._enc_batch_extend_vocab), axis=2)  # shape (batch_size, enc_t, 2)
            shape = [self._config.batch_size, extended_vocab_size]

            # list length max_dec_steps (batch_size, extended_vocab_size)
            attn_dists_projected = [tf.scatter_nd(indices, copy_dist, shape) for copy_dist in attn_dists]

            # Add the vocab distributions and the copy distributions together to get
            # the final distributions. final_dists is a list length max_dec_steps;
            # each entry is a tensor shape (batch_size, extended_vocab_size) giving the
            # final distribution for that decoder timestep. Note that for decoder
            # timesteps and examples corresponding to a [PAD] token, this is junk - ignore.
            final_dists = [vocab_dist + copy_dist
                           for vocab_dist, copy_dist in zip(vocab_dists_extended, attn_dists_projected)]

            # OOV part of vocab is max_art_oov long. Not all the sequences in a batch
            # will have max_art_oov tokens. This will cause some entries to be 0 in
            # the distribution, which results in a NaN when calculating log_dists.
            # Clip the dist or add epsilon when that happens.
            # def add_epsilon(dist, epsilon=sys.float_info.epsilon):
            #     epsilon_mask = tf.ones_like(dist) * epsilon
            #     return dist + epsilon_mask
            #
            # final_dists = [add_epsilon(dist) for dist in final_dists]
            final_dists = [tf.clip_by_value(dist, sys.float_info.epsilon, 1.) for dist in final_dists]

            return final_dists

    def _add_emb_vis(self, embedding_var):
        """
        Do setup so that we can view word embedding visualization in TensorBoard, as described here:

            https://www.tensorflow.org/get_started/embedding_viz

        Make the vocab metadata file, then make the projector config file pointing to it.

        :param embedding_var:
        :return:
        """
        train_dir = os.path.join(self._config.log_root, 'train')
        vocab_metadata_path = os.path.join(train_dir, 'vocab_metadata.tsv')
        self._vocab.write_metadata(vocab_metadata_path)  # write metadata file
        summary_writer = tf.summary.FileWriter(train_dir)
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = embedding_var.name
        embedding.metadata_path = vocab_metadata_path
        projector.visualize_embeddings(summary_writer, config)

    def _add_seq2seq(self):
        """ Add the whole sequence-to-sequence model to the graph. """
        config = self._config
        vocab_size = self._vocab.size()
        with tf.variable_scope('seq2seq'):
            # Some initializers
            self.rand_unif_init = tf.random_uniform_initializer(-config.rand_unif_init_mag,
                                                                config.rand_unif_init_mag,
                                                                seed=123)
            self.trunc_norm_init = tf.truncated_normal_initializer(stddev=config.trunc_norm_init_std)

            # Add embedding matrix (shared by the encoder and decoder inputs)
            with tf.variable_scope('embedding'):
                embedding = tf.get_variable('embedding', [vocab_size, config.emb_dim], dtype=tf.float32,
                                            initializer=self.trunc_norm_init)
                if config.mode == 'train':
                    self._add_emb_vis(embedding)  # add to TensorBoard

                # tensor with shape (batch_size, max_enc_steps, emb_size)
                emb_enc_inputs = tf.nn.embedding_lookup(embedding, self._enc_batch)

                # list length max_dec_steps containing shape (batch_size, emb_size)
                emb_dec_inputs = [tf.nn.embedding_lookup(embedding, x)
                                  for x in tf.unstack(self._dec_batch, axis=1)]

            # Add the encoder
            enc_outputs, fw_st, bw_st = self._add_encoder(emb_enc_inputs, self._enc_lens)
            self._enc_states = enc_outputs

            # Our encoder is bidirectional and our decoder is unidirectional
            # so we need to reduce the final encoder hidden state to the right
            # size to be the initial decoder hidden state
            self._dec_in_state = self._reduce_states(fw_st, bw_st)

            # Add the decoder
            with tf.variable_scope('decoder'):
                decoder_outputs, self._dec_out_state, self.attn_dists, self.p_gens, self.coverage = \
                    self._add_decoder(emb_dec_inputs)

            # Add the output projection to obtain the vocabulary distribution
            with tf.variable_scope('output_projection'):
                w = tf.get_variable('w', [config.hidden_dim, vocab_size], dtype=tf.float32,
                                    initializer=self.trunc_norm_init)
                # w_t = tf.transpose(w)
                v = tf.get_variable('v', [vocab_size], dtype=tf.float32, initializer=self.trunc_norm_init)

                # vocab_scores is the vocabulary distribution before applying softmax.
                # Each entry on the list corresponds to one decoder step.
                vocab_scores = []
                for i, output in enumerate(decoder_outputs):
                    if i > 0:
                        tf.get_variable_scope().reuse_variables()

                    vocab_scores.append(tf.nn.xw_plus_b(output, w, v))  # apply the linear layer

                # The vocabulary distributions. List length max_dec_steps of (batch_size, vocab_size) arrays.
                # The words are in the order they appear in the vocabulary file.
                vocab_dists = [tf.nn.softmax(s) for s in vocab_scores]

            # For pointer-generator model, calc final distribution from copy distribution
            # and vocabulary distribution
            if config.pointer_gen:
                final_dists = self._calc_final_dist(vocab_dists, self.attn_dists)
            else:  # final distribution is just vocabulary distribution
                final_dists = vocab_dists

            if config.mode in ['train', 'eval']:
                # Calculate loss
                with tf.variable_scope('loss'):
                    if config.pointer_gen:
                        # Calculate loss per step
                        # This is fiddly; we use tf.gather_nd to pick out the probabilities of the gold target words
                        loss_per_step = []  # will be list length max_dec_steps containing shape (batch_size)
                        batch_nums = tf.range(0, limit=config.batch_size)  # shape (batch_size)
                        for dec_step, dist in enumerate(final_dists):
                            # The indices of the target words, shape (batch_size)
                            targets = self._target_batch[:, dec_step]
                            indices = tf.stack((batch_nums, targets), axis=1)  # shape (batch_size, 2)

                            # shape (batch_size). prob of correct words on this step.
                            gold_probs = tf.gather_nd(dist, indices)
                            losses = -tf.log(gold_probs)
                            loss_per_step.append(losses)

                        # Apply dec_padding_mask and get loss
                        self._loss = _mask_and_avg(loss_per_step, self._dec_padding_mask)

                    else:  # baseline model
                        # applies softmax internally
                        self._loss = tf.contrib.seq2seq.sequence_loss(tf.stack(vocab_scores, axis=1),
                                                                      self._target_batch,
                                                                      self._dec_padding_mask)
                    tf.summary.scalar('loss', self._loss)

                    # Calculate coverage loss from the attention distributions
                    if config.coverage:
                        with tf.variable_scope('coverage_loss'):
                            self._coverage_loss = _coverage_loss(self.attn_dists, self._dec_padding_mask)
                            tf.summary.scalar('coverage_loss', self._coverage_loss)

                        self._total_loss = self._loss + config.cov_loss_wt * self._coverage_loss
                        tf.summary.scalar('total_loss', self._total_loss)

        if config.mode == 'decode':
            # We run decode beam search mode one decoder step at a time
            # final_dists is a singleton list containing shape (batch_size, extended_vocab_size)
            assert len(final_dists) == 1
            final_dists = final_dists[0]

            # take the k largest probs. note batch_size=beam_size in decode mode.
            topk_probs, self._topk_ids = tf.nn.top_k(final_dists, config.batch_size * 2)
            self._topk_log_probs = tf.log(topk_probs)

    def _add_train_op(self):
        """ Sets self._train_op, the op to run for training. """
        # Take gradients of the trainable variables w.r.t. the loss function to minimize
        loss_to_minimize = self._total_loss if self._config.coverage else self._loss
        tvars = tf.trainable_variables()
        gradients = tf.gradients(loss_to_minimize, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)

        # Clip the gradients
        with tf.device('/gpu:0'):
            grads, global_norm = tf.clip_by_global_norm(gradients, self._config.max_grad_norm)

        # Add a summary
        tf.summary.scalar('global_norm', global_norm)

        # Apply adagrad optimizer
        optimizer = tf.train.AdagradOptimizer(self._config.learning_rate,
                                              initial_accumulator_value=self._config.adagrad_init_acc)
        with tf.device('/gpu:0'):
            self._train_op = optimizer.apply_gradients(zip(grads, tvars),
                                                       global_step=self.global_step, name='train_step')

    def build_graph(self):
        """ Add the placeholders, model, global step, train_op and summaries to the graph """
        tf.logging.info('Building graph...')
        t0 = time.time()
        self._add_placeholders()
        with tf.device('/gpu:0'):
            self._add_seq2seq()

        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        if self._config.mode == 'train':
            self._add_train_op()

        self._summaries = tf.summary.merge_all()
        t1 = time.time()
        tf.logging.info('Time to build graph: %i seconds', t1 - t0)

    def run_train_step(self, sess, batch):
        """
        Runs one training iteration. Returns a dictionary containing train op,
        summaries, loss, global_step and (optionally) coverage loss.

        :param sess:
        :param batch:
        :return:
        """
        feed_dict = self._make_feed_dict(batch)
        to_return = {
            'train_op': self._train_op,
            'summaries': self._summaries,
            'loss': self._loss,
            'global_step': self.global_step
        }
        if self._config.coverage:
            to_return['coverage_loss'] = self._coverage_loss

        return sess.run(to_return, feed_dict)

    def run_eval_step(self, sess, batch):
        """
        Runs one evaluation iteration. Returns a dictionary containing summaries,
        loss, global_step and (optionally) coverage loss.

        :param sess:
        :param batch:
        :return:
        """
        feed_dict = self._make_feed_dict(batch)
        to_return = {
            'summaries': self._summaries,
            'loss': self._loss,
            'global_step': self.global_step
        }
        if self._config.coverage:
            to_return['coverage_loss'] = self._coverage_loss

        return sess.run(to_return, feed_dict)

    def run_encoder(self, sess, batch):
        """
        For beam search decoding. Run the encoder on the batch and return
        the encoder states and decoder initial state.

        :param sess: TensorFlow session
        :param batch: Batch object that is the same example repeated across the batch (for beam search)
        :return:
            enc_states: The encoder states. A tensor of shape [batch_size, <=max_enc_steps, 2*hidden_dim].
            dec_in_state: A LSTMStateTuple of shape ([1, hidden_dim], [1, hidden_dim])
        """
        feed_dict = self._make_feed_dict(batch, just_enc=True)  # feed the batch into the placeholders

        # run the encoder
        enc_states, dec_in_state, global_step = \
            sess.run([self._enc_states, self._dec_in_state, self.global_step], feed_dict)

        # dec_in_state is LSTMStateTuple shape ([batch_size, hidden_dim], [batch_size, hidden_dim])
        # Given that the batch is a single example repeated, dec_in_state is identical across the
        # batch, so we just take the top row.
        dec_in_state = tf.contrib.rnn.LSTMStateTuple(dec_in_state.c[0], dec_in_state.h[0])

        return enc_states, dec_in_state

    def decode_one_step(self, sess, batch, latest_tokens, enc_states, dec_init_states, prev_coverage):
        """
        For beam search decoding. Run the decoder for one step.

        :param sess: TensorFlow session
        :param batch: Batch object containing single example repeated across the batch
        :param latest_tokens: Tokens to be fed as input into the decoder for this timestep
        :param enc_states: The encoder states.
        :param dec_init_states: list of beam_size LSTMStateTuples; the decoder states from the previous timestep
        :param prev_coverage: list of np arrays. The coverage vectors from the previous timestep.
               List of None if not using coverage.
        :return:
        ids: top 2k ids, shape [beam_size, 2*beam_size]
        probs: top 2k log probabilities, shape [beam_size, 2*beam_size]
        new_states: new states of the decoder, a list (length beam_size) containing
                    LSTMStateTuples each of shape ([hidden_dim, ], [hidden_dim, ])
        attn_dists: list (length beam_size) containing lists of length attn_length
        p_gens: Generation probabilities for this step. A list of length beam_size.
                List of None if in baseline mode.
        new_coverage: Coverage vectors for this step. A list of arrays.
                      List of None if coverage is not turned on.
        """
        beam_size = len(dec_init_states)

        # Turn dec_init_states (a list of LSTMStateTuples) into a single LSTMStateTuple for the batch
        cells = [np.expand_dims(state.c, axis=0) for state in dec_init_states]
        hiddens = [np.expand_dims(state.h, axis=0) for state in dec_init_states]
        new_c = np.concatenate(cells, axis=0)  # shape [batch_size, hidden_dim]
        new_h = np.concatenate(hiddens, axis=0)  # shape [batch_size, hidden_dim]
        new_dec_in_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)
        feed_dict = {
            self._enc_states: enc_states,
            self._enc_padding_mask: batch.enc_padding_mask,
            self._dec_in_state: new_dec_in_state,
            self._dec_batch: np.transpose(np.array([latest_tokens]))
        }
        to_return = {
            'ids': self._topk_ids,
            'probs': self._topk_log_probs,
            'states': self._dec_out_state,
            'attn_dists': self.attn_dists
        }
        if self._config.pointer_gen:
            feed_dict[self._enc_batch_extend_vocab] = batch.enc_batch_extend_vocab,
            feed_dict[self._max_art_oovs] = batch.max_art_oovs,
            to_return['p_gens'] = self.p_gens

        if self._config.coverage:
            feed_dict[self.prev_coverage] = np.stack(prev_coverage, axis=0)
            to_return['coverage'] = self.coverage

        results = sess.run(to_return, feed_dict=feed_dict)  # run the decoder step

        # Convert results['states'] (a single LSTMStateTuple) into a list of LSTMStateTuple -- one for each hypothesis
        new_states = [tf.contrib.rnn.LSTMStateTuple(results['states'].c[i, :], results['states'].h[i, :])
                      for i in range(beam_size)]

        # Convert singleton list containing a tensor to a list of k arrays
        assert len(results['attn_dists']) == 1
        attn_dists = results['attn_dists'][0].tolist()

        if self._config.pointer_gen:
            # Convert singleton list containing a tensor to a list of k arrays
            assert len(results['p_gens']) == 1
            p_gens = results['p_gens'][0].tolist()
        else:
            p_gens = [None for _ in range(beam_size)]

        # Convert the coverage tensor to a list length k containing the coverage vector for each hypothesis
        if self._config.coverage:
            new_coverage = results['coverage'].tolist()
            assert len(new_coverage) == beam_size
        else:
            new_coverage = [None for _ in range(beam_size)]

        return results['ids'], results['probs'], new_states, attn_dists, p_gens, new_coverage


def _mask_and_avg(values, padding_mask):
    """
    Applies mask to values then returns overall average (a scalar)

    :param values: a list (length max_dec_steps) containing arrays shape (batch_size)
    :param padding_mask: tensor shape (batch_size, max_dec_steps) containing 1s and 0s
    :return: a scalar
    """
    dec_lens = tf.reduce_sum(padding_mask, axis=1)  # shape batch_size. float32
    values_per_step = [v * padding_mask[:, dec_step] for dec_step, v in enumerate(values)]
    values_per_ex = sum(values_per_step) / dec_lens  # shape (batch_size); normalized value for each batch member

    return tf.reduce_mean(values_per_ex)  # overall average


def _coverage_loss(attn_dists, padding_mask):
    """
    Calculates the coverage loss from the attention distributions.

    :param attn_dists: The attention distributions for each decoder timestep.
           A list (length max_dec_steps) containing shape (batch_size, attn_length).
    :param padding_mask: shape (batch_size, max_dec_steps)
    :return:
        coverage_loss: scalar
    """
    coverage = tf.zeros_like(attn_dists[0])  # shape (batch_size, attn_length). Initial coverage is zero.

    # Coverage loss per decoder timestep. Will be list (length max_dec_steps) containing shape (batch_size).
    cov_losses = []
    for a in attn_dists:
        # calculate the coverage loss for this step
        cov_loss = tf.reduce_sum(tf.minimum(a, coverage), [1])
        cov_losses.append(cov_loss)
        coverage += a  # update the coverage vector

    coverage_loss = _mask_and_avg(cov_losses, padding_mask)

    return coverage_loss
