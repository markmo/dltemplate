import tensorflow as tf
from tensorflow.python.ops import array_ops, math_ops, nn_ops, variable_scope


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

    def attention(decoder_state, coverage=None):
        """
        Calculate the context vector and attention distribution from the decoder state.

        :param decoder_state: state of the decoder
        :param coverage: (optional) previous timestep's coverage vector, shape (batch_size, attn_len, 1, 1)
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

            def masked_attention(e):
                """ Take softmax of e then apply enc_padding_mask and re-normalize """
                attn_dist = nn_ops.softmax(e)  # take softmax, shape (batch_size, attn_length)
                attn_dist *= enc_padding_mask  # apply mask
                masked_sums = tf.reduce_sum(attn_dist, axis=1)  # shape (batch_size)
                return attn_dist / tf.reshape(masked_sums, [-1, 1])  # re-normalize

            if use_coverage and coverage is not None:  # non-first step of coverage
                # Multiply coverage vector by w_c to get coverage_features
                # c has shape (batch_size, attn_length, 1, attention_vec_size)
                coverage_features = nn_ops.conv2d(coverage, w_c, [1, 1, 1, 1], padding='SAME')

                # Calculate v^T tanh(W_h h_i + W_s s_t + w_c c_i^t + b_attn)
                # shape (batch_size, attn_length)
                e = math_ops.reduce_sum(v * math_ops.tan(encoder_features + decoder_features + coverage_features),
                                        [2, 3])

                # Calculate attention distribution
                attn_dist = masked_attention(e)

                # Update coverage vector
                coverage += array_ops.reshape(attn_dist, [batch_size, -1, 1, 1])
            else:
                # Calculate v^T tanh(W_h h_i + W_s s_t + b_attn)
                e = math_ops.reduce_sum(v * math_ops.tanh(encoder_features + decoder_features), [2, 3])

                # Calculate attention distribution
                attn_dist = masked_attention(e)

                if use_coverage:  # first step of training
                    coverage = tf.expand_dims(tf.expand_dims(attn_dist, 2), 2)  # initialize coverage

            # Calculate the context vector from attn_dist and encoder_states
            # shape (batch_size, attn_size)
            context_vector = math_ops.reduce_sum(array_ops.reshape(attn_dist, [batch_size, -1, 1, 1]) * encoder_features, [1, 2])
            context_vector = array_ops.reshape(context_vector, [-1, attn_size])

        return context_vector, attn_dist, coverage

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
        tf.logging.info('Adding attention_decoder timestep %i of %i', i, len(decoder_inputs))
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
                context_vector, attn_dist, _ = attention(state, coverage)  # don't allow coverage to update
        else:
            context_vector, attn_dist, coverage = attention(state, coverage)

        attn_dists.append(attn_dist)

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
