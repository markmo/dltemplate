import tensorflow as tf
from tensorflow.python.util import nest


def dynamic_decode(decoder_cell, max_iterations):
    """
    Similar to `dynamic_rnn` except to decode

    :param decoder_cell: (DecoderCell) with step method
    :param max_iterations: (int)
    :return:
    """
    try:
        max_iterations = tf.convert_to_tensor(max_iterations, dtype=tf.int32)
    except ValueError:
        pass

    # Create TA for outputs by mimicking the structure of DecoderCell output
    def create_ta(d):
        return tf.TensorArray(dtype=d, size=0, dynamic_size=True)

    initial_time = tf.constant(0, dtype=tf.int32)
    initial_outputs_ta = nest.map_structure(create_ta, decoder_cell.output_dtype)
    initial_state, initial_inputs, initial_finished = decoder_cell.initialize()

    # noinspection PyUnusedLocal
    def condition(time, unused_outputs_ta, unused_state, unused_inputs, finished):
        return tf.logical_not(tf.reduce_all(finished))

    def body(time, outputs_ta, state, inputs, finished):
        new_output, new_state, new_inputs, new_finished = decoder_cell.step(time, state, inputs, finished)
        outputs_ta = nest.map_structure(lambda ta, out: ta.write(time, out), outputs_ta, new_output)
        new_finished = tf.logical_or(tf.greater_equal(time, max_iterations), new_finished)
        return time + 1, outputs_ta, new_state, new_inputs, new_finished

    with tf.variable_scope('rnn'):
        res = tf.while_loop(condition,
                            body,
                            loop_vars=[initial_time, initial_outputs_ta, initial_state,
                                       initial_inputs, initial_finished],
                            back_prop=False)

    # get final outputs and states
    final_outputs_ta, final_state = res[1], res[2]

    # unfold and stack the structure from the nested tas
    final_outputs = nest.map_structure(lambda ta: ta.stack(), final_outputs_ta)

    # finalize the computation from the decoder cell
    final_outputs = decoder_cell.finalize(final_outputs, final_state)

    # transpose the final output
    final_outputs = nest.map_structure(transpose_batch_time, final_outputs)

    return final_outputs, final_state


def transpose_batch_time(t):
    n_dims = t.shape.ndims
    if n_dims == 2:
        return tf.transpose(t, [1, 0])
    elif n_dims == 3:
        return tf.transpose(t, [1, 0, 2])
    elif n_dims == 4:
        return tf.transpose(t, [1, 0, 2, 3])
    else:
        raise NotImplementedError
