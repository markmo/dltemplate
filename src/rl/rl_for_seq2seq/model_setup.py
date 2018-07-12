from keras.layers import Dense, Embedding
from rl.rl_for_seq2seq.util import infer_length
import tensorflow as tf


class BasicTranslationModel(object):

    # noinspection PyUnusedLocal
    def __init__(self, name, inp_voc, out_voc, n_embedding, n_hidden):
        self.name = name
        self.inp_voc = inp_voc
        self.out_voc = out_voc
        with tf.variable_scope(name):
            self.emb_inp = Embedding(len(inp_voc), n_embedding)
            self.emb_out = Embedding(len(out_voc), n_embedding)
            self.enc0 = tf.nn.rnn_cell.GRUCell(n_hidden)
            self.dec_start = Dense(n_hidden)
            self.dec0 = tf.nn.rnn_cell.GRUCell(n_hidden)
            self.logits = Dense(len(out_voc))

            # run on dummy output to build all layers (and therefore create weights)
            inp = tf.placeholder('int32', [None, None])
            out = tf.placeholder('int32', [None, None])
            h0 = self.encode(inp)
            h1 = self.decode(h0, out[:, 0])

        self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)

    # noinspection PyUnusedLocal
    def encode(self, inp, **flags):
        """
        Takes symbolic input sequence, computes initial state

        :param inp: matrix of input tokens [batch, time]
        :param flags:
        :return: a list of initial decoder state tensors
        """
        inp_lengths = infer_length(inp, self.inp_voc.eos_id)
        inp_emb = self.emb_inp(inp)
        _, enc_last = tf.nn.dynamic_rnn(self.enc0, inp_emb, sequence_length=inp_lengths, dtype=inp_emb.dtype)
        dec_start = self.dec_start(enc_last)
        return [dec_start]

    # noinspection PyUnusedLocal
    def decode(self, prev_states, prev_tokens, **flags):
        """
        Takes previous decoder state and tokens, returns new state and logits

        :param prev_states: a list of previous decoder state tensors
        :param prev_tokens: previous output tokens, an int vector of [batch_size]
        :param flags:
        :return: a list of next decoder state tensors, a tensor of logits [batch, n_tokens]
        """
        [prev_dec] = prev_states
        prev_emb = self.emb_out(prev_tokens[:, None])[:, 0]
        new_dec_out, new_dec_state = self.dec0(prev_emb, prev_dec)
        output_logits = self.logits(new_dec_out)

        return [new_dec_state], output_logits

    def symbolic_score(self, inp, out, epsilon=1e-30, **flags):
        """
        Takes symbolic int32 matrices of Hebrew words and their English translations.

        Computes the log-probabilities of all possible English characters given
        English prefixes and Hebrew word.

        NOTE: the log-probabilities time axis is synchronized with out. In other words,
        logp are probabilities of the __current__ output at each tick, not the next one.
        Therefore, you can get likelihood as `log_probs * tf.one_hot(out, n_tokens)`.

        :param inp: input sequence, int32 matrix of shape [batch, time]
        :param out: output sequence, int32 matrix of shape [batch, time]
        :param epsilon:
        :param flags:
        :return: log-probabilities of all possible English characters of shape [bath, time, n_tokens]
        """
        first_state = self.encode(inp, **flags)
        batch_size = tf.shape(inp)[0]
        bos = tf.fill([batch_size], self.out_voc.bos_id)
        first_logits = tf.log(tf.one_hot(bos, len(self.out_voc)) + epsilon)

        def step(blob, y_prev):
            h_prev = blob[:-1]
            h_new, logits = self.decode(h_prev, y_prev, **flags)
            return list(h_new) + [logits]

        results = tf.scan(step, initializer=list(first_state) + [first_logits], elems=tf.transpose(out))

        # gather state and logits, each of shape [time, batch, ...]
        states_seq, logits_seq = results[:-1], results[-1]

        # add initial state and logits
        logits_seq = tf.concat((first_logits[None], logits_seq), axis=0)
        # states_seq = [tf.concat((init[None], states), axis=0)
        #               for init, states in zip(first_state, states_seq)]

        # convert from [time, batch, ...] to [batch, time, ...]
        logits_seq = tf.transpose(logits_seq, [1, 0, 2])
        # states_seq = [tf.transpose(states, [1, 0] + list(range(2, states.shape.ndims)))
        #               for states in states_seq]

        return tf.nn.log_softmax(logits_seq)

    # noinspection PyUnusedLocal
    def symbolic_translate(self, inp, greedy=False, max_len=None, epsilon=1e-30, **flags):
        """
        Takes symbolic int32 matrix of Hebrew words, produces output tokens sampled
        from the model and output log-probabilities for all possible tokens at each tick.

        :param inp: input sequence, int32 matrix of shape [batch, time]
        :param greedy: if greedy, takes token with highest probability at each tick.
                       Otherwise, samples proportionally to probability.
        :param max_len: max length of output, defaults to 2 * input length
        :param epsilon:
        :param flags:
        :return: output tokens int32[batch, time] and
                 log-probabilities of all tokens at each tick, [batch, time, n_tokens]
        """
        first_state = self.encode(inp, **flags)
        batch_size = tf.shape(inp)[0]
        bos = tf.fill([batch_size], self.out_voc.bos_id)
        first_logits = tf.log(tf.one_hot(bos, len(self.out_voc)) + epsilon)
        max_len = tf.reduce_max(tf.shape(inp)[1]) * 2

        # noinspection PyUnusedLocal
        def step(blob, t):
            h_prev, y_prev = blob[:-2], blob[-1]
            h_new, logits = self.decode(h_prev, y_prev, **flags)
            y_new = tf.argmax(logits, axis=-1) if greedy else tf.multinomial(logits, 1)[:, 0]
            return list(h_new) + [logits, tf.cast(y_new, y_prev.dtype)]

        results = tf.scan(step, initializer=list(first_state) + [first_logits, bos], elems=[tf.range(max_len)])

        # gather state, logits, outs, each of shape [time, batch, ...]
        states_seq, logits_seq, out_seq = results[:-2], results[-2], results[-1]

        # add initial state, logits and out
        logits_seq = tf.concat((first_logits[None], logits_seq), axis=0)
        out_seq = tf.concat((bos[None], out_seq), axis=0)
        # states_seq = [tf.concat((init[None], states), axis=0)
        #               for init, states in zip(first_state, states_seq)]

        # convert from [time, batch, ...] to [batch, time, ...]
        logits_seq = tf.transpose(logits_seq, [1, 0, 2])
        out_seq = tf.transpose(out_seq)
        # states_seq = [tf.transpose(states, [1, 0] + list(range(2, states.shape.ndims)))
        #               for states in states_seq]

        return out_seq, tf.nn.log_softmax(logits_seq)
