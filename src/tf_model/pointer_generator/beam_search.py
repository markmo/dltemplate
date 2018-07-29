""" This file contains code to run beam search decoding """
import tf_model.pointer_generator.data as data
import numpy as np
import tensorflow as tf


FLAGS = tf.app.flags.FLAGS


class Hypothesis(object):
    """ Class to represent a hypothesis during beam search. Holds all the information needed for the hypothesis. """

    def __init__(self, tokens, log_probs, state, attn_dists, p_gens, coverage):
        """
        Hypothesis constructor.

        :param tokens: List of integers. The ids of the tokens that form the summary so far.
        :param log_probs: List, same length as tokens, of floats, giving the log probabilities
               of the tokens so far.
        :param state: Current state of the decoder, a LSTMStateTuple.
        :param attn_dists: List, same length as tokens, of numpy arrays with shape (attn_length).
               These are the attention distributions so far.
        :param p_gens: List, same length as tokens, of floats, or None if not using pointer-generator
               model. The values of the generation probability so far.
        :param coverage: Numpy array of shape (attn_length), or None if not using coverage.
               The current coverage vector.
        """
        self.tokens = tokens
        self.log_probs = log_probs
        self.state = state
        self.attn_dists = attn_dists
        self.p_gens = p_gens
        self.coverage = coverage

    def extend(self, token, log_prob, state, attn_dist, p_gen, coverage):
        """
        Return a NEW hypothesis, extended with the information from the latest step of beam search.

        :param token: (int) latest token produced by beam search
        :param log_prob: (float) log prob of the latest token
        :param state: current decoder state, a LSTMStateTuple
        :param attn_dist: attention distribution from latest step, numpy array of shape (attn_length)
        :param p_gen: (float) generation probability on latest step
        :param coverage: latest coverage vector, numpy array of shape (attn_length),
               or None if not using coverage
        :return: new Hypothesis for next step
        """
        return Hypothesis(tokens=self.tokens + [token],
                          log_probs=self.log_probs + [log_prob],
                          state=state,
                          attn_dists=self.attn_dists + [attn_dist],
                          p_gens=self.p_gens + [p_gen],
                          coverage=coverage)

    @property
    def latest_token(self):
        return self.tokens[-1]

    @property
    def log_prob(self):
        # the log probability of the hypothesis so far is the sum of the log probabilities
        # of the tokens so far
        return sum(self.log_probs)

    @property
    def avg_log_prob(self):
        # normalize log probability by number of tokens (otherwise longer sequences always
        # have lower probability)
        return self.log_prob / len(self.tokens)


def run_beam_search(sess, model, vocab, batch):
    """
    Performs beam search decoding on the given example.

    :param sess: a tf.Session
    :param model: a seq2seq model
    :param vocab: Vocabulary object
    :param batch: Batch object that is the same example repeated across the batch
    :return:
        best_hyp: Hypothesis object; the best hypothesis found by beam search.
    """
    # Run the encoder to get the encoder hidden states and decoder initial state
    enc_states, dec_in_state = model.run_encoder(sess, batch)
    # dec_in_state is a LSTMStateTuple
    # enc_states has shape [batch_size, <=max_enc_steps, 2*hidden_dim].

    # Initialize hypotheses (beam_size)
    hyps = [Hypothesis(tokens=[vocab.word2id(data.START_DECODING)],
                       log_probs=[0.],
                       state=dec_in_state,
                       attn_dists=[],
                       p_gens=[],
                       coverage=np.zeros([batch.enc_shape[1]])  # zero vector of length attention_length
                       ) for _ in range(FLAGS.beam_size)]
    results = []  # this will contain finished hypotheses (those that have emitted the [STOP] token)
    steps = 0
    while steps < FLAGS.max_dec_steps and len(results) < FLAGS.beam_size:
        latest_tokens = [h.latest_token for h in hyps]  # latest token produced by each hypothesis

        # change any in-article temporary OOV ids to [UNK] id, so that we can lookup word embeddings
        latest_tokens = [t if t in range(vocab.size()) else vocab.word2id(data.UNKNOWN_TOKEN)
                         for t in latest_tokens]
        states = [h.state for h in hyps]  # list of current decoder states of the hypotheses
        prev_coverage = [h.coverage for h in hyps]  # list of coverage vectors (or None)

        # Run one step of the decoder to get the new info
        top_k_ids, top_k_log_probs, new_states, attn_dists, p_gens, new_coverage = \
            model.decode_onestep(sess=sess, batch=batch, latest_tokens=latest_tokens,
                                 enc_states=enc_states, dec_init_states=states, prev_coverage=prev_coverage)

        # Extend each hypothesis and collect them all in all_hyps
        all_hyps = []

        # On the first step, we only had one original hypothesis (the initial hypothesis).
        # On subsequent steps, all original hypotheses are distinct.
        num_orig_hyps = 1 if steps == 0 else len(hyps)

        for i in range(num_orig_hyps):
            # take the ith hypothesis and new decoder state info
            h = hyps[i]
            new_state = new_states[i]
            attn_dist = attn_dists[i]
            p_gen = p_gens[i]
            new_coverage_i = new_coverage[i]

            # for each of the top 2 * beam_size hypotheses
            for j in range(FLAGS.beam_size * 2):
                # extend the ith hypothesis with the jth option
                new_hyp = h.extend(token=top_k_ids[i, j],
                                   log_prob=top_k_log_probs[i, j],
                                   state=new_state,
                                   attn_dist=attn_dist,
                                   p_gen=p_gen,
                                   coverage=new_coverage_i)
                all_hyps.append(new_hyp)

        # Filter and collect any hypotheses that have produced the end token.
        hyps = []  # will contain hypotheses for the next step
        for h in sort_hyps(all_hyps):  # in order of most likely hypothesis
            if h.latest_token == vocab.word2id(data.STOP_DECODING):  # if stop token is reached...
                # If this hypothesis is sufficiently long, put in results. Otherwise discard.
                if steps >= FLAGS.min_dec_steps:
                    results.append(h)
            else:  # hasn't reached stop token, so continue to extend this hypothesis
                hyps.append(h)

            if len(hyps) == FLAGS.bean_size or len(results) == FLAGS.beam_size:
                # Once we've collected beam_size-many hypotheses for the next step,
                # or beam_size-many complete hypotheses, stop.
                break

        steps += 1

    # At this point, either we've got beam_size results, or we've reached maximum decoder steps

    # if we don't have any complete results, then add all current hypotheses (incomplete summaries) to results
    if len(results) == 0:
        results = hyps

    # Sort hypotheses by average log probability
    hyps_sorted = sort_hyps(results)

    # Return the hypothesis with highest average log prob
    return hyps_sorted[0]


def sort_hyps(hyps):
    """ Return a list of Hypothesis objects, sorted by descending average log probability """
    return sorted(hyps, key=lambda h: h.avg_log_prob, reverse=True)
