Summarization using LSTM
========================

A simpler LSTM based approach until I get the `Pointer Generator Network <src/tf_model/pointer_generator>`_
to perform.

This model implements abstractive text summarization. It uses a character-level seq2seq
model to predict summaries. The model employs a biLSTM architecture.

Character-level Sequence-to-sequence Algorithm:

1. Start with input sequences from a domain (e.g. text documents) and corresponding
   target sequences from another domain (e.g. text summaries).
2. An encoder LSTM transforms input sequences into two state vectors. (We keep the
   last LSTM state and discard the outputs.)
3. A decoder LSTM is trained to transform the target sequences into the same sequence
   but offset by one timestep in the future - a training process known as "teacher
   forcing" in this context. It uses the state vectors from the encoder as initial
   state. Essentially, the decoder learns to generate 'targets[t + 1...]' given
   'targets[...t]', conditioned on the input sequence.
4. In inference mode, to decode unseen input sequences:

   * Encode the input sequence into state vectors
   * Start with a target sequence of size 1 (just the "start-of-sequence character")
   * Feed the state vectors and 1-char target sequence into the decoder to
     produce predictions of the next character
   * Sample the next character using these predictions (using `argmax`)
   * Append the sampled character to the target sequence
   * Repeat until the "end-of-sequence character" is generated or we reach the
     character limit.

Relevant datasets:

* `English to French sentence pairs <http://www.manythings.org/anki/fra-eng.zip>`_
* Lots of neat sentence pairs datasets can be found at `http://www.manythings.org/anki/ <http://www.manythings.org/anki/>`_

References:

1. `Sequence to Sequence Learning with Neural Networks <https://arxiv.org/abs/1409.3215>`_
2. `Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation <https://arxiv.org/abs/1406.1078>`_
