from keras_model.image_captioning.util import generate_vocabulary
from keras_model.image_captioning.util import get_captions, load_embeddings
from keras_model.image_captioning.util import PAD, UNK, START, END
import numpy as np
import random
import tqdm


def _test_vocab():
    embeddings = load_embeddings()
    captions_train = get_captions(embeddings['img_filenames_train'], embeddings['img_filenames_val'])
    vocab = generate_vocabulary(captions_train)
    return [
        len(vocab),
        len(np.unique(list(vocab.values()))),
        int(all([_ in vocab for _ in [PAD, UNK, START, END]]))
    ]


def _test_captions_indexing(captions_indexed_train, vocab, unk):
    starts = set()
    ends = set()
    between = set()
    unk_count = 0
    for caps in captions_indexed_train:
        for cap in caps:
            starts.add(cap[0])
            between.update(cap[1:-1])
            ends.add(cap[-1])
            for w in cap:
                if w == vocab[unk]:
                    unk_count += 1

    return [
        len(starts),
        len(ends),
        len(between),
        len(between | starts | ends),
        int(all([isinstance(x, int) for x in (between | starts | ends)])),
        unk_count
    ]


def _test_captions_batching(batch_captions_to_matrix):
    return (batch_captions_to_matrix([[1, 2, 3], [4, 5]], -1, max_len=None).ravel().tolist()
            + batch_captions_to_matrix([[1, 2, 3], [4, 5]], -1, max_len=2).ravel().tolist()
            + batch_captions_to_matrix([[1, 2, 3], [4, 5]], -1, max_len=10).ravel().tolist())


def _get_feed_dict_for_testing(decoder, img_embed_size, vocab):
    return {
        decoder.img_embeds: np.random.random((32, img_embed_size)),
        decoder.sentences: np.random.randint(0, len(vocab), (32, 20))
    }


def _test_decoder_shapes(decoder, img_embed_size, vocab, sess):
    tensors_to_test = [
        decoder.h0,
        decoder.word_embeds,
        decoder.flat_hidden_states,
        decoder.flat_token_logits,
        decoder.flat_ground_truth,
        decoder.flat_loss_mask,
        decoder.loss
    ]
    all_shapes = []
    for t in tensors_to_test:
        _ = sess.run(t, feed_dict=_get_feed_dict_for_testing(decoder, img_embed_size, vocab))
        all_shapes.extend(_.shape)

    return all_shapes


def _test_random_decoder_loss(decoder, img_embed_size, vocab, sess):
    return sess.run(decoder.loss, feed_dict=_get_feed_dict_for_testing(decoder, img_embed_size, vocab))


def test_validation_loss(decoder, sess, generate_batch, img_embeds_val, captions_indexed_val):
    np.random.seed(300)
    random.seed(300)
    val_loss = 0
    for _ in tqdm.tqdm_notebook(range(1000)):
        val_loss += sess.run(decoder.loss, generate_batch(img_embeds_val, captions_indexed_val, 32, 20))

    val_loss /= 1000.
    return val_loss
