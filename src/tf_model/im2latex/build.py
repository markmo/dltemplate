from argparse import ArgumentParser
from tf_model.im2latex.utils.data_generator import DataGenerator
from tf_model.im2latex.utils.general import Config
from tf_model.im2latex.utils.text import build_vocab, write_vocab


def run(data, vocab):
    data_conf = Config(data)
    train_set = DataGenerator(path_formulas=data_conf.path_formulas_train,
                              dir_images=data_conf.dir_images_train,
                              path_matching=data_conf.path_matching_train)

    val_set = DataGenerator(path_formulas=data_conf.path_formulas_val,
                            dir_images=data_conf.dir_images_val,
                            path_matching=data_conf.path_matching_val)

    test_set = DataGenerator(path_formulas=data_conf.path_formulas_test,
                             dir_images=data_conf.dir_images_test,
                             path_matching=data_conf.path_matching_test)

    # produce images and matching files
    train_set.build(buckets=data_conf.buckets)
    val_set.build(buckets=data_conf.buckets)
    test_set.build(buckets=data_conf.buckets)

    # vocab
    vocab_conf = Config(vocab)
    vocab = build_vocab([train_set], min_count=vocab_conf.min_count_tok)
    write_vocab(vocab, vocab_conf.path_vocab)


if __name__ == '__main__':
    # read args
    parser = ArgumentParser(description='Run img2latex build')
    parser.add_argument('--data', dest='data', default='configs/data_small.yml', help='path to data config')
    parser.add_argument('--vocab', dest='vocab', default='configs/vocab_small.yml', help='path to vocab config')
    args = parser.parse_args()
    v = vars(args)
    run(data=v['data'], vocab=v['vocab'])
