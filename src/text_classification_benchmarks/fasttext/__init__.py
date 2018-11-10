from fastText import train_supervised
import os
from text_classification_benchmarks.fasttext.util import preprocess_csv


def print_results(n, p, r):
    print('N\t' + str(n))
    print('P@{}\t{:.3f}'.format(1, p))
    print('R@{}\t{:.3f}'.format(1, r))


if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_path = dir_path + '/train.txt'
    if not os.path.exists(data_path):
        x_train, y_train, x_val, y_val, x_test, y_test = preprocess_csv()
        print('Counts x_train: {}, y_train: {}, x_val: {}, y_val: {}, x_test: {}, y_test: {}'
              .format(len(x_train), len(y_train), len(x_val), len(y_val), len(x_test), len(y_test)))

    model = train_supervised(data_path, epoch=25, lr=1.0, wordNgrams=3, verbose=2, minCount=1, loss='hs')
    print_results(*model.test(dir_path + '/val.txt'))
    model.save_model('classifier.bin')

    model.quantize(input=data_path, qnorm=True, retrain=True, cutoff=100000)
    print_results(*model.test(dir_path + '/val.txt'))
    model.save_model('classifier.ftz')
