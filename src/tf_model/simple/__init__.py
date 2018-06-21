from argparse import ArgumentParser
from common.load_data import load_mnist_dataset
from common.serving_util import serve
from common.util import merge_dict, one_hot_encode
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from tf_model.simple.hyperparams import get_constants
from tf_model.simple.model_setup import get_parameters, model_builder, network_builder
from tf_model.setup import get_inputs
from tf_model.util import train


def run(constant_overwrites):
    constants = merge_dict(get_constants(), constant_overwrites)
    x_train, y_train, x_val, y_val, x_test, y_test = load_mnist_dataset(flatten=True)

    if constants['retrain']:
        plt.figure(figsize=[6, 6])
        for i in range(4):
            plt.subplot(2, 2, i + 1)
            plt.title('Label: %i' % y_train[i])
            plt.imshow(x_train[i].reshape([28, 28]), cmap='gray')
        plt.show()

        n_classes = constants['n_classes']
        data = {
            'X_train': x_train,
            'y_train': one_hot_encode(y_train, n_classes),
            'X_val': x_val,
            'y_val': one_hot_encode(y_val, n_classes),
            'X_test': x_test,
            'y_test': one_hot_encode(y_test, n_classes)
        }
        input_x, input_y = get_inputs(constants)
        parameters = get_parameters(constants)
        optimizer, loss_op, predict_op, model, y_ = \
            model_builder(network_builder, input_x, input_y, parameters, constants)
        saver = tf.train.Saver()

        train(data, constants, (input_x, input_y), optimizer, loss_op, model, y_, minibatch=True, saver=saver)

    app_name = constants['app_name']
    model_name = constants['model_name']
    version = constants['version']
    model_dir = 'data'
    predict_op_name = 'predict_op:0'
    input_name = 'input_X:0'

    sess = tf.Session()
    saver = tf.train.import_meta_graph('{}/{}.meta'.format(model_dir, model_name))
    saver.restore(sess, tf.train.latest_checkpoint(model_dir))

    print('Testing...:')
    sample = x_test[random.randint(0, len(x_test))]
    plt.imshow(sample.reshape([28, 28]), cmap='gray')
    plt.show()
    pred = sess.run(predict_op_name, feed_dict={input_name: [sample]})
    print('prediction:', pred[0])

    if constants['serve']:
        print('Serving model.')
        serve(app_name=app_name,
              model_name=model_name,
              model_dir=model_dir,
              version=version,
              predict_op_name=predict_op_name,
              input_name=input_name,
              input_type='floats',
              default_output='-1.0')


if __name__ == '__main__':
    # read args
    parser = ArgumentParser(description='Run TF Simple NN model')
    parser.add_argument('--epochs', dest='n_epochs', type=int, help='number epochs')
    parser.add_argument('--batch-size', dest='batch_size', type=int, help='batch size')
    parser.add_argument('--learning-rate', dest='learning_rate', type=float, help='learning rate')
    parser.add_argument('--hidden-layers', dest='n_hidden', type=int, help='number hidden layers')
    parser.add_argument('--retrain', dest='retrain', help='retrain flag', action='store_true')
    parser.add_argument('--serve', dest='serve', help='serve flag', action='store_true')
    parser.set_defaults(retrain=False)
    parser.set_defaults(serve=False)
    args = parser.parse_args()

    run(vars(args))
