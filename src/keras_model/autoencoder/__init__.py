from argparse import ArgumentParser
from common.load_data import load_faces_dataset
from common.util import apply_gaussian_noise, merge_dict
from common.util_keras import reset_tf_session, TqdmProgressCallback
from keras_model.autoencoder.hyperparams import get_constants
from keras_model.autoencoder.model_setup import network_builder, model_builder
from keras_model.autoencoder.util import show_image, show_similar, visualize
from keras_model.util import train
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.neighbors.unsupervised import NearestNeighbors


def denoising_autoencoder(constant_overwrites):
    img_shape, attr, x_train, x_test = load_faces_dataset()
    constants = merge_dict(get_constants(), constant_overwrites)
    constants['img_shape'] = img_shape
    constants['code_size'] = 512
    autoencoder, encoder, decoder = model_builder(network_builder, constants)
    reset_tf_session()
    iterations = 25
    for i in range(iterations):
        print('Epoch %i/%i, Generating corrupted samples...' % (i + 1, iterations))
        x_train_noise = apply_gaussian_noise(x_train)
        x_test_noise = apply_gaussian_noise(x_test)

        # continue to train model with new noise-augmented data
        autoencoder.fit(x=x_train_noise, y=x_train, epochs=1,
                        validation_data=[x_test_noise, x_test],
                        callbacks=[TqdmProgressCallback()],
                        verbose=0)

    x_test_noise = apply_gaussian_noise(x_test)
    denoising_mse = autoencoder.evaluate(x_test_noise, x_test, verbose=0)
    print('Denoising MSE:', denoising_mse)
    for i in range(5):
        img = x_test_noise[i]
        visualize(img, encoder, decoder)


def image_retrieval(constant_overwrites):
    img_shape, attr, x_train, x_test = load_faces_dataset()
    constants = merge_dict(get_constants(), constant_overwrites)
    constants['img_shape'] = img_shape
    encoder_filename = constants['encoder_filename']
    decoder_filename = constants['decoder_filename']
    reset_tf_session()
    autoencoder, encoder, decoder = model_builder(network_builder, constants)
    if os.path.exists(encoder_filename) and not constants['retrain']:
        encoder.load_weights(encoder_filename)
    else:
        data = {
            'X_train': x_train,
            'X_test': x_test
        }
        train(autoencoder, data, constants)
        encoder.save_weights(encoder_filename)
        decoder.save_weights(decoder_filename)

    images = x_train
    codes = encoder.predict(images)
    assert len(codes) == len(images)
    nei_clf = NearestNeighbors(metric="euclidean")
    nei_clf.fit(codes)

    # Cherry-picked examples:

    # smiles
    show_similar(x_test[247], nei_clf, encoder, images)

    # ethnicity
    show_similar(x_test[56], nei_clf, encoder, images)

    # glasses
    show_similar(x_test[63], nei_clf, encoder, images)


def image_morphing(constant_overwrites):
    """
    We can take linear combinations of image codes to produce new images with decoder.

    :param constant_overwrites:
    :return:
    """
    img_shape, attr, x_train, x_test = load_faces_dataset()
    constants = merge_dict(get_constants(), constant_overwrites)
    constants['img_shape'] = img_shape
    encoder_filename = constants['encoder_filename']
    decoder_filename = constants['decoder_filename']
    files = [encoder_filename, decoder_filename]
    reset_tf_session()
    autoencoder, encoder, decoder = model_builder(network_builder, constants)
    if all([os.path.exists(f) for f in files]) and not constants['retrain']:
        encoder.load_weights(encoder_filename)
        decoder.load_weights(decoder_filename)
    else:
        data = {
            'X_train': x_train,
            'X_test': x_test
        }
        train(autoencoder, data, constants)
        encoder.save_weights(encoder_filename)
        decoder.save_weights(decoder_filename)

    for _ in range(5):
        image1, image2 = x_test[np.random.randint(0, len(x_test), size=2)]
        code1, code2 = encoder.predict(np.stack([image1, image2]))
        plt.figure(figsize=[10, 4])
        for i, a in enumerate(np.linspace(0, 1, num=7)):
            output_code = code1 * (1 - a) + code2 * a
            output_image = decoder.predict(output_code[None])[0]

            plt.subplot(1, 7, i + 1)
            show_image(output_image)
            plt.title('a=%.2f' % a)

        plt.show()


def run(constant_overwrites):
    img_shape, attr, x_train, x_test = load_faces_dataset()

    plt.figure(figsize=[6, 6])
    plt.title('Sample images')
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        show_image(x_train[i])

    print('X shape:', x_train.shape)
    print('attr shape:', attr.shape)

    constants = merge_dict(get_constants(), constant_overwrites)
    constants['img_shape'] = img_shape
    model_filename = constants['model_filename']
    encoder_filename = constants['encoder_filename']
    decoder_filename = constants['decoder_filename']
    # files = [model_filename, encoder_filename, decoder_filename]
    reset_tf_session()
    autoencoder, encoder, decoder = model_builder(network_builder, constants)
    # if all([os.path.exists(f) for f in files]) and not constants['retrain']:
    if os.path.exists(model_filename) and not constants['retrain']:
        autoencoder = load_model(model_filename.format(constants['n_epochs']))
        # encoder.load_weights(encoder_filename)
        # decoder.load_weights(decoder_filename)
        encoder = autoencoder.layers[1]
        decoder = autoencoder.layers[2]
    else:
        data = {
            'X_train': x_train,
            'X_test': x_test
        }
        train(autoencoder, data, constants)
        encoder.save_weights(encoder_filename)
        decoder.save_weights(decoder_filename)

    reconstruction_mse = autoencoder.evaluate(x_test, x_test, verbose=0)
    print('Convolutional autoencoder MSE:', reconstruction_mse)
    for i in range(5):
        img = x_test[i]
        visualize(img, encoder, decoder)

    print(autoencoder.evaluate(x_test, x_test, verbose=0))
    print(reconstruction_mse)


if __name__ == "__main__":
    model_type_choices = ['default', 'denoising', 'morphing', 'retrieval']
    # read args
    parser = ArgumentParser(description='Run Keras Autoencoder')
    parser.add_argument('--epochs', dest='n_epochs', type=int, help='number epochs')
    parser.add_argument('--model-filename', dest='model_filename', help='model filename')
    # noinspection SpellCheckingInspection
    parser.add_argument('--ncoder-filename', dest='encoder_filename', help='encoder filename')
    parser.add_argument('--decoder-filename', dest='decoder_filename', help='decoder filename')
    parser.add_argument('--retrain', dest='retrain', help='retrain flag', action='store_true')
    parser.add_argument('--type', dest='model_type', help='model_type', choices=model_type_choices)
    parser.set_defaults(retrain=False)
    args = parser.parse_args()
    model_type = args.model_type
    if model_type == 'denoising':
        denoising_autoencoder(vars(args))
    elif model_type == 'morphing':
        image_morphing(vars(args))
    elif model_type == 'retrieval':
        image_retrieval(vars(args))
    else:
        run(vars(args))
