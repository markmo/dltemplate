from argparse import ArgumentParser
from common.load_data import load_quickdraw_dataset
from common.util import merge_dict
from keras_model.gan.hyperparams import get_constants
from keras_model.gan.model_setup import adversarial_builder, generator_builder, get_discriminator
from keras_model.gan.util import train
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


DATA_DIR = os.path.expanduser('~/src/DeepLearning/dltemplate/data/quickdraw/')


def run(constant_overwrites):
    # if using a GPU
    # os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    # # os.environ['CUDA_VISIBLE_DEVICES'] = ''
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    constants = merge_dict(get_constants(), constant_overwrites)
    file_path = load_quickdraw_dataset('apple', DATA_DIR)
    data = np.load(file_path)  # 28x28 grayscale bitmap in numpy format, images are centered

    print('')
    print('data.shape:', data.shape)

    # normalize and reshape
    data = data / 255
    data = np.reshape(data, (data.shape[0], 28, 28, 1))
    img_w, img_h = data.shape[1:3]

    # Sample image
    random_idx = np.random.randint(data.shape[0])
    plt.imshow(data[random_idx, :, :, 0], cmap='Greys')
    plt.show()

    generator = generator_builder(constants)
    discriminator = get_discriminator(img_w, img_h, constants)
    adversarial_model = adversarial_builder(generator, discriminator, constants)

    a_metrics_complete, d_metrics_complete = train(generator, discriminator, adversarial_model, data, constants)

    df0 = pd.DataFrame({
        'Generative Loss': [metric[0] for metric in a_metrics_complete],
        'Discriminative Loss': [metric[0] for metric in d_metrics_complete],
    })
    ax0 = df0.plot(title='Training Loss', logy=True)
    ax0.set_xlabel('Epochs')
    ax0.set_ylabel('Loss')

    df1 = pd.DataFrame({
        'Generative Loss': [metric[1] for metric in a_metrics_complete],
        'Discriminative Loss': [metric[1] for metric in d_metrics_complete],
    })
    ax1 = df1.plot(title='Training Accuracy')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')


if __name__ == '__main__':
    # read args
    parser = ArgumentParser(description='Run Keras GAN')
    parser.add_argument('--epochs', dest='n_epochs', type=int, help='number epochs')
    parser.add_argument('--learning-rate', dest='learning_rate', type=float, help='learning_rate')
    parser.add_argument('--model-filename', dest='model_filename', help='model filename')
    parser.add_argument('--retrain', dest='retrain', help='retrain flag', action='store_true')
    parser.set_defaults(retrain=False)
    args = parser.parse_args()
    run(vars(args))
