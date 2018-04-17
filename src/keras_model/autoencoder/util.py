import matplotlib.pyplot as plt
import numpy as np


def get_similar(image, nei_clf, encoder, images, n_neighbors=5):
    assert image.ndim == 3, 'image must be [batch,height,width,3]'
    code = encoder.predict(image[None])
    (distances,), (idx,) = nei_clf.kneighbors(code, n_neighbors=n_neighbors)
    return distances, images[idx]


def show_image(x):
    plt.imshow(np.clip(x + 0.5, 0, 1))


def show_similar(image, nei_clf, encoder, images):
    distances, neighbours = get_similar(image, nei_clf, encoder, images, n_neighbors=3)
    plt.figure(figsize=[8, 7])
    plt.subplot(1, 4, 1)
    show_image(image)
    plt.title('Original image')
    for i in range(3):
        plt.subplot(1, 4, i + 2)
        show_image(neighbours[i])
        plt.title('Dist=%.3f' % distances[i])

    plt.show()


def visualize(img, encoder, decoder):
    """
    Draws original, encoded and decoded images

    :param img:
    :param encoder:
    :param decoder:
    :return:
    """
    code = encoder.predict(img[None])[0]  # img[None] is the same as img[np.newaxis, :]
    reconstruction = decoder.predict(code[None])[0]

    plt.subplot(1, 3, 1)
    plt.title('Original')
    show_image(img)

    plt.subplot(1, 3, 2)
    plt.title('Code')
    plt.imshow(code.reshape([code.shape[-1] // 2, -1]))

    plt.subplot(1, 3, 3)
    plt.title('Reconstructed')
    show_image(reconstruction)
    plt.show()
