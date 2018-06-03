import tensorflow as tf
from tf_model.im2latex.model.positional import add_timing_signal_nd


class Encoder(object):
    """
    Applies convolutions to an image
    """

    def __init__(self, config):
        self._config = config

    def __call__(self, training, imgs, dropout):
        """
        Applies convolutions to the image

        :param training: (tf.placeholder) tf.bool
        :param imgs: batch of img, shape (?, height, width, channels) of type tf.uint8
        :param dropout:
        :return: the encoded images, shape (?, h', w', c')
        """
        imgs = tf.cast(imgs, tf.float32) / 255.

        with tf.variable_scope('convolutional_encoder'):
            # conv + max pool -> /2
            out = tf.layers.conv2d(imgs, 64, 3, 1, 'SAME', activation=tf.nn.relu)
            out = tf.layers.max_pooling2d(out, 2, 2, 'SAME')

            # conv + max pool -> /2
            out = tf.layers.conv2d(out, 128, 3, 1, 'SAME', activation=tf.nn.relu)
            out = tf.layers.max_pooling2d(out, 2, 2, 'SAME')

            # regular conv -> id
            out = tf.layers.conv2d(out, 256, 3, 1, 'SAME', activation=tf.nn.relu)
            out = tf.layers.conv2d(out, 256, 3, 1, 'SAME', activation=tf.nn.relu)

            if self._config.encoder_cnn == 'vanilla':
                out = tf.layers.conv2d(out, 512, 3, 1, 'SAME', activation=tf.nn.relu)
            elif self._config.encoder_cnn == 'cnn':
                # conv with stride /2 (replaces the 2 max pool)
                out = tf.layers.conv2d(out, 512, (2, 4), 2, 'SAME')

            # conv
            out = tf.layers.conv2d(out, 512, 3, 1, 'VALID', activation=tf.nn.relu)

            if self._config.positional_embeddings:
                # from tensor2tensor lib - positional embeddings
                out = add_timing_signal_nd(out)

        return out
