import matplotlib.pyplot as plt
import numpy as np


def make_trainable(net, is_trainable):
    net.trainable = is_trainable
    for layer in net.layers:
        layer.trainable = is_trainable


def train(generator, discriminator, adversarial_model, data, constants):
    d_metrics = []
    a_metrics = []

    running_d_loss = 0
    running_d_accuracy = 0
    running_a_loss = 0
    running_a_accuracy = 0

    batch_size = constants['batch_size']

    print('batch_size:', batch_size)
    print('n_epochs:', constants['n_epochs'])
    print('n_report_steps:', constants['n_report_steps'])

    for i in range(constants['n_epochs']):
        if (i + 1) % 10 == 0:
            print('epoch #{}'.format(i + 1))

        real_images = np.reshape(data[np.random.choice(data.shape[0], batch_size, replace=False)],
                                 (batch_size, 28, 28, 1))
        fake_images = generator.predict(np.random.uniform(-1.0, 1.0, size=[batch_size, 100]))

        x = np.concatenate((real_images, fake_images))
        y = np.ones([2 * batch_size, 1])
        y[batch_size:, :] = 0

        make_trainable(discriminator, is_trainable=True)

        d_metrics.append(discriminator.train_on_batch(x, y))
        running_d_loss += d_metrics[-1][0]
        running_d_accuracy += d_metrics[-1][1]

        make_trainable(discriminator, is_trainable=False)

        noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
        y = np.ones([batch_size, 1])

        a_metrics.append(adversarial_model.train_on_batch(noise, y))
        running_a_loss += a_metrics[-1][0]
        running_a_accuracy += a_metrics[-1][1]

        if (i + 1) % constants['n_report_steps'] == 0:
            print('Epoch #{}'.format(i + 1))
            log_msg = '%d: [D loss: %f, acc: %f]' % (i + 1, running_d_loss / i, running_d_accuracy / i)
            log_msg = '%s: [A loss: %f, acc: %f]' % (log_msg, running_a_loss / i, running_a_accuracy / i)
            print(log_msg)

            noise = np.random.uniform(-1.0, 1.0, size=[16, 100])
            gen_imgs = generator.predict(noise)

            plt.figure(figsize=(5, 5))

            for k in range(gen_imgs.shape[0]):
                plt.subplot(4, 4, k + 1)
                plt.imshow(gen_imgs[k, :, :, 0], cmap='gray')
                plt.axis('off')

            plt.tight_layout()
            plt.show()

    return a_metrics, d_metrics
