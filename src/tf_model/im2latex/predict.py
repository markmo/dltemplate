from scipy.misc import imread
from tf_model.im2latex.model.img2seq import Img2SeqModel
from tf_model.im2latex.utils.general import Config, run
from tf_model.im2latex.utils.image import crop_image, downsample_image, greyscale, pad_image, TIMEOUT
from tf_model.im2latex.utils.text import Vocab


def interactive_shell(model):
    """
    Creates interactive shell to play with model

    :param model:
    :return:
    """
    model.logger.info("""
This is an interactive shell. To exit, enter 'exit'.
Enter a path to a file
input> data/images_test/0.png""")

    while True:
        img_path = input('input> ')
        if img_path == 'exit':
            break

        if img_path[-3:] == 'png':
            img = imread(img_path)
        elif img_path[-3:] == 'pdf':
            # call magick to convert the pdf into a png file
            buckets = [
                [240, 100], [320, 80], [400, 80], [400, 100], [480, 80], [480, 100],
                [560, 80], [560, 100], [640, 80], [640, 100], [720, 80], [720, 100],
                [720, 120], [720, 200], [800, 100], [800, 320], [1000, 200],
                [1000, 400], [1200, 200], [1600, 200], [1600, 1600]
            ]
            dir_output = 'tmp/'
            name = img_path.split('/')[-1].split('.')[0]
            run('magick convert -density {} -quality {} {} {}'.format(
                200, 100, img_path, dir_output + '{}.png'.format(name)), TIMEOUT)
            img_path = dir_output + '{}.png'.format(name)
            crop_image(img_path, img_path)
            pad_image(img_path, img_path, buckets=buckets)
            downsample_image(img_path, img_path, ratio=2)
            img = imread(img_path)
        else:
            raise NotImplementedError('Unsupported file type {}'.format(img_path[-3:]))

        img = greyscale(img)
        hyps = model.predict(img)
        model.logger.info(hyps[0])


if __name__ == '__main__':
    dir_out = 'results/small/'
    conf_vocab = Config(dir_out + 'vocab.yml')
    conf_model = Config(dir_out + 'model.yml')
    vocab = Vocab(conf_vocab)
    model_ = Img2SeqModel(conf_model, dir_out, vocab)
    model_.build_pred()
    model_.restore_session(dir_out + 'model.weights/')
    interactive_shell(model_)
