from multiprocessing import Pool
import numpy as np
from PIL import Image
from tf_model.im2latex.utils.general import delete_file, get_files, init_dir, run


TIMEOUT = 10


def get_max_shape(arrays):
    """

    :param arrays: list of arrays
    :return:
    """
    shapes = [list(x.shape) for x in arrays]
    ndim = len(arrays[0].shape)
    max_shape = []
    for d in range(ndim):
        max_shape += [max(shapes, key=lambda x: x[d])[d]]

    return max_shape


def pad_batch_images(images, max_shape=None):
    """

    :param images: list of arrays
    :param max_shape:
    :return:
    """
    # 1. max shape
    if max_shape is None:
        max_shape = get_max_shape(images)

    # 2. apply formatting
    batch_images = 255 * np.ones([len(images)] + list(max_shape))
    for idx, img in enumerate(images):
        batch_images[idx, :img.shape[0], :img.shape[1]] = img

    return batch_images.astype(np.uint8)


def greyscale(state):
    """
    Preprocess state (:, :, 3) image into greyscale

    :param state:
    :return:
    """
    state = state[:, :, 0] * 0.299 + state[:, :, 1] * 0.587 + state[:, :, 2] * 0.114
    state = state[:, :, np.newaxis]
    return state.astype(np.uint8)


def downsample(state):
    """
    Downsamples an image on the first 2 dimensions

    :param state: (np array) with 3 dimensions
    :return:
    """
    return state[::2, ::2, :]


def pad_image(img, output_path, pad_size=None, buckets=None):
    """
    Pads image with pad size and with buckets

    :param img: (string) path to image
    :param output_path: (string) path to output image
    :param pad_size: list of 4 ints
    :param buckets: ascending ordered list of sizes [(width, height), ...]
    :return:
    """
    if pad_size is None:
        pad_size = [8, 8, 8, 8]

    top, left, bottom, right = pad_size
    old_img = Image.open(img)
    old_size = (old_img.size[0] + left + right, old_img.size[1] + top + bottom)
    new_size = get_new_size(old_size, buckets)
    new_img = Image.new('RGB', new_size, (255, 255, 255))
    new_img.paste(old_img, (left, top))
    new_img.save(output_path)


def get_new_size(old_size, buckets):
    """
    Computes new size from buckets

    :param old_size: (width, height)
    :param buckets: list of sizes
    :return: new_size: original size or first bucket in iter order
             that matches the size
    """
    if buckets is None:
        return old_size

    w, h = old_size
    for (w_b, h_b) in buckets:
        if w_b >= w and h_b >= h:
            return w_b, h_b

    return old_size


def crop_image(img, output_path):
    """
    Crops image to content

    :param img: (string) path to image
    :param output_path: (string) path to output image
    :return:
    """
    old_img = Image.open(img).convert('L')
    img_data = np.asarray(old_img, dtype=np.uint8)  # height, width
    nnz_inds = np.where(img_data != 255)
    if len(nnz_inds[0]) == 0:
        old_img.save(output_path)
        return False

    y_min = np.min(nnz_inds[0])
    y_max = np.max(nnz_inds[0])
    x_min = np.min(nnz_inds[1])
    x_max = np.max(nnz_inds[1])
    old_img = old_img.crop((x_min, y_min, x_max + 1, y_max + 1))
    old_img.save(output_path)
    return True


def downsample_image(img, output_path, ratio=2):
    """
    Downsample image by ratio

    :param img:
    :param output_path:
    :param ratio:
    :return:
    """
    assert ratio >= 1, 'Ratio {} is less than 1'.format(ratio)
    if ratio == 1:
        return True

    old_img = Image.open(img)
    old_size = old_img.size
    new_size = (int(old_size[0] / ratio), int(old_size[1] / ratio))
    new_img = old_img.resize(new_size, Image.LANCZOS)
    new_img.save(output_path)
    return True


def convert_to_png(formula, dir_output, name, quality=100, density=200, down_ratio=2, buckets=None):
    """
    Converts LaTex to png image

    :param formula: (string) of latex
    :param dir_output: (string) path to output directory
    :param name: (string) name of file
    :param quality:
    :param density:
    :param down_ratio: (int) downsampling ratio
    :param buckets: list of tuples (list of sizes) to produce similar shape images
    :return:
    """
    # write formula into a .tex file
    with open(dir_output + '{}.tex'.format(name), 'w') as f:
        # noinspection SpellCheckingInspection
        f.write(r"""
                \documentclass[preview]{standalone}
                \begin{document}
                   $$ %s $$
                \end{document}""" % formula)

    # call pdflatex to create pdf
    # noinspection SpellCheckingInspection
    run('pdflatex -interaction=nonstopmode -output-directory={} {}'.format(
        dir_output, dir_output + '{}.tex'.format(name)), TIMEOUT)

    # call magick to convert the pdf into a png file
    run('magick convert -density {} -quality {} {} {}'.format(
        density, quality, dir_output + '{}.pdf'.format(name), dir_output + '{}.png'.format(name)), TIMEOUT)

    # cropping and downsampling
    img_path = dir_output + '{}.png'.format(name)
    try:
        crop_image(img_path, img_path)
        pad_image(img_path, img_path, buckets=buckets)
        downsample_image(img_path, img_path, down_ratio)
        clean(dir_output, name)
        return '{}.png'.format(name)
    except Exception as e:
        print(e)
        clean(dir_output, name)
        return False


def clean(dir_output, name):
    delete_file(dir_output + '{}.aux'.format(name))
    delete_file(dir_output + '{}.log'.format(name))
    delete_file(dir_output + '{}.pdf'.format(name))
    delete_file(dir_output + '{}.tex'.format(name))


def build_image(item):
    idx, form, dir_images, quality, density, down_ratio, buckets = item
    name = str(idx)
    path_img = convert_to_png(form, dir_images, name, quality, density, down_ratio, buckets)
    return path_img, idx


def build_images(formulas, dir_images, quality=100, density=200, down_ratio=2, buckets=None, n_threads=4):
    """
    Parallel procedure to produce images from formulas

    :param formulas: (dict) idx -> string
    :param dir_images:
    :param quality:
    :param density:
    :param down_ratio:
    :param buckets:
    :param n_threads:
    :return: list of (path_img, idx). If an exception is raised during
             image generation, then path_img = False
    """
    init_dir(dir_images)
    existing_idxs = sorted(set([int(filename.split('.')[0])
                                for filename in get_files(dir_images)
                                if filename.split('.')[-1] == 'png']))
    pool = Pool(n_threads)
    result = pool.map(build_image, [(idx, form, dir_images, quality, density, down_ratio, buckets)
                                    for idx, form in formulas.items()
                                    if idx not in existing_idxs])
    pool.close()
    pool.join()
    result += [(str(idx) + '.png', idx) for idx in existing_idxs]
    return result
