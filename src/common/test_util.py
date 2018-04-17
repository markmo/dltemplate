from common.load_data import load_faces_dataset
from common.util import apply_gaussian_noise


def test_applying_noise():
    _, _, x_train, _ = load_faces_dataset()
    sample = x_train[:100]
    theoretical_std = (sample.std()**2 + 0.5**2)**0.5
    our_std = apply_gaussian_noise(sample, sigma=0.5).std()
    assert abs(theoretical_std - our_std) < 0.01, \
        "Standard deviation does not match it's required value. Make sure you use sigma as std."
    assert abs(apply_gaussian_noise(sample, sigma=0.5).mean() - sample.mean()) < 0.01, \
        'Mean has changed. Please add zero-mean noise.'
