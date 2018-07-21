from argparse import ArgumentParser
from common.util import load_hyperparams, merge_dict
from keras.models import load_model
from keras_model.wake_word_detection.model_setup import build_model, fit
from keras_model.wake_word_detection.util import chime_on_activate, DATA_DIR, detect_wake_word_from_file
from keras_model.wake_word_detection.util import load_raw_audio, plot_probability, preprocess_audio
from keras_model.wake_word_detection.util import detect_wake_word, has_wake_word
from keras_model.wake_word_detection.util import get_audio_input_stream, get_spectrogram
from keras_model.wake_word_detection.util import plot_spectrogram_and_probs_from_file, plot_spectrogram
import numpy as np
import os
import pyaudio
from pydub import AudioSegment
from pydub.playback import play
from queue import Queue
import sys
import time


def run(constant_overwrites):
    config_path = os.path.join(os.path.dirname(__file__), 'hyperparams.yml')
    constants = merge_dict(load_hyperparams(config_path), constant_overwrites)
    activates, negatives, backgrounds = load_raw_audio()

    # Should be 10,000, since it is a 10 sec clip
    print('background len:', str(len(backgrounds[0])))

    # Maybe around 1000, since an "activate" audio clip is usually around 1 sec (but varies a lot)
    print('activate[0] len:', str(len(activates[0])))

    # Different "activate" clips can have different lengths
    print('activate[1] len:', str(len(activates[1])))

    x = np.load(DATA_DIR + 'XY_train/X.npy')
    y = np.load(DATA_DIR + 'XY_train/Y.npy')
    x_dev = np.load(DATA_DIR + 'XY_dev/X_dev.npy')
    y_dev = np.load(DATA_DIR + 'XY_dev/Y_dev.npy')

    if constants['retrain']:
        # number of time steps input to the model from the spectrogram
        # use tx=1101, ty=272 for 2sec audio input
        tx = 5511

        # number of frequencies input to the model at each time step of the spectrogram
        n_freq = 101

        model = build_model(input_shape=(tx, n_freq))
        n_epochs = constants['n_epochs']
        learning_rate = constants['learning_rate']

    else:
        # Load pre-trained model
        # Wake-word detection takes a long time to train. To save time, this model has been
        # trained for about 3 hours on a GPU using the same architecture and a large training
        # set of about 4000 examples.
        model = load_model(DATA_DIR + 'models/tr_model.h5')
        n_epochs = 1
        learning_rate = 0.0001

    batch_size = constants['batch_size']

    fit(x, y, model, epochs=n_epochs, batch_size=batch_size, learning_rate=learning_rate)

    chunk_duration = 0.5  # read length in seconds from mic
    fs = 44100  # sampling rate for mic
    chunk_samples = int(fs * chunk_duration)  # read length in number of samples
    prob_threshold = 0.5

    if constants['live']:
        # Record audio stream from mic

        # model input duration in seconds (int)
        feed_duration = 10
        feed_samples = int(fs * feed_duration)

        assert feed_duration / chunk_duration == int(feed_duration / chunk_duration)

        # Queue to communicate between the audio callback and main thread
        q = Queue()
        listening = True
        silence_threshold = 100

        # Run the demo until timeout
        timeout = time.time() + 30  # 30 seconds

        # data buffer for the input waveform
        data = np.zeros(feed_samples, dtype='int16')

        # noinspection PyUnusedLocal
        def callback(data_in, frame_count, time_info, status):
            nonlocal data, listening, silence_threshold, timeout
            if time.time() > timeout:
                listening = False

            data_ = np.frombuffer(data_in, dtype='int16')
            if np.abs(data_).mean() < silence_threshold:
                sys.stdout.write('-')
                return data_in, pyaudio.paContinue
            else:
                sys.stdout.write('.')

            data = np.append(data, data_)
            if len(data) > feed_samples:
                data = data[-feed_samples:]
                q.put(data)

            return data_in, pyaudio.paContinue

        stream = get_audio_input_stream(fs, chunk_samples, callback)
        stream.start_stream()

        print('Listening...')
        try:
            while listening:
                data = q.get()
                spectrum = get_spectrogram(data)
                predictions = detect_wake_word(model, spectrum)
                if has_wake_word(predictions, chunk_duration, feed_duration, prob_threshold):
                    sys.stdout.write('1')

                sys.stdout.flush()

        except (KeyboardInterrupt, SystemExit):
            stream.stop_stream()
            stream.close()
            timeout = time.time()
            # listening = False

        print('\nStopped!')
        stream.stop_stream()
        stream.close()

    elif constants['test_mic']:
        data = None

        # noinspection PyUnusedLocal
        def callback(data_in, frame_count, time_info, status):
            nonlocal data
            data = np.frombuffer(data_in, dtype='int16')
            print('mean:', np.abs(data).mean(), 'max:', np.abs(data).max())
            return data_in, pyaudio.paContinue

        stream = get_audio_input_stream(fs, chunk_samples, callback)
        stream.start_stream()
        time.sleep(10)  # 10 seconds
        stream.stop_stream()
        stream.close()
        plot_spectrogram(data)

    else:
        # Test the model
        loss, acc = model.evaluate(x_dev, y_dev)
        print('Dev set accuracy =', acc)

        chime_threshold = 0.5

        filename = DATA_DIR + 'raw_data/dev/1.wav'
        predictions = detect_wake_word_from_file(model, filename)
        plot_spectrogram_and_probs_from_file(model, filename)
        chime_on_activate(filename, predictions, chime_threshold)
        play(AudioSegment.from_wav('./chime_output.wav'))

        filename = DATA_DIR + 'audio_examples/my_audio.wav'
        preprocess_audio(filename)
        predictions = detect_wake_word_from_file(model, filename)
        plot_spectrogram_and_probs_from_file(model, filename)
        chime_on_activate(filename, predictions, chime_threshold)
        play(AudioSegment.from_wav('./chime_output.wav'))


if __name__ == '__main__':
    # read args
    parser = ArgumentParser(description='Run Wake-word Detection model')
    parser.add_argument('--epochs', dest='n_epochs', type=int, help='number epochs')
    parser.add_argument('--batch-size', dest='batch_size', type=int, help='batch size')
    parser.add_argument('--learning-rate', dest='learning_rate', type=float, help='learning rate')
    parser.add_argument('--retrain', dest='retrain', help='retrain flag', action='store_true')
    parser.add_argument('--live', dest='live', help='live flag', action='store_true')
    parser.add_argument('--test-mic', dest='test_mic', help='test mic flag', action='store_true')
    parser.set_defaults(retrain=False)
    parser.set_defaults(live=False)
    parser.set_defaults(test_mic=False)
    args = parser.parse_args()

    run(vars(args))
