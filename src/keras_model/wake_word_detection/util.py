import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
import os
import pyaudio
from pydub import AudioSegment
from scipy.io import wavfile


DATA_DIR = '../../../data/wake_word_detection/'

CHIME_FILE = DATA_DIR + 'audio_examples/chime.wav'


def chime_on_activate(filename, predictions, threshold):
    audio_clip = AudioSegment.from_wav(filename)
    chime = AudioSegment.from_wav(CHIME_FILE)
    ty = predictions.shape[1]
    consecutive_timesteps = 0
    for i in range(ty):
        consecutive_timesteps += 1
        if predictions[0, i, 0] > threshold and consecutive_timesteps > 75:
            # Superpose audio and background using pydub
            audio_clip = audio_clip.overlay(chime, position=(i / ty) * audio_clip.duration_seconds * 1000)
            consecutive_timesteps = 0

    audio_clip.export('chime_output.wav', format='wav')


# noinspection PyUnusedLocal
def create_training_example(ty, background, activates, negatives):
    """
    Creates a training example with a given background, activates, and negatives.

    :param ty:
    :param background: a 10 second background audio recording
    :param activates: a list of audio segments with the wake-word
    :param negatives: a list of audio segments of random words that are not the wake-word
    :return: x - the spectrogram of the training example,
             y - the label at each time step of the spectrogram
    """
    # set the random seed
    np.random.seed(18)

    # make the background less noisy
    background = background - 20

    # initialize y (label vector) with zeros
    y = np.zeros((1, ty))

    # initialize segment times as empty list
    previous_segments = []

    # select 0-4 random "activate" audio clips from the entire list of "activates" recordings
    n_activates = np.random.randint(0, 5)
    sample_indices = np.random.randint(len(activates), size=n_activates)
    sample_activates = [activates[i] for i in sample_indices]

    # loop over randomly selected "activate" clips and insert in background
    for activate in sample_activates:
        # insert the audio clip onto the background
        background, segment_time = insert_audio_clip(background, activate, previous_segments)

        # retrieve segment_start and segment_end from segment_time
        segment_start, segment_end = segment_time

        # insert labels in y
        y = insert_ones(y, segment_end)

    # select 0-2 random "negative" audio recordings from the entire list of "negatives" recordings
    n_negatives = np.random.randint(0, 3)
    sample_indices = np.random.randint(len(negatives), size=n_negatives)
    sample_negatives = [negatives[i] for i in sample_indices]

    # loop over randomly selected "negative" clips and insert in background
    for negative in sample_negatives:
        # insert the audio clip onto the background
        background, _ = insert_audio_clip(background, negative, previous_segments)

    # standardize the volume of the audio clip
    background = match_target_amplitude(background, -20.0)

    # export new training example
    file_handle = background.export('train.wav', format('wav'))
    print('File (train.wav) was saved')

    # get and plot spectrogram of the new recording (background with
    # superposition of positive and negatives)
    x = graph_spectrogram('train.wav')

    return x, y


def detect_wake_word(model, x):
    """
    Predict the location of the wake-word.

    :param model:
    :param x: spectrum of shape (freqs, Tx) ~ (number frequencies, number time steps)
    :return: flattened numpy array to shape (number of output time steps)
    """
    # the spectrogram outputs (freqs, Tx) and we want (Tx, freqs) to input into the model
    x = x.swapaxes(0, 1)
    x = np.expand_dims(x, axis=0)

    return model.predict(x).reshape(-1)


def detect_wake_word_from_file(model, filename):
    """ runs audio (saved in a wav file) through the network """
    x = graph_spectrogram(filename)

    # the spectrogram outputs (freqs, Tx) and we want (Tx, freqs) to input into the model
    x = x.swapaxes(0, 1)
    x = np.expand_dims(x, axis=0)

    return model.predict(x)


def get_audio_input_stream(fs, chunk_samples, callback):
    return pyaudio.PyAudio().open(
        format=pyaudio.paInt16,
        channels=1,
        rate=fs,
        input=True,
        frames_per_buffer=chunk_samples,
        input_device_index=0,
        stream_callback=callback
    )


def get_random_time_segment(segment_ms):
    """
    Gets a random time segment of duration segment_ms in a 10,000 ms audio clip,
    onto which we can insert an audio clip of duration `segment_ms`.

    :param segment_ms: the duration of the audio clip in ms
    :return: a random segment_time, a tuple of (segment_start, segment_end) in ms
    """
    # Make sure segment doesn't run past the 10sec background
    segment_start = np.random.randint(low=0, high=10000-segment_ms)
    segment_end = segment_start + segment_ms - 1

    return segment_start, segment_end


# noinspection SpellCheckingInspection
def get_spectrogram(data):
    """
    Compute a spectrogram

    :param data: one channel / dual channel audio data as numpy array
    :return: spectrogram, 2D array, columns are the periodograms of successive segments
    """
    nfft = 200  # length of each window segment
    fs = 8000  # sample frequency
    n_overlap = 120  # overlap between windows
    n_channels = data.ndim
    pxx = None
    if n_channels == 1:
        pxx, _, _ = mlab.specgram(data, nfft, fs, noverlap=n_overlap)
    elif n_channels == 2:
        pxx, _, _ = mlab.specgram(data[:, 0], nfft, fs, noverlap=n_overlap)

    return pxx


# noinspection SpellCheckingInspection
def graph_spectrogram(wav_file):
    """ Calculate and plot spectrogram for a wav audio file """
    rate, data = load_wave_file(wav_file)
    nfft = 200  # length of each window segment
    fs = 8000  # sample frequency
    n_overlap = 120  # overlap between windows
    n_channels = data.ndim
    pxx = None
    if n_channels == 1:
        pxx, freqs, bins, im = plt.specgram(data, nfft, fs, noverlap=n_overlap)
    elif n_channels == 2:
        pxx, freqs, bins, im = plt.specgram(data[:, 0], nfft, fs, noverlap=n_overlap)

    return pxx


def has_wake_word(predictions, chunk_duration, feed_duration, threshold=0.5):
    """
    Detect wake-word in the chunk of input audio.

    It is looking for the rising edge of the predictions data belonging to the latest chunk.

    :param predictions: predicted labels from the model
    :param chunk_duration: time in seconds of a chunk
    :param feed_duration: time in seconds of the input to the model
    :param threshold: threshold for probability to be considered positive
    :return: True if wake-word is detected
    """
    predictions = predictions > threshold
    chunk_predictions_samples = int(len(predictions) * chunk_duration / feed_duration)
    chunk_predictions = predictions[-chunk_predictions_samples:]
    level = chunk_predictions[0]
    for pred in chunk_predictions:
        if pred > level:
            return True
        else:
            level = pred

    return False


def insert_audio_clip(background, audio_clip, previous_segments):
    """
    Insert a new audio segment over the background noise at a random time step,
    ensuring that the audio segment does not overlap with existing segments.

    :param background: a 10 second background audio recording
    :param audio_clip: the audio clip to be inserted/overlaid
    :param previous_segments: times where audio segments have already been placed
    :return: new_background, the updated background audio
    """
    # Get the duration of the audio clip in ms.
    segment_ms = len(audio_clip)

    # Pick a random time segment onto which to insert the new audio clip.
    segment_time = get_random_time_segment(segment_ms)

    # Check if the new segment_time overlaps with one of the previous_segments.
    # If so, keep picking new segment_time's at random until it doesn't overlap.
    while is_overlapping(segment_time, previous_segments):
        segment_time = get_random_time_segment(segment_ms)

    # Add the new segment_time to the list of previous_segments
    previous_segments.append(segment_time)

    # Superpose audio segment and background
    new_background = background.overlay(audio_clip, position=segment_time[0])

    return new_background, segment_time


def insert_ones(y, segment_end_ms):
    """
    Update the label vector y. The labels of the 50 output steps strictly after the end of the segment
    should be set to 1. By strictly, we mean that the label of segment_end_y should be 0 while the
    following 50 labels should be ones.

    y is a (1,1375) dimensional vector, since Ty=1375

    If the wake-word ended at time step t, then set y⟨t+1⟩ = 1 as well as for up to 49 additional
    consecutive values. However, make sure you don't run off the end of the array and try to update
    y[0][1375], since the valid indices are y[0][0] through y[0][1374] because Ty=1375. So if the
    wake-word ends at step 1370, you would get only y[0][1371] = y[0][1372] = y[0][1373] = y[0][1374] = 1

    Warning! has side-effects - mutates y

    :param y: numpy array of shape (1, Ty), the labels of the training example
    :param segment_end_ms: the end time of the segment in ms
    :return: y, updated labels
    """
    ty = y.shape[1]

    # duration of the background (in terms of spectrogram time-steps)
    segment_end_y = int(segment_end_ms * ty / 10000.0)

    # add 1 to the correct index in the background label (y)
    for i in range(segment_end_y + 1, segment_end_y + 51):
        if i < ty:
            y[0, i] = 1

    return y


def is_overlapping(segment_time, previous_segments):
    """
    Checks if the time of a segment overlaps with the times of existing segments.

    There is overlap if the segment starts before the previous segment ends, and
    the segment ends after the previous segment starts.

    :param segment_time: a tuple of (segment_start, segment_end) for the new segment
    :param previous_segments: a list of tuples of (segment_start, segment_end) for
                              the existing segments
    :return: True if the time segment overlaps with any of the existing segments,
             False otherwise
    """
    segment_start, segment_end = segment_time
    overlap = False

    # Loop over the previous_segments start and end times.
    # Compare start/end times and set the flag to True if there is an overlap
    for previous_start, previous_end in previous_segments:
        if segment_start <= previous_end and segment_end >= previous_start:
            overlap = True

    return overlap


def load_raw_audio():
    """ Load raw audio files for speech synthesis """
    activates, negatives, backgrounds = [], [], []
    for filename in os.listdir(DATA_DIR + 'raw_data/activates'):
        if filename.endswith('wav'):
            activate = AudioSegment.from_wav(DATA_DIR + 'raw_data/activates/' + filename)
            activates.append(activate)

    for filename in os.listdir(DATA_DIR + 'raw_data/negatives'):
        if filename.endswith('wav'):
            negative = AudioSegment.from_wav(DATA_DIR + 'raw_data/negatives/' + filename)
            negatives.append(negative)

    for filename in os.listdir(DATA_DIR + 'raw_data/backgrounds'):
        if filename.endswith('wav'):
            background = AudioSegment.from_wav(DATA_DIR + 'raw_data/backgrounds/' + filename)
            backgrounds.append(background)

    return activates, negatives, backgrounds


def load_wave_file(wav_file):
    """ Load a wav file """
    rate, data = wavfile.read(wav_file)
    return rate, data


# noinspection PyPep8Naming
def match_target_amplitude(sound, target_sBFS):
    change_in_dBFS = target_sBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)


def plot_probability(predictions):
    plt.plot(predictions[0, :, 0])
    plt.ylabel('probability')
    plt.show()


# noinspection SpellCheckingInspection
def plot_spectrogram(data):
    """
    Compute and plot a spectrogram

    :param data: one channel / dual channel audio data as numpy array
    :return: spectrogram, 2D array, columns are the periodograms of successive segments
    """
    nfft = 200  # length of each window segment
    fs = 8000  # sample frequency
    n_overlap = 120  # overlap between windows
    n_channels = data.ndim
    pxx = None
    if n_channels == 1:
        pxx, _, _, _ = plt.specgram(data, nfft, fs, noverlap=n_overlap)
    elif n_channels == 2:
        pxx, _, _, _ = plt.specgram(data[:, 0], nfft, fs, noverlap=n_overlap)

    plt.show()

    return pxx


def plot_spectrogram_and_probs_from_file(model, wav_file):
    plt.subplot(2, 1, 1)
    x = graph_spectrogram(wav_file)

    # the spectrogram outputs (freqs, Tx) and we want (Tx, freqs) to input into the model
    x = x.swapaxes(0, 1)
    x = np.expand_dims(x, axis=0)

    predictions = model.predict(x)

    plt.subplot(2, 1, 2)
    plt.plot(predictions[0, :, 0])
    plt.ylabel('probability')

    plt.show()


def preprocess_audio(filename):
    """ Preprocess the audio to the correct format """
    # trim or pad audio segment to 10,000ms
    padding = AudioSegment.silent(duration=10000)
    segment = AudioSegment.from_wav(filename)[:10000]
    segment = padding.overlay(segment)

    # set frame rate to 44,100
    segment = segment.set_frame_rate(44100)

    # export as wav
    segment.export(filename, format='wav')
