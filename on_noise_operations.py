import numpy as np


def extract_noise(data, rate, duration=2):
    """ Extracts the first 'duration' seconds of audio data as noise. """
    return data[:int(rate * duration)]


def extend_noise(noise, total_length):
    """ Repeat the noise sample to match the desired total length of the audio data. """
    if noise.ndim == 1:  # Mono
        repeat_count = int(np.ceil(total_length / len(noise)))
        extended_noise = np.tile(noise, repeat_count)[:total_length]
    else:  # Stereo
        repeat_count = int(np.ceil(total_length / noise.shape[0]))
        extended_noise = np.tile(noise, (repeat_count, 1))[:total_length, :]
    return extended_noise


def remove_noise(data, noise):
    # Subtract the noise from the original data
    # Assuming both data and noise are aligned in length
    return data - noise