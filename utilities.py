import numpy as np
from scipy.fft import fft


def get_volume(audio):
    if audio.size == 0:
        return 0
    rms_value = np.sqrt(np.mean(audio ** 2))
    if np.isnan(rms_value):
        return -np.inf
    epsilon = 1e-10
    reference = 1.0
    return 20 * np.log10(max(rms_value, epsilon) / reference)


def find_loudest_frequency(audio, fs, min_freq, max_freq):
    fft_values = fft(audio)
    mag_fft = np.abs(fft_values)

    freqs = np.fft.fftfreq(len(audio), 1 / fs)
    indices = np.where((freqs >= min_freq) & (freqs <= max_freq))

    loudest_freq = freqs[indices][np.argmax(mag_fft[indices])]
    return loudest_freq
