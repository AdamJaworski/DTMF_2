import numpy as np
from scipy.signal import firwin, lfilter, butter
from scipy.signal.windows import hamming, get_window
import scipy.io.wavfile as wav
from scipy.fft import fft, ifft


def apply_fir(audio: np.ndarray, fs: float, frequencies: list, bandwidth: int = 5, n_fir: int = 1025, window: str = "hann") -> np.ndarray:
    """
    Applies FIR filter with window to audio signal

    :param audio: ndarray of audio signal
    :param fs: sampling frequency of audio signal
    :param frequencies: list of frequencies on which filter should be applied
    :param bandwidth: size of FIR filter window
    :param n_fir: numtaps for FIR filter - default value is odd
    :param window: Name of window used in FIR filter - default is Hann

    :return: processed audio as ndarray
    """

    output_audio_list = []
    i=0
    for frequency in frequencies:
        i+=1
        filter_fir = firwin(n_fir, [frequency - bandwidth/2, frequency + bandwidth/2], pass_zero=False, window=window, fs=fs)
        processed_part = lfilter(filter_fir, 1.0, audio)

        wav.write(f'{i}.wav', fs, audio)
        output_audio_list.append(processed_part)

    combined_audio = np.sum(output_audio_list, axis=0)
    max_val = np.max(np.abs(combined_audio))
    filtered_audio = combined_audio / max_val if max_val != 0 else combined_audio

    return filtered_audio


def apply_python_filter(audio: np.ndarray, fs: float, frequencies: list, bandwidth: int = 5):
    """

    :param audio: ndarray of audio signal
    :param fs: sampling frequency of audio signal
    :param frequencies: list of frequencies on which filter should be applied

    :return: processed audio as ndarray
    """
    window = hamming(len(audio))
    audio_ = audio * window

    fft_data = fft(audio_)
    new_fft = np.zeros_like(fft_data)
    freq_bin_width = fs / len(audio)

    i = 0
    for frequency in frequencies:
        i += 1
        lower_freq_bin = int((frequency - bandwidth / 2) / freq_bin_width)
        upper_freq_bin = int((frequency + bandwidth / 2) / freq_bin_width)

        new_fft[lower_freq_bin:upper_freq_bin] = fft_data[lower_freq_bin:upper_freq_bin]
        n = len(audio)
        new_fft[n - upper_freq_bin:n - lower_freq_bin] = fft_data[n - upper_freq_bin:n - lower_freq_bin]

    filtered_audio = ifft(new_fft)
    filtered_audio /= (window + np.finfo(float).eps)
    filtered_audio = np.real(filtered_audio).astype(audio.dtype)
    return filtered_audio


def apply_python_filter2(audio: np.ndarray, fs: float, frequencies: list, bandwidth: int = 5):
    """
    Apply a bandpass filter around each specified frequency with a specified bandwidth.

    :param audio: ndarray of audio signal
    :param fs: sampling frequency of audio signal
    :param frequencies: list of frequencies around which to apply the bandpass filter
    :param bandwidth: bandwidth around each frequency to allow through
    :return: processed audio as ndarray
    """
    fft_data = fft(audio)
    n = len(audio)
    filtered_fft = np.zeros_like(fft_data)

    for freq in frequencies:
        lower_bound = int((freq - bandwidth / 2) * n / fs)
        upper_bound = int((freq + bandwidth / 2) * n / fs)

        # Ensure indices remain within valid range
        lower_bound = max(lower_bound, 0)
        upper_bound = min(upper_bound, n // 2)

        # Create a window to smoothly transition in and out of the bandpass range
        window = get_window('hamming', upper_bound - lower_bound, False)
        window = np.pad(window, (lower_bound, n - upper_bound), 'constant', constant_values=(0, 0))

        # Apply window to both positive and negative frequencies
        filtered_fft[:upper_bound] += window[:upper_bound] * fft_data[:upper_bound]
        filtered_fft[-upper_bound:] += window[:upper_bound][::-1] * fft_data[-upper_bound:]

    # Perform the inverse FFT
    filtered_audio = ifft(filtered_fft)
    filtered_audio = np.real(filtered_audio).astype(audio.dtype)

    return filtered_audio


def butter_bandpass(audio, fs, freq, bandwidth=20, order=4):
    """

    :param audio:
    :param fs:
    :param freq:
    :param bandwidth:
    :param order:
    :return:
    """
    nyq = 0.5 * fs
    low = (freq - bandwidth / 2) / nyq
    high = (freq + bandwidth / 2) / nyq
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, audio)
    return y