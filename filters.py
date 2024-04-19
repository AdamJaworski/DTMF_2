import numpy as np
from scipy.signal import firwin


def apply_fir(audio: np.ndarray, fs: float, frequencies: list, bandwidth: int = 5, n_fir: int = 99, window: str = "hann") -> np.ndarray:
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
    for frequency in frequencies:
        filter = firwin(n_fir, [frequency - bandwidth/2, frequency + bandwidth/2], pass_zero=False, window=window, fs=fs)
        processed_part = np.convolve(audio, filter, mode="same")
        output_audio_list.append(processed_part)

    filtered_audio = None
    for audio_signal in output_audio_list:
        if filtered_audio is None:
            filtered_audio = audio_signal
        filtered_audio += audio_signal

    return filtered_audio
