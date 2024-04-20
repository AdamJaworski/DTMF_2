import numpy as np
import utilities
from utilities import get_volume
from scipy.fft import fft, ifft
from filters import butter_bandpass, apply_python_filter, apply_python_filter2
import codes


def normalize_audio_in_time(audio: np.ndarray, fs: float, resolution: float = 2.99, step: float = 512, target_db: float = 35):
    """

    :param audio:
    :param fs:
    :param resolution: time in second for each frame
    :param step:
    :param target_db:
    :return:
    """
    for i in range(1, int(len(audio) / (fs * (resolution + 0.1))) + 1):
        left  = int((i - 1) * (fs * (resolution + 0.1)) + step)
        right = min(int(i * fs * (resolution + 0.1)), len(audio))
        segment = audio[left:right]

        loudness_dB = get_volume(segment)
        if loudness_dB == 0:
            continue

        delta_dB = target_db - loudness_dB
        scale_factor = 10 ** (delta_dB / 20)

        filtered_all = np.zeros_like(segment)
        for freq in codes.set_of_freq:
            filtered_freq = apply_python_filter2(segment, fs, [freq], bandwidth=30)
            if np.any(np.isnan(filtered_freq)):
                continue
            if get_volume(filtered_freq) < -20:
                continue

            filtered_all += filtered_freq.astype(filtered_all.dtype)

        filtered_all = filtered_all * scale_factor

        audio[left:right] = filtered_all.astype(audio.dtype)
    return audio


def extract_audio_parts(audio, fs, step: float = 0.05, threshold: float = 25, expected_len: float = 0.5, tolerance: float = 0.03, return_len: bool = False):
    extracting_sound = False
    star_pos = 0
    top_freq = 0
    bottom_freq = 0
    borders = []
    for i in range(int(len(audio) / (fs * step))):
        left_limit = int((i - 1) * (fs * step))
        right_limit = int(i * fs * step)
        volume = get_volume(audio[left_limit:right_limit])

        # check if volume of over threshold limit
        if volume > threshold:
            bottom_freq_ = utilities.find_loudest_frequency(audio[left_limit:right_limit], fs, 550, 1000)
            top_freq_    = utilities.find_loudest_frequency(audio[left_limit:right_limit], fs, 1150, 1500)

            if extracting_sound:
                # check if sound is the same:
                if (top_freq * (1 - tolerance)) < top_freq_ < (top_freq * (1 + tolerance)) and (bottom_freq * (1 - tolerance)) < bottom_freq_ < (bottom_freq * (1 + tolerance)):
                    continue
                else:
                    borders.append((star_pos, left_limit))
                    extracting_sound = False
            else:
                extracting_sound = True
                star_pos    = left_limit
                top_freq    = top_freq_
                bottom_freq = bottom_freq_

        elif extracting_sound:
            borders.append((star_pos, left_limit))
            extracting_sound = False

    total_len = 0
    borders_ = []
    for border in borders:
        if not (border[1] - border[0]) / fs < expected_len:
            local_len = border[1] - border[0]
            total_len += local_len
            borders_.append(audio[border[0]:border[1]])

    borders = borders_
    print(f"Found {len(borders)} parts")
    length = (total_len / len(borders)) / fs

    if return_len:
        return borders, length
    else:
        return borders

