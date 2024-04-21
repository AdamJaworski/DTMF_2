import scipy.io.wavfile as wav
import numpy as np
from scipy.fft import fft, ifft
from scipy.signal.windows import get_window

codes = {}
set_of_freq = set()


def apply_python_filter(audio: np.ndarray, fs: float, frequencies: list, bandwidth: int = 20) -> np.ndarray:
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

        lower_bound = max(lower_bound, 0)
        upper_bound = min(upper_bound, n // 2)

        window = get_window('hamming', upper_bound - lower_bound, False)
        window = np.pad(window, (lower_bound, n - upper_bound), 'constant', constant_values=(0, 0))

        filtered_fft[:upper_bound] += window[:upper_bound] * fft_data[:upper_bound]
        filtered_fft[-upper_bound:] += window[:upper_bound][::-1] * fft_data[-upper_bound:]

    filtered_audio = ifft(filtered_fft)
    filtered_audio = np.real(filtered_audio).astype(audio.dtype)

    return filtered_audio


def get_volume(audio) -> float:
    """
    Returns volume in dB
    :param audio:
    :return:
    """
    if audio.size == 0:
        return 0
    rms_value = np.sqrt(np.mean(audio ** 2))
    if np.isnan(rms_value):
        return -np.inf
    epsilon = 1e-10
    reference = 1.0
    return 20 * np.log10(max(rms_value, epsilon) / reference)


def find_loudest_frequency(audio, fs, min_freq, max_freq) -> float:
    """
    Returns loudest freq from range (min, max)
    :param audio:
    :param fs:
    :param min_freq:
    :param max_freq:
    :return:
    """
    fft_values = fft(audio)
    mag_fft = np.abs(fft_values)

    freqs = np.fft.fftfreq(len(audio), 1 / fs)
    indices = np.where((freqs >= min_freq) & (freqs <= max_freq))

    loudest_freq = freqs[indices][np.argmax(mag_fft[indices])]
    return loudest_freq


def normalize_audio_in_time(audio: np.ndarray, fs: float, step: float = 512, target_db: float = 35):
    """
    Changes volume of each frame to target_dB
    :param audio: ndarray of audio signal
    :param fs: sampling freq of audio
    :param step: Distance between frames, so they won't overlap
    :param target_db: target volume in dB
    :return: normalized audio single
    """
    resolution = np.clip((len(audio) / fs) / (target_db + (step / 4) / 100), 2.7, 3.2)
    for i in range(1, int(len(audio) / (fs * resolution)) + 1):
        left  = int((i - 1) * (fs * resolution) + step)
        right = min(int(i * fs * resolution), len(audio))
        segment = audio[left:right]

        loudness_dB = get_volume(segment)
        if loudness_dB == 0:
            continue

        delta_dB = target_db - loudness_dB
        scale_factor = 10 ** (delta_dB / 20)

        filtered_all = np.zeros_like(segment)
        for freq in set_of_freq:
            filtered_freq = apply_python_filter(segment, fs, [freq], bandwidth=30) # This is not filter!!! it should be only selecting freq, so bandwidth is higher
            if np.any(np.isnan(filtered_freq)):
                continue
            if get_volume(filtered_freq) < -target_db:
                continue

            filtered_all += filtered_freq.astype(filtered_all.dtype)

        filtered_all = filtered_all * scale_factor

        audio[left:right] = filtered_all.astype(audio.dtype)
    return audio


def extract_audio_parts(audio, fs, step: float = 0.05, threshold: float = 25, expected_len: float = 0.5, tolerance: float = 0.03, return_len: bool = False) -> list:
    """

    :param audio: ndarray of audio signal
    :param fs: sampling freq of audio
    :param step: Distance between frames, so they won't overlap
    :param threshold:
    :param expected_len:
    :param tolerance:
    :param return_len:
    :return: List of audio parts
    """
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
            bottom_freq_ = find_loudest_frequency(audio[left_limit:right_limit], fs, 550, 1000)
            top_freq_    = find_loudest_frequency(audio[left_limit:right_limit], fs, 1150, 1500)

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
    length = (total_len / len(borders)) / fs

    if return_len:
        return borders, length
    else:
        return borders


def create_code_dict(audio_segments: list, fs: float):
    """
    Creates code dict used for decoding
    :param audio_segments:
    :param fs:
    :return:
    """
    global set_of_freq
    set_of_freq_ = set()
    if len(audio_segments) != 12:
        raise RuntimeWarning("Not enough data")

    for audio in audio_segments:
        bottom_freq = find_loudest_frequency(audio, fs, 600 , 1000)
        top_freq    = find_loudest_frequency(audio, fs, 1100, 1550)
        set_of_freq_.add(bottom_freq)
        set_of_freq_.add(top_freq)

    array_700  = np.array([])
    array_770  = np.array([])
    array_850  = np.array([])
    array_940  = np.array([])
    array_1200 = np.array([])
    array_1300 = np.array([])
    array_1400 = np.array([])
    code_values = [array_700, array_770, array_850, array_940, array_1200, array_1300, array_1400]
    vector     = np.array([697, 770, 852, 941, 1209, 1336, 1477])
    for freq in set_of_freq_:
        result = abs(vector - freq)
        index = np.where(result == np.min(result))[0][0]
        code_values[index] = np.append(code_values[index], freq)

    for index, array in enumerate(code_values):
        code_values[index] = int(array.mean())

    set_of_freq = set(code_values)

    codes[(code_values[0], code_values[4])] = '1'
    codes[(code_values[0], code_values[5])] = '2'
    codes[(code_values[0], code_values[6])] = '3'

    codes[(code_values[1], code_values[4])] = '4'
    codes[(code_values[1], code_values[5])] = '5'
    codes[(code_values[1], code_values[6])] = '6'

    codes[(code_values[2], code_values[4])] = '7'
    codes[(code_values[2], code_values[5])] = '8'
    codes[(code_values[2], code_values[6])] = '9'

    codes[(code_values[3], code_values[4])] = '*'
    codes[(code_values[3], code_values[5])] = '0'
    codes[(code_values[3], code_values[6])] = '#'


def decode(audio, fs):
    """
    Decodes code from audio
    :param audio: chunk of audio containing only one DTMF tone
    :param fs:
    :return:
    """
    vector = np.array(list(set_of_freq))
    low = find_loudest_frequency(audio, fs, 600, 1000)
    top = find_loudest_frequency(audio, fs, 1100, 1550)
    low_vector = abs(vector - low)
    top_vector = abs(vector - top)
    low_value = vector[np.where(low_vector == np.min(low_vector))][0]
    top_value = vector[np.where(top_vector == np.min(top_vector))][0]
    return codes[(low_value, top_value)]


def main(audio: np.ndarray, fs: int, calibration_sequence: bool = True, write_file: bool = False, calibration_sequence_data: tuple = (17, 28)) -> str:
    """
    Main process of decoding sequence
    :param audio: ndarray of audio signal
    :param fs: sampling frequency of audio signal
    :param calibration_sequence:
    :param write_file:
    :param calibration_sequence_data: Tuple for calibration sequence, first is length, second is threshold
    :return: code as str
    """

    global set_of_freq
    if calibration_sequence:
        audio_chunks, avg_len = extract_audio_parts(audio[:fs * calibration_sequence_data[0]], fs, threshold=calibration_sequence_data[1], return_len=True)
        create_code_dict(audio_chunks, fs)
    else:
        avg_len = 1.0
        set_of_freq = {770, 1209, 1476, 942, 853, 1336, 697}
        codes = {(697, 1209): '1', (697, 1336): '2', (697, 1476): '3', (770, 1209): '4', (770, 1336): '5', (770, 1476): '6', (853, 1209): '7', (853, 1336): '8', (853, 1476): '9', (942, 1209): '*', (942, 1336): '0', (942, 1476): '#'}

    audio = apply_python_filter(audio, fs, set_of_freq)
    audio = normalize_audio_in_time(audio, fs)
    audio_chunks = extract_audio_parts(audio, fs, expected_len=avg_len * 0.45)

    code = ''

    for chunk in audio_chunks:
        code += decode(chunk, fs)

    if write_file:
        wav.write('output_norm.wav', fs, audio)
    return code


if __name__ == "__main__":
    # audio file
    audio_path = r'./challenge 2024.wav'
    sample_rate, data = wav.read(audio_path)

    # Tutaj trzeba ustawić długość w sekunda pierwszych 12 dźwięków z kalibracji, oraz threshold głośności
    # dla kolejnych lat jest to:
    # 2024 -> (17, 28)
    # 2022 -> (19, 28)
    # 2021 -> (13, 22)
    code = main(data, sample_rate, calibration_sequence=True, calibration_sequence_data=(17, 28))
    print(code)