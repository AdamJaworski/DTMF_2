import numpy as np
from utilities import find_loudest_frequency
import scipy.io.wavfile as wav
codes = {}
set_of_freq = set()


def create_code_dict(audio_segments: list, fs: float):
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
    vector = np.array(list(set_of_freq))
    low = find_loudest_frequency(audio, fs, 600, 1000)
    top = find_loudest_frequency(audio, fs, 1100, 1550)
    low_vector = abs(vector - low)
    top_vector = abs(vector - top)
    low_value = vector[np.where(low_vector == np.min(low_vector))][0]
    top_value = vector[np.where(top_vector == np.min(top_vector))][0]
    return codes[(low_value, top_value)]

