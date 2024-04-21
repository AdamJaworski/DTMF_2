import numpy as np
import scipy.io.wavfile as wav
import codes
from audio_processing import normalize_audio_in_time, extract_audio_parts
from filters import apply_python_filter
from codes import create_code_dict, decode


def main(audio: np.ndarray, fs: int, calibration_sequence: bool = True, write_file: bool = False) -> str:
    """
    Main process of decoding sequence
    :param audio: ndarray of audio signal
    :param fs: sampling frequency of audio signal
    :param calibration_sequence:
    :param write_file:
    :return: code as str
    """

    if calibration_sequence:
        print("Calibrating signal..")
        # Tutaj trzeba ustawić długość w sekunda pierwszych 12 dźwięków z kalibracji, dla 2022 jest to 19 dla 2024 jest to 17 sekund
        audio_chunks, avg_len = extract_audio_parts(audio[:fs * 19], fs, threshold=28, return_len=True)
        create_code_dict(audio_chunks, fs)
        print(codes.codes)
        print(codes.set_of_freq)
    else:
        avg_len = 1.0
        codes.set_of_freq = {770, 1209, 1476, 942, 853, 1336, 697}
        codes.codes = {(697, 1209): '1', (697, 1336): '2', (697, 1476): '3', (770, 1209): '4', (770, 1336): '5', (770, 1476): '6', (853, 1209): '7', (853, 1336): '8', (853, 1476): '9', (942, 1209): '*', (942, 1336): '0', (942, 1476): '#'}

    audio = apply_python_filter(audio, fs, codes.set_of_freq)
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
    audio_path = r'D:/DTMF/data/'
    sample_rate, data = wav.read(audio_path + 'challenge 2022.wav')

    # starting main process
    code = main(data, sample_rate)
    print(code)