import numpy as np
import scipy.io.wavfile
import scipy.io.wavfile as wav
import codes
from audio_processing import normalize_audio_in_time, extract_audio_parts
from filters import apply_python_filter2
from on_noise_operations import extend_noise, extract_noise, remove_noise
import warnings
from codes import create_code_dict, decode
"""
Plan działania:
1. odzielenie sekcji głównej od reszty
2. wyznaczenie częstotliwości tonów
3. utworzenie profilu szumu
4. wyznaczenie długości ( czas trwania ) tonu
5. zastoswanie filtrów FIR - tych bez przyszłości
6. normalizacja dźwięku w czasie
7. (odjęcie szumów) ?
8. ustawienie górnego ograniczenia dźwięku
9. podział na kolejne tony
"""


def main(audio: np.ndarray, fs: int, calibration_sequence: bool = True) -> str:
    """
    Main process of decoding sequence
    :param audio: ndarray of audio signal
    :param fs: sampling frequency of audio signal
    :param calibration_sequence:
    :return: code as str
    """

    code = ''
    avg_len = 1.0
    if calibration_sequence:
        print("Calibrating signal..")
        audio_chunks, avg_len = extract_audio_parts(audio[:fs * 17], fs, return_len=True) # trzeba ustawić długość pierwszy 12 dźwięków , dla 2022 - 19, dla 2024 - 17
        create_code_dict(audio_chunks, fs)
        print(codes.codes)
        print(codes.set_of_freq)

    audio = apply_python_filter2(audio, fs, codes.set_of_freq, bandwidth=20)
    audio = normalize_audio_in_time(audio, fs)
    audio_chunks = extract_audio_parts(audio, fs, expected_len=avg_len * 0.45)

    for chunk in audio_chunks:
        code += decode(chunk, fs)

    wav.write('output_norm.wav', fs, audio)
    return code


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=wav.WavFileWarning)
    # audio file
    audio_path = r'D:/DTMF/data/'
    sample_rate, data = wav.read(audio_path + 'challenge 2024.wav')

    # starting main process
    code = main(data, sample_rate)
    print(code)