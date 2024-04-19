import numpy as np
import scipy.io.wavfile as wav
import librosa
from filters import apply_fir

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
    if calibration_sequence:
        print("Calibrating signal..")

    audio = apply_fir(audio, fs, [607, 770, 852, 941, 1209, 1336, 1477])
    wav.write('test.wav', fs, audio)
    return code


if __name__ == "__main__":
    # audio file
    audio_path = r'D:/DTMF/data/'
    audio_on_open, fsr = librosa.load(audio_path + 'challenge 2024.wav')

    # starting main process
    code = main(audio_on_open, fsr)
    print(code)