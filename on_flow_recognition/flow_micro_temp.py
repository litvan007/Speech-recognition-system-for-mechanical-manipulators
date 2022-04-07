import librosa
import numpy as np

import logging

from Speech_grid.speech_grid import sp_grid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    logger.info('Work start')

    arr, sr = librosa.load('../test_audio_3.m4a', sr=22050)

    frame_length = 25
    hop_length = 10

    n = arr.size
    time_all = float(n / sr)
    dt = time_all * 1000 / n  # Время в ms через которое начнется следующий элумент sample
    chunk = int(frame_length / dt)
    sound_translate = np.array([])
    all_sound = np.array([])

    sound1 = sp_grid(sr, frame_length, hop_length)
    # sound2 = sp_grid(arr, sr, frame_length, hop_length)

    i = 0
    while arr.size - i*chunk > 0:
        sound_translate = arr[i * chunk:(i + 1) * chunk]
        sound1.speech_split_zc_onflow(sound_translate, i+1, False)
        i += 1

    sound1.speech_split_zc_onflow(sound_translate, 1, True)
