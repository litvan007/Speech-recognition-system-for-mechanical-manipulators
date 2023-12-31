import librosa
import numpy as np

import logging

from Speech_recognition.speech_grid import sp_grid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    logger.info('Work start')

    name = 'test_audio_5'
    arr, sr = librosa.load('../'+name+'.m4a', sr=22050)
    arr = arr / np.max(arr) # For beautiful graphics
    frame_length = 25
    hop_length = 10

    n = arr.size
    time_all = float(n / sr)
    dt = time_all * 1000 / n  # Время в ms через которое начнется следующий элумент sample
    chunk = int(frame_length / dt)

    sound_split = np.array([])
    sound_zc = sp_grid(sr, frame_length, hop_length, name)
    sound_entropy = sp_grid(sr, frame_length, hop_length, name)

    # Voice feeding
    logger.info('Start Telling for ZC')

    sound_zc = sp_grid(sr, frame_length, hop_length, name)
    i = 0
    while arr.size - i*chunk > 0:
        sound_split = arr[i * chunk:(i + 1) * chunk]
        sound_zc.speech_split_zc_onflow(sound_split, i + 1, False)
        i += 1
    # Ending
    logger.info('End Telling for ZC')
    sound_zc.speech_split_zc_onflow(sound_split, 1, True)
    del sound_zc

    logger.info('Start Telling for Entropy')
    sound_entropy = sp_grid(sr, frame_length, hop_length, name)
    i = 0
    while arr.size - i*chunk > 0:
        sound_split = arr[i * chunk:(i + 1) * chunk]
        sound_entropy.speech_split_entropy_onflow(sound_split, i + 1, False)
        i += 1
    # Ending
    logger.info('End Telling for Entropy')
    sound_entropy.speech_split_entropy_onflow(sound_split, 1, True)
    del sound_entropy
