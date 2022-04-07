import librosa

import logging
import time

from Speech_grid.speech_grid import sp_grid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    logger.info('Work start')
    SNR_db = 20
    snr = 10.0**(SNR_db/10.0)
    arr, sr = librosa.load('../test_audio_7.wav', sr=22050)
    frame_length = 10
    hop_length = 10

    sound = sp_grid(arr, sr, frame_length, hop_length, snr)

    t = time.time()
    sound.speech_split_with_zrc()
    words_grid3 = sound.sp_edges
    print('Time of VAD splitting method with zero cross rate {}s'.format(time.time() - t))
    print(words_grid3)
    sound.plot_voice_range(words_grid3, title='Time of VAD splitting method with zero cross rate {:f}s (test 1)'.format(time.time() - t))
