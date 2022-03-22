import librosa

import logging
import time

from pydub import AudioSegment

from Speech_grid.speech_grid import sp_grid
from Speech_grid.speech_info import sp_info

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    logger.info('Work start')
    SNR_db = 20
    snr = 10.0**(SNR_db/10.0)
    arr, sr = librosa.load('../test_audio_1.wav', sr=22050)
    frame_length = 25
    hop_length = 10

    sound = sp_grid(arr, sr, frame_length, hop_length, snr)

    t = time.time()
    mark3 = sound.speech_split_with_zero_crossing_rate()
    print('Time of VAD splitting method with zero cross rate {}s with {}db level noise'.format(time.time() - t, SNR_db))
    words_grid3 = sound.find_words_edges()
    sound.plot_voice_range(words_grid3, title='Time of VAD splitting method with zero cross rate {:f}s with {} db level noise (test 1)'.format(time.time() - t, SNR_db))
    print(mark3)

    # t = time.time()
    # mark2 = sound.speech_split_with_entropy()
    # print('Time of VAD splitting method with entropy {}s'.format(time.time() - t))
    # words_grid2 = sound.find_words_edges()
    # sound.plot_voice_range(words_grid2, title='Time of VAD splitting method with entropy {:f}s (test 1)'.format(time.time() - t))
    # print(mark2)

    # sound.export_words(words_grid2, title='entropy_test1')
    sound.export_words(words_grid3, title='zero_cross_test1')
