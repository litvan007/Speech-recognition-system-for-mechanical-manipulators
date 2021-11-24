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

    arr, sr = librosa.load('../test_audio_4.m4a', sr=22050)

    frame_length = 25
    hop_length = 10

    sound = sp_grid(arr, sr, frame_length, hop_length)

    t = time.time()
    mark3 = sound.speech_split_with_zero_crossing_rate()
    print('Time of VAD splitting method with zero cross rate {}s'.format(time.time() - t))
    words_grid3 = sound.find_words_edges(mark3)
    sound.plot_voice_range(words_grid3, title='Time of VAD splitting method with zero cross rate {:f}s (test 4)'.format(time.time() - t))
    print(mark3)

    t = time.time()
    mark2 = sound.speech_split_with_entropy()
    print('Time of VAD splitting method with entropy {}s'.format(time.time() - t))
    words_grid2 = sound.find_words_edges(mark2)
    sound.plot_voice_range(words_grid2, title='Time of VAD splitting method with entropy {:f}s (test 4)'.format(time.time() - t))
    print(mark2)

    sound.export_words(words_grid2, title='entropy_test4')
    sound.export_words(words_grid3, title='zero_cross_test4')