import librosa

import logging
import time

from Speech_grid.speech_grid import sp_grid
from Speech_grid.speech_recognition import sp_recognition
from Speech_grid.speech_info import sp_info

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    logger.info('Work start')
    file_name = '../test_audio_3.m4a'
    SNR_db = 20
    snr = 10.0**(SNR_db/10.0)
    arr, sr = librosa.load(file_name, sr=22050)
    frame_length = 25
    hop_length = 10

    sound = sp_grid(arr, sr, frame_length, hop_length, file_name, snr)

    t = time.time()
    mark3 = sound.speech_split_with_zero_crossing_rate()
    all_time = time.time() - t
    print('Time of VAD splitting method with zero cross rate {}s with {}db level noise'.format(all_time, SNR_db))
    words_grid3 = sound.find_words_edges()

    sound.export_words(words_grid3, title='zero_cross_test3')

    transcription = sp_recognition(sound.words_names)
    answer = transcription.transcript()
    print(answer)

    sound.plot_voice_range(words_grid3, title='Time of VAD splitting method with zero cross rate {:f}s with {} SNR db and recognition words (test 3)'.format(all_time, SNR_db), words=answer)

    # t = time.time()
    # mark2 = sound.speech_split_with_entropy()
    # print('Time of VAD splitting method with entropy {}s'.format(time.time() - t))
    # words_grid2 = sound.find_words_edges()
    # sound.plot_voice_range(words_grid2, title='Time of VAD splitting method with entropy {:f}s (test3)'.format(time.time() - t))
    # print(mark2)


    # sound.export_words(words_grid2, title='entropy_test3')
    sound.export_words(words_grid3, title='zero_cross_test3')
