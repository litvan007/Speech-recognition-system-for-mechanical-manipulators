import librosa

import logging
import time

from pydub import AudioSegment

from Speech_recognition.speech_grid import sp_grid
from Speech_recognition.speech_info import sp_info
from Speech_recognition.speech_recognition import sp_recognition

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    logger.info('Work start')
    file_name = '../test_audio_4.m4a'
    arr, sr = librosa.load('../test_audio_4.m4a', sr=22050)

    frame_length = 25
    hop_length = 10

    sound = sp_grid(arr, sr, frame_length, hop_length, file_name)

    t = time.time()
    mark3 = sound.speech_split_with_zero_crossing_rate()
    all_time = time.time() - t
    print('Time of VAD splitting method with zero cross rate {}s'.format(all_time - t))
    words_grid3 = sound.find_words_edges()
    sound.export_words(words_grid3, title='zero_cross_test4')

    transcription = sp_recognition(sound.words_files)
    answer = transcription.sber_model_transcript()
    print(answer)

    sound.plot_voice_range(words_grid3, title='Time of VAD splitting method with zero cross rate {:f}s with recognition words (test 4)'.format(all_time), words=answer)

    # t = time.time()
    # mark2 = sound.speech_split_with_entropy()
    # print('Time of VAD splitting method with entropy {}s'.format(time.time() - t))
    # words_grid2 = sound.find_words_edges()
    # sound.plot_voice_range(words_grid2, title='Time of VAD splitting method with entropy {:f}s (test 4)'.format(time.time() - t))
    # print(mark2)
    #
    # sound.export_words(words_grid2, title='entropy_test4')
