import logging

from Voice_recognition_system.Voice_recognition import voice_recognition

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    logger.info('Procedure start')

    sr = 22050 # частота дискретизации
    frame_length = 25 # длина фрейма в мс
    hop_length = 10 # шаг разбиения на фреймы в мс
    file_name = '../test_audio_2.wav' # название файла

    system = voice_recognition(file_name, sr, frame_length, hop_length)
    system.start()

    # print(f'Words: {system.words}\nTime: {system.words_time}')


    del system


