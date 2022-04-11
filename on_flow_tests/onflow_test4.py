import logging

from Speech_recognition.speech_recognition import sp_recognition
from pydub import AudioSegment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    logger.info('Procedure start')

    file_name = '../test_audio_4.m4a'
    sound_file = AudioSegment.from_file(file_name)
    print(len(sound_file))

    i = 0
    sr = 22050
    frame_length = 25
    hop_length = 10
    all_sound = None
    sound_temp = None
    chunk = frame_length

    model = sp_recognition(sr, frame_length, hop_length, file_name, 'zero_cross')
    while len(sound_file) - i*chunk > 0:
        sound_temp = sound_file[i * chunk:(i+1) * chunk]
        if i == 0:
            all_sound = sound_temp
        else:
            all_sound += sound_temp
        model.signal_input_recongition(sound_temp, all_sound, i+1, False)
        i += 1
    model.signal_input_recongition(sound_temp, all_sound, 1, True)
    del model
