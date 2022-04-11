import logging

import librosa
import nemo.collections.asr as nemo_asr
from vosk import Model, KaldiRecognizer
from content.speech_recognizer import SpeechRecognizer

from Speech_recognition.speech_grid import sp_grid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class sp_recognition():
    def __init__(self, *args):
        if (len(args) == 1):
            self.files_name = args[0]
        else:
            self.words_edges = None
            self.split_type = args[4]
            self.sound = sp_grid(*args)
        self.sber_quartzNet = nemo_asr.models.EncDecCTCModel.restore_from("../content/QuartzNet15x5_golos.nemo")
        # self.vosk_model = KaldiRecognizer(Model('../content/vosk-model-small-ru-0.22'), 8000)
        # self.recognizer = SpeechRecognizer()

    def signal_input_recongition(self, sound_temp, all_sound, num, state):
        all_sound.export('../all_sound.wav', format='wav')
        sound_temp.export('../sound_temp.wav', format='wav')
        arr, sr = librosa.load('../sound_temp.wav', sr=22050)
        if self.split_type == 'zero_cross':
            self.sound.speech_split_zc_onflow(arr, num, state)
        if self.split_type == 'entropy':
            self.sound.speech_split_entropy_onflow(arr, num, state)

        self.__speech_recognition_onflow()

    def sber_model_transcript(self):
        return self.sber_quartzNet.transcribe(paths2audio_files=self.files_name, batch_size=1024,
                                              return_hypotheses=False, logprobs=False)

    def vosk_model_transcript(self):
        self.answer = [self.vosk_model.AcceptWaveform(file) for file in self.files_name]
        return self.answer

    def __speech_recognition_onflow(self, name_model='sber'):
        if name_model == 'sber':
            if self.sound.last_file == None:
                return 0
            else:
                temp = self.sber_quartzNet.transcribe(paths2audio_files=[self.sound.last_file], batch_size=1024)
                self.sound.words_names.append(temp[0])
                self.sound.last_file = None

    def __del__(self):
        del self.sound
        logger.info('Deleting...')
    # def sova_model_transcript(self):
    #     self.answers = [self.recognizer.recognize(file).text for file in self.files_name]
    #     return self.answers
