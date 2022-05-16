import logging
logging.disable(logging.CRITICAL)
import nemo.collections.asr as nemo_asr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class speech_recognition:
    def __init__(self, Q, cond, words):
        # Data init for words recognition
        self.Q = Q
        self.cond = cond
        self.words = words

        # Models of recognition
        self.sber_quartzNet = nemo_asr.models.EncDecCTCModel.restore_from("../content/QuartzNet15x5_golos.nemo")

    def __del__(self):
        logger.info('Deleting...')

    def processer(self):
        while True:
            file = self.Q.get()
            if file is not None:
                word = self.sber_quartzNet.transcribe(paths2audio_files=[file], batch_size=1024)
                self.words.append(word[0])
                self.cond.value = 3
            else:
                self.cond.value = 4
                break


