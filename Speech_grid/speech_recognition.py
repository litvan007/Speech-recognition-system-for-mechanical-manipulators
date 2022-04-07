import logging
import nemo
import nemo.collections.asr as nemo_asr
from Speech_grid.speech_info import sp_info

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class sp_recognition():
    def __init__(self, *args):
        self.files_name = [f'../sounds_answers/{name}' for name in args[0]]
        self.transcription = list()
        self.sber_quartzNet = nemo_asr.models.EncDecCTCModel.restore_from("../content/QuartzNet15x5_golos.nemo")

    def transcript(self):
        return self.sber_quartzNet.transcribe(paths2audio_files=self.files_name, batch_size=20)
