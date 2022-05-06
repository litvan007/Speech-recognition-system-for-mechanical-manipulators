import logging
import time
import multiprocessing as mp

import numpy as np
from pydub import AudioSegment
import librosa
from share_array.share_array import get_shared_array, make_shared_array
from ctypes import c_char_p
logging.disable(logging.CRITICAL)
import nemo.collections.asr as nemo_asr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class speech_recognition():
    def __init__(self, Q, W):
        # Data init for words recognition
        self.Q = Q
        self.W = W

        # Models of recognition
        self.sber_quartzNet = nemo_asr.models.EncDecCTCModel.restore_from("../content/QuartzNet15x5_golos.nemo")

    def __del__(self):
        self.W.put(None)
        logger.info('Deleting...')

    def processer(self):
        while True:
            file = self.Q.get()
            print(f'{file} is cur file')
            if file is not None:
                self.W.put(self.sber_quartzNet.transcribe(paths2audio_files=[file], batch_size=1024))
            else:
                break


