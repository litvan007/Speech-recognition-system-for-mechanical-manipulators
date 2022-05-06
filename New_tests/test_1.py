import logging

from Voice_recognition_system.Find_words import speech_split
from Voice_recognition_system.Recognition_words import speech_recognition
import multiprocessing as mp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_split_object(Q, W, file_name, sr, frame_length, hop_length):
    system = speech_split(Q, W, file_name, sr, frame_length, hop_length)
    system.processer()

def create_recognition_object(Q, W):
    system = speech_recognition(Q, W)
    system.processer()

if __name__ == '__main__':
    logger.info('Procedure start')

    sr = 22050
    frame_length = 25
    hop_length = 10
    file_name = '../test_audio_2.wav'

    Q = mp.Manager().Queue()
    W = mp.Manager().Queue()

    p1 = mp.Process(target=create_split_object, args=(Q, W, file_name, sr, frame_length, hop_length,),
                    name='Split_process')
    p2 = mp.Process(target=create_recognition_object, args=(Q, W,),
                    name='Recognition_process')
    p2.start()
    p1.start()

    p1.join()
    p2.join()

    item = W.get()
    while item is not None:
        print(f'{item} is word')
        item = W.get()


