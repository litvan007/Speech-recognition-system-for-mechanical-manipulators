import logging
import time
import multiprocessing as mp

import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
from celluloid import Camera
import librosa.display
import librosa

from Voice_recognition_system.Find_words import speech_split
from Voice_recognition_system.Recognition_words import speech_recognition

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class voice_recognition:
    def __init__(self, file_name, sr, frame_length, hop_length):
        self.Q = mp.Manager().Queue()
        self.W = mp.Manager().Queue()
        self.e = mp.Event()
        self.cond = mp.Manager().Value('i', 0)
        self.words = mp.Manager().list()
        self.arr = mp.Manager().list()
        self.sr = sr

        self.p1 = mp.Process(target=self.create_split_object, args=(self.Q, self.W, self.arr, self.cond, file_name, sr, frame_length, hop_length,),
                             name='Split_process')
        self.p2 = mp.Process(target=self.create_recognition_object, args=(self.Q, self.W, self.words),
                    name='Recognition_process')

        self.fig, self.ax = plt.subplots(figsize=(15, 10))
        self.ax.set_prop_cycle(cycler('color', ['#1e73ad']))
        self.camera = Camera(self.fig)
        self.name = file_name.split('/')[1].split('.')[0]

    def start(self):
        self.p2.start()
        self.p1.start()
        self.__print_it()

        self.p2.join()
        self.p1.join()

        animation = self.camera.animate(interval=25, repeat=True,
                                    repeat_delay=500)
        animation.save('../onflow_graphs_answers/On_flow_VAD_zero_cross_of_' + self.name + '.mp4')

    def __print_it(self): # Отрисовка должна работать быстрее чем нахождение гололсвой активности на фрейме
        while True:
            if self.cond.value == 1:
                self.fig.suptitle('On flow VAD splitting method with zero_cross with sber recognition', x=0.5, y=0.91)
                self.ax = librosa.display.waveplot(np.array(self.arr[0]), self.sr)
                self.camera.snap()
                self.cond.value = 0

            if self.cond.value == 2:
                break

    @staticmethod
    def create_split_object(Q, W, arr, cond, file_name, sr, frame_length, hop_length):
        system = speech_split(Q, W, arr, cond, file_name, sr, frame_length, hop_length)
        system.processer()

    @staticmethod
    def create_recognition_object(Q, W, words):
        system = speech_recognition(Q, W, words)
        system.processer()

    def __del__(self):
        print(self.arr)
        print(self.words)
        plt.figure(figsize=(15, 10))
        librosa.display.waveplot(np.array(self.arr[0]), self.sr)
        plt.show()
        logger.info('Job is done.')
