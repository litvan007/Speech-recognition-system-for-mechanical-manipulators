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
        self.cond = mp.Manager().Value('i', 0)
        self.words = mp.Manager().list()
        self.words_time = mp.Manager().list()
        self.arr = mp.Manager().list()
        self.sr = sr

        # Процесс нахождения
        self.p1 = mp.Process(target=self.create_split_object, args=(self.Q, self.arr, self.cond, self.words_time, file_name, sr, frame_length, hop_length,),
                             name='Split_process')
        # Процесс распознавания
        self.p2 = mp.Process(target=self.create_recognition_object, args=(self.Q, self.cond, self.words),
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

        # Сохранение анимации графика
        animation = self.camera.animate(interval=25, repeat=True,
                                    repeat_delay=500)
        animation.save('../onflow_graphs_answers/On_flow_VAD_zero_cross_of_' + self.name + '.mp4')

    def __print_it(self): # Отрисовка в динамике
        while True:
            if self.cond.value == 1:
                self.cond.value = -1
                self.fig.suptitle('On flow VAD splitting method with zero_cross with sber recognition', x=0.5, y=0.91)
                self.ax = librosa.display.waveplot(np.array(self.arr[0]), self.sr)
                self.camera.snap()

            if self.cond.value == 2:
                self.cond.value = -2
                self.fig.suptitle('On flow VAD splitting method with zero_cross with sber recognition', x=0.5, y=0.91)
                self.ax = librosa.display.waveplot(np.array(self.arr[0]), self.sr)
                for t in self.words_time:
                    self.ax = plt.vlines(t[0], -max(self.arr[0]), max(self.arr[0]), colors='g', linewidth=2)
                    self.ax = plt.vlines(t[1], -max(self.arr[0]), max(self.arr[0]), colors='r', linewidth=2)
                self.camera.snap()

            if self.cond.value == 3:
                self.cond.value = -3
                self.fig.suptitle('On flow VAD splitting method with zero_cross with sber recognition', x=0.5, y=0.91)
                self.ax = librosa.display.waveplot(np.array(self.arr[0]), self.sr)
                k = 0
                for t in self.words_time:
                    self.ax = plt.vlines(t[0], -max(self.arr[0]), max(self.arr[0]), colors='g', linewidth=2)
                    self.ax = plt.vlines(t[1], -max(self.arr[0]), max(self.arr[0]), colors='r', linewidth=2)
                    if len(self.words_time) == len(self.words):
                        self.ax = plt.text((t[0] + t[1]) / 2 - 0.1, -max(self.arr[0]), self.words[k])
                        k += 1
                self.camera.snap()

            if self.cond.value == 4:
                break

    @staticmethod
    def create_split_object(Q, arr, cond, words_time, file_name, sr, frame_length, hop_length):
        system = speech_split(Q, arr, cond, words_time, file_name, sr, frame_length, hop_length)
        system.processer()

    @staticmethod
    def create_recognition_object(Q, cond, words):
        system = speech_recognition(Q, cond, words)
        system.processer()

    def __del__(self):
        logger.info('Job is done.')
