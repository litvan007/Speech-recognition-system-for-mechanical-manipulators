import time

import librosa
from scipy.fft import fft, fftfreq
from scipy.stats.mstats import gmean
import numpy as np
import math

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class sp_info:
    def __init__(self, _arr, _sr, _frame_length, _hop_length):
        self.frame_length = _frame_length
        self.hop_length = _hop_length
        self.arr_original = _arr
        self.arr = _arr / max(abs(_arr))  # нормализация значений массива
        self.sr = _sr
        self.time = float(_arr.size / _sr)

        self.spectr = fft(self.arr)
        self.A = np.mean(self.spectr)
        self.G = gmean(self.spectr)

        self.frames_length = None
        self.ind_noise = None
        self.frames_time = None

        self.ind_noise_test = 65

        self.Voice = []

        logger.info(
            "\nSample's count: {count}\nFrequency of discretization: {freq}\nSound time: {time}\ndelta t of sample: {dt}"
                .format(count=self.arr.size, freq=self.sr, time=self.time, dt=float(10 ** 3 / self.sr)))

        self.frames = self.__frames_create()
        self.N = self.frames.shape[1]
        # self.count_frames = self.counts_frames(self.frames[100])
        # self.frame_entropy = self.frames_entropy(self.frames[100])
        # self.frame_energy = self.frames_energy(self.frames[100])
        # self.frame_mfcc = self.frames_MFCC(self.frames[100])

    def __frames_create(self):
        t = time.time()
        n = self.arr.size
        flag = False
        dt = self.time * 1000 / n  # Время в ms через которое начнется следующий sample
        step = int(self.hop_length / dt) # tpart/time * n (перевод времени в индекс массива)
        frame_edge = int(self.frame_length / dt)

        frames_matrix = np.array([])
        for i in range(0, n - 2 * step, step):  # наскоки для последовательности фреймов
            if i * dt > 10 ** 3 and flag == False: # это точнее и по времени также, чем просто self.frames_length / self.time
                self.ind_noise = int(i / step)
                flag = True

            temp = np.array([self.arr[j] for j in range(i, i + frame_edge, 1)])  # фрейм
            frames_matrix = np.append(frames_matrix, temp, axis=0)

        frames_matrix = np.reshape(frames_matrix, (
            int((n - 2 * step) / step) + 1, frame_edge))  # reshape матрицы, чтобы на каждом значении были фремйм
        self.frames_length = int((n - 2 * step) / step) + 1  # всего фреймов
        # FIXME Если вдруг не работает split слов, то проблема здесь

        logger.info("Frames was created for {}s".format(time.time() - t))
        # print(self.time*10**3 / self.frames_length, self.frame_length, step)
        return frames_matrix

    def counts_frames(self, frame, l=9):
        amp_min = min(frame)
        amp_max = max(frame)
        step = (amp_max - amp_min) / l
        count_frame = {interval: 0 for interval in range(l)}

        for amp in frame: # есть много времени
            for i in range(0, l, 1):
                ledge = amp_min + step * i
                redge = amp_min + (i + 1) * step

                if ledge <= amp <= redge:
                    count_frame[i] += 1
                    break


        # plt.figure(figsize=(15, 4))
        # plt.hist(frame, edgecolor="black", bins=l)
        # plt.show()
        # proposition(ampl) = count of ampl divided by length of frame

        # logger.info("Counts calculated")
        return count_frame

    def frames_entropy(self, frame):
        dict_count = self.counts_frames(frame)
        N = frame.size
        sm = 0
        for count in dict_count.values():
            if count != 0:
                p = count / N
                sm += p * math.log(p)

        # logger.info('Entropy was calculated')
        return -sm

    def frames_energy(self, frame):
        # logger.info('Frame energy was calculated')
        return np.var(frame) # biased variance estimate

    def frames_rootmeansquare(self, frame):
        return np.sqrt(np.mean(np.square(frame)))

    def frames_zero(self, frame):
        sm = 0
        for m in np.arange(1, self.N):
            sm += np.abs(np.sign(frame[m]) - np.sign(frame[m-1]))
        return 1/2*sm

    def frames_SFM(self, frame):
        return 10*np.log10(self.G/self.A)

    def frames_MFCC(self, frame):
        logger.info('Frame MFCC was created')
        return librosa.feature.mfcc(frame, sr=self.sr)

