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
    # def __init__(self, _arr, _sr, _frame_length, _hop_length):
    def __init__(self, *args):
        # static
        if not isinstance(args[3], str):
            self.arr_original = args[0]
            # self.arr = args[0] / max(abs(args[0]))  # нормализация значений массива
            self.arr = args[0]
            self.time_old = float(args[0].size / args[1])
            self.sr = args[1]
            self.frame_length = args[2]
            self.hop_length = args[3]
            self.name = args[3]

            self.spectr = fft(self.arr)
            self.A = np.mean(self.spectr)
            self.G = gmean(self.spectr)

            self.frames_length = None
            self.frames_time = None

            self.ind_noise_test = 65
            self.Voice = []

            logger.info(
                "\nSample's count: {count}\nFrequency of discretization: {freq}\nSound time: {time}\ndelta t of sample: {dt}"
                    .format(count=self.arr.size, freq=self.sr, time=self.time_old, dt=float(10 ** 3 / self.sr)))

            self.frames_matrix = self.__frames_create()
            self.N = self.frames_matrix.shape[1]
        # on flow
        else:
            self.ind = None
            self.sr = args[0]
            self.frame_length = args[1]
            self.hop_length = args[2]
            self.name = args[3]

            self.frames_matrix = None
            self.arr = np.array([])
            self.size = 0
            self.time = None
            self.step = None
            self.frame_edge = None

            self.ind_noise_test = 65
            self.Voice = []
    def __del__(self):
        logger.info("...")

    def frames_onflow_create(self, chunk, num):
        # chunk = chunk / np.max(chunk)
        if num == -1:
            pass
        elif num == 1:
            n = chunk.size
            self.time = float(chunk.size / self.sr)
            dt = self.time * 1000 / n  # Время в ms через которое начнется следующий элумент sample
            self.step = int(self.hop_length / dt)  # tpart/time * n (перевод времени в индекс массива)
            self.frame_edge = int(self.frame_length / dt)

            self.arr = chunk.astype('float64')
            self.frames_matrix = [self.arr]
            self.ind = self.step

        else:
            self.arr = np.append(self.arr, chunk.astype('float64'))
            n = self.arr.size
            ind = self.ind
            i = 0
            while n - self.ind - i - self.frame_edge >= 0:
                temp = self.arr[ind:ind + self.frame_edge]
                self.frames_matrix = np.append(self.frames_matrix, [temp], axis=0)
                ind += self.step
                i += self.step
            self.ind = ind

    def __frames_create(self):
        self.arr = self.arr/np.max(self.arr)
        t = time.time()
        n = self.arr.size
        flag = False
        dt = self.time_old * 1000 / n  # Время в ms через которое начнется следующий элумент sample
        step = int(self.hop_length / dt)  # tpart/time * n (перевод времени в индекс массива)
        frame_edge = int(self.frame_length / dt)

        frames_matrix = None
        i = 0
        while n - i - frame_edge >= 0:
            temp = self.arr[i:i + frame_edge]
            if i == 0:
                frames_matrix = [temp]
            else:
                frames_matrix = np.append(frames_matrix, [temp], axis=0)
            i += step

        self.frames_length = int((n - 3 * step) / step) + 1  # всего фреймов
        logger.info("Frames was created for {}s".format(time.time() - t))

        frames_matrix = frames_matrix.astype('float64')
        return frames_matrix

    def counts_frames(self, frame, l=9):
        amp_min = min(frame)
        amp_max = max(frame)
        step = (amp_max - amp_min) / l
        count_frame = {interval: 0 for interval in range(l)}

        for amp in frame:  # есть много времени
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
        return count_frame

    def frames_entropy(self, frame):
        dict_count = self.counts_frames(frame)
        N = frame.size
        sm = 0
        for count in dict_count.values():
            if count != 0:
                p = count / N
                sm += p * math.log(p)

        return -sm

    def frames_energy(self, frame):
        return np.var(frame)  # biased variance estimate

    def frames_rootmeansquare(self, frame):
        return np.sqrt(np.mean(np.square(frame)))

    def frames_zero(self, frame):
        sm = 1e-10
        for m in np.arange(1):
            sm += np.abs(np.sign(frame[m]) - np.sign(frame[m - 1]))
        return 1 / 2 * sm

    def frames_SFM(self, frame):
        return 10 * np.log10(self.G / self.A)

    def frames_MFCC(self, frame):
        logger.info('Frame MFCC was created')
        return librosa.feature.mfcc(frame, sr=self.sr)
