import logging
import time
import multiprocessing as mp

import numpy as np
from pydub import AudioSegment
import librosa
from share_array.share_array import get_shared_array, make_shared_array
from ctypes import c_char_p

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class speech_split():
    def __init__(self, Q, W, file_name, sr, frame_length, hop_length):
        # Data sound input
        self.sound_file = AudioSegment.from_file(file_name)
        self.hop_length = hop_length
        self.frame_length = frame_length
        self.chunk = self.frame_length
        self.sr = sr
        self.temp_audio = np.array([])
        self.current_audio = np.array([])
        self.file_name = file_name
        self.name = file_name.split('/')[1].split('.')[0]

        # Data sound output
        self.Q = Q
        self.W = W
        self.words_files = []

        # Parameters for splitting
        self.ind_end_noise = 65
        self.n = self.ind_end_noise
        self.ind_split = self.n
        self.speech_range = 30
        self.T = 0
        self.right_indent = 1e-1
        self.left_indent = 1e-1
        self.e = None
        self.z = None
        self.flag = False
        self.words_flag = False

        # Data for splitting
        self.frames_matrix = None
        self.Noise = set()
        self.Mark = {}
        self.speech_edges = np.array([[0, 0]])
        self.words_time = np.array([[0, 0]])
        self.word_time = np.array([])

    def __del__(self):
        self.Q.put(None)
        logger.info('Deleting...')

    def processer(self):
        # Microphone input
        all_sound = None
        sound_temp = None
        i = 0
        while len(self.sound_file) - i * self.chunk > 0:
            sound_temp = self.sound_file[i * self.chunk:(i + 1) * self.chunk]
            if i == 0:
                all_sound = sound_temp
            else:
                all_sound += sound_temp
            self.__subprocesser(sound_temp, all_sound, i + 1, False)
            i += 1
        self.__subprocesser(sound_temp, all_sound, 1, True)

    def __subprocesser(self, sound_temp, all_sound, num, state):
        all_sound.export('../all_sound.wav', format='wav')
        sound_temp.export('../sound_temp.wav', format='wav')
        arr, sr = librosa.load('../sound_temp.wav', sr=self.sr)

        self.speech_split_onflow(arr, num, state)

    def __frames_onflow_create(self, chunk, num):
        if num == -1:
            pass
        elif num == 1:
            n = chunk.size
            self.time = float(chunk.size / self.sr)
            dt = self.time * 1000 / n  # Время в ms через которое начнется следующий элумент sample
            self.step = int(self.hop_length / dt)  # tpart/time * n (перевод времени в индекс массива)
            self.frame_edge = int(self.frame_length / dt) + 1

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

    # Traditional Energy
    def frames_energy(self, frame):
        return np.sum(np.square(frame))

    def __e_edge(self, Noise):
        temp_e = np.array([self.frames_energy(self.frames_matrix[m]) for m in Noise])
        Me = np.mean(temp_e)
        return Me

    # Check it in the distant future
    def frames_zero(self, frame):
        sm = 1e-10
        for m in np.arange(1):
            sm += np.sum(np.abs(np.sign(frame[m]) - np.sign(frame[m - 1])))
        return 1 /(2*self.frames_matrix.shape[0]) * sm

    def frames_rootmeansquare(self, frame):
        return np.sqrt(np.mean(np.square(frame)))

    def __z_edge(self, Noise):
        temp_z = np.array([self.frames_zero(self.frames_matrix[m]) for m in Noise])
        temp_re = np.array([self.frames_rootmeansquare(self.frames_matrix[m]) for m in Noise])
        Mz = np.mean(temp_z)
        Dz = np.var(temp_z)
        Erms = np.mean(temp_re)
        return Erms / Mz

    # Splitting by zero crossing method
    def speech_split_onflow(self, chunk, num, isEnd):
        if isEnd:
            # logger.info("Signal file create:...")
            # animation = self.camera.animate(interval=25
            #                                 repeat_delay=500)
            # animation.save('../onflow_graphs_answers/On_flow_VAD_zero_cross_of_' + self.name + '.mp4')
            return 0

        self.__frames_onflow_create(chunk, num)
        # self.ax = librosa.display.waveplot(self.arr, self.sr)
        # self.fig.suptitle('On flow VAD splitting method with zero cross rate with sber recognition', x=0.5, y=0.91)
        if num != 1 and not self.flag and self.frames_matrix.shape[0] >= self.ind_end_noise:
            self.n = self.frames_matrix.shape[0]
            self.Mark = {frame: 0 for frame in np.arange(self.ind_end_noise)}
            self.Noise = set([i for i in np.arange(self.ind_end_noise)])  # номер фреймов содержащие только шум

            self.e, self.z = self.__e_edge(self.Noise), self.__z_edge(self.Noise)
            self.ind_split = self.ind_end_noise
            self.flag = True

        if self.flag and self.frames_matrix.shape[0] > self.ind_end_noise:
            self.n = self.frames_matrix.shape[0]
            for m in np.arange(self.ind_split, self.n, 1):
                if self.frames_energy(self.frames_matrix[m]) < self.e:
                    self.__Notice(0, m)
                    self.Noise.add(m)
                    self.e = self.__e_edge(self.Noise)
                    # self.z = self.__z_edge(Noise)
                else:
                    if self.frames_rootmeansquare(self.frames_matrix[m]) \
                            / self.frames_zero(
                        self.frames_matrix[m]) > self.z:  # Можно просто сравнивать zero cross с z, но точность меньше
                        self.__Notice(1, m)

                    else:
                        self.__Notice(2, m)
                        # self.Noise.add(m)
                        # self.z = self.__z_edge(Noise)
            self.ind_split = self.n
        # k = 0
        #     # print(self.words_time[1:].shape[0], len(self.words_names))
        #     # print(self.words_time[1:], self.words_names, k)
        # for t in self.words_time[1:]:
        #     self.ax = plt.vlines(t[0], -0.5, 0.5, colors='g', linewidth=2)
        #     self.ax = plt.vlines(t[1], -0.5, 0.5, colors='r', linewidth=2)
        #     if self.words_time[1:].shape[0] == len(self.words_names):
        #         self.ax = plt.text((t[0] + t[1]) / 2 - 0.1, -0.5 - 0.01, self.words_names[k])
        #         k += 1
        # self.camera.snap()

    def __Notice(self, num, m):
        if num == 0:
            self.T = 0
            temp = {m: 0}
            self.Mark.update(temp)

            if self.words_flag:
                self.word_time = np.append(self.word_time, m / 100)
                self.words_time = np.append(self.words_time, [self.word_time], axis=0)
                self.export_words([self.word_time], self.name.split('.')[0])
                self.word_time = np.array([])
                self.words_flag = False

        if num == 1 and self.T < self.speech_range:
            self.T += 1
            temp = {m: 0}
            self.Mark.update(temp)
        elif num == 1 and self.T == self.speech_range:
            for i in np.arange(m - self.T + 1, m - self.T + self.speech_range, 1):
                self.Mark[i] = 1
            self.T += 1

            self.word_time = np.append(self.word_time, (m - self.T + 1) / 100)
            self.words_flag = True
        elif num == 1 and self.T > self.speech_range:
            self.R = 0
            # self.T += 1
            temp = {m: 1}
            self.Mark.update(temp)

        if num == 2:
            temp = {m: 2}
            self.Mark.update(temp)

    def export_words(self, words_grid, title):
        sound_file = AudioSegment.from_file(self.file_name)
        for x, y in words_grid:
            name = '../sounds_answers/' + title + ': ' + str(x) + ' - ' + str(y) + ".wav"
            song = sound_file[(x - self.left_indent) * 1000:(y + self.right_indent) * 1000]
            self.Q.put(name)
            self.words_files.append(name)
            song.export(name, format='wav')











