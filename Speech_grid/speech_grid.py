import logging

import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
from celluloid import Camera
from pydub import AudioSegment

from Speech_grid.speech_info import sp_info

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class sp_grid(sp_info):
    # def __init__(self, _arr, _sr, _frames_length, _hop_length):
    def __init__(self, *args):
        super().__init__(*args)
        self.T = 0
        self.R = 0
        self.speech_range = 30  # позволяет убрать ложные срабатвания, но возможно правильный подобраный ind_noise также может помочь
        self.noise_range = 5

        self.Mark = {}
        self.Noise = set()

        self.e = None
        self.z = None
        self.n = None
        self.h = None
        self.ind_split = None
        self.flag = False
        self.words_flag = False

        self.words_time = np.array([np.array([0, 0])])
        self.word_time = np.array([])

        self.fig, self.ax = plt.subplots(figsize=(15, 10))
        self.ax.set_prop_cycle(cycler('color', ['#1e73ad']))
        self.camera = Camera(self.fig)

    def __del__(self):
        super().__del__()

    # On flow
    def speech_split_zc_onflow(self, chunk, num, isEnd):
        if isEnd:
            logger.info("Zero cross rate file create:...")
            animation = self.camera.animate(interval=25, repeat=True,
                                            repeat_delay=500)
            animation.save('../onflow_graphs_answers/On_flow_VAD_zero_cross_of_'+self.name+'.mp4')
            return 0

        self.frames_onflow_create(chunk, num)
        self.ax = librosa.display.waveplot(self.arr, self.sr)
        self.fig.suptitle('On flow VAD splitting method with zero cross rate', x=0.5, y=0.91)

        # On flow zero cross rate algorithm
        if num != 1 and not self.flag and self.frames_matrix.shape[0] >= self.ind_noise_test:
            self.n = self.frames_matrix.shape[0]
            self.Mark = {frame: 0 for frame in np.arange(self.ind_noise_test)}
            self.Noise = set([i for i in np.arange(self.ind_noise_test)])  # номер фреймов содержащие только шум

            self.e, self.z = self.__e_edge(self.Noise), self.__z_edge(self.Noise)
            self.ind_split = self.ind_noise_test
            self.flag = True

        if self.flag and self.frames_matrix.shape[0] > self.ind_noise_test:
            self.n = self.frames_matrix.shape[0]
            for m in np.arange(self.ind_split, self.n, 1):
                if self.frames_energy(self.frames_matrix[m]) < self.e:
                    self.Mark = self.__Notice(0, self.Mark, m)
                    self.Noise.add(m)
                    self.e = self.__e_edge(self.Noise)
                    # self.z = self.__z_edge(Noise)
                else:
                    if self.frames_rootmeansquare(self.frames_matrix[m]) \
                            / self.frames_zero(
                        self.frames_matrix[m]) > self.z:  # Можно просто сравнивать zero cross с z, но точность меньше
                        self.Mark = self.__Notice(1, self.Mark, m)

                    else:
                        self.Mark = self.__Notice(2, self.Mark, m)
                        # self.Noise.add(m)
                        # self.z = self.__z_edge(Noise)
            self.ind_split = self.n

        for t in self.words_time[1:]:
            self.ax = plt.vlines(t[0], -1, 1, colors='g', linewidth=2)
            self.ax = plt.vlines(t[1], -1, 1, colors='r', linewidth=2)
        self.camera.snap()

    # Static zero cross rate
    def speech_split_with_zero_crossing_rate(self):
        n = self.frames_matrix.shape[0]

        self.Mark = {frame: 0 for frame in
                     np.arange(self.ind_noise_test)}  # FIXME добавить вариацию времени входной калибровки
        self.Noise = set([i for i in np.arange(self.ind_noise_test)])  # номер фреймов содержащие только шум
        e, z = self.__e_edge(self.Noise), self.__z_edge(self.Noise)

        for m in np.arange(self.ind_noise_test, n, 1):
            if self.frames_energy(self.frames_matrix[m]) < e:
                self.Mark = self.__add_edges(0, self.Mark, m)
                self.Noise.add(m)
                e = self.__e_edge(self.Noise)
                # z = self.__z_edge(self.Noise)
            else:
                if self.frames_rootmeansquare(self.frames_matrix[m]) / self.frames_zero(
                        self.frames_matrix[m]) > z:  # Можно просто сравнивать zero cross с z, но точность меньше
                    self.Mark = self.__add_edges(1, self.Mark, m)
                else:
                    self.Mark = self.__add_edges(2, self.Mark, m)
                    # self.Noise.add(m)
                    # z = self.__z_edge(self.Noise)
        return self.Mark

    # On flow
    def speech_split_entropy_onflow(self, chunk, num, isEnd):
        if isEnd:
            logger.info("Entropy file create:...")
            animation = self.camera.animate(interval=25, repeat=True,
                                            repeat_delay=500)
            animation.save('../onflow_graphs_answers/On_flow_VAD_entropy_of_'+self.name+'.mp4')
            return 0

        self.frames_onflow_create(chunk, num)
        self.ax = librosa.display.waveplot(self.arr, self.sr)
        self.fig.suptitle('On flow VAD splitting method with entropy', x=0.5, y=0.91)

        # On flow entropy algorithm
        if num != 1 and not self.flag and self.frames_matrix.shape[0] >= self.ind_noise_test:
            self.n = self.frames_matrix.shape[0]
            self.Mark = {frame: 0 for frame in np.arange(self.ind_noise_test)}
            self.Noise = set([i for i in np.arange(self.ind_noise_test)])  # номер фреймов содержащие только шум

            self.e, self.h = self.__e_edge(self.Noise), self.__h_edge(self.Noise)
            self.ind_split = self.ind_noise_test
            self.flag = True

        if self.flag and self.frames_matrix.shape[0] > self.ind_noise_test:
            self.n = self.frames_matrix.shape[0]
            for m in np.arange(self.ind_split, self.n, 1):
                if self.frames_energy(self.frames_matrix[m]) < self.e:  # FIXME подумать над тем, чтобы делать пересчитывание только в том случае, если меняется систематический шум
                    self.Mark = self.__Notice(0, self.Mark, m)
                    self.Noise.add(m)
                    self.e = self.__e_edge(self.Noise)
                    # self.h = self.__h_edge(Noise)
                else:
                    if self.frames_entropy(self.frames_matrix[m]) < self.h:
                        self.Mark = self.__Notice(1, self.Mark, m)
                    else:
                        self.Mark = self.__Notice(2, self.Mark, m)
                        self.Noise.add(m)
                        self.h = self.__h_edge(self.Noise)
            self.ind_split = self.n

        for t in self.words_time[1:]:
            self.ax = plt.vlines(t[0], -1, 1, colors='g', linewidth=2)
            self.ax = plt.vlines(t[1], -1, 1, colors='r', linewidth=2)
        self.camera.snap()

    # Static entropy
    def speech_split_with_entropy(self):  # может чаще ошибаться
        n = self.frames_matrix.shape[0]

        self.Mark = {frame: 0 for frame in
                     np.arange(self.ind_noise_test)}  # FIXME добавить вариацию времени входной калибровки
        self.Noise = set([i for i in np.arange(self.ind_noise_test)])  # номер фреймов содержащие только шум
        e, h = self.__e_edge(self.Noise), self.__h_edge(self.Noise)

        for m in np.arange(self.ind_noise_test, n, 1):
            if self.frames_energy(self.frames_matrix[m]) < e:  # FIXME подумать над тем, чтобы делать пересчитывание только в том случае, если меняется систематический шум
                self.Mark = self.__add_edges(0, self.Mark, m)
                self.Noise.add(m)
                e = self.__e_edge(self.Noise)
                # h = self.__h_edge(Noise)
            else:
                if self.frames_entropy(self.frames_matrix[m]) < h:
                    self.Mark = self.__add_edges(1, self.Mark, m)
                else:
                    self.Mark = self.__add_edges(2, self.Mark, m)
                    self.Noise.add(m)
                    h = self.__h_edge(self.Noise)
        return self.Mark

    def __h_edge(self, Noise):
        temp_h = np.array([self.frames_entropy(self.frames_matrix[m]) for m in Noise])
        Mh = np.mean(temp_h)
        Dh = np.var(temp_h)
        return Mh + np.sqrt(Dh)

    def __e_edge(self, Noise):
        temp_e = np.array([self.frames_energy(self.frames_matrix[m]) for m in Noise])
        Me = np.mean(temp_e)
        return Me

    def __z_edge(self, Noise):
        temp_z = np.array([self.frames_zero(self.frames_matrix[m]) for m in Noise])
        temp_re = np.array([self.frames_rootmeansquare(self.frames_matrix[m]) for m in Noise])
        Mz = np.mean(temp_z)
        Dz = np.var(temp_z)
        Erms = np.mean(temp_re)
        return Erms / Mz

    def __Notice(self, num, Mark, m):
        if num == 0:
            self.T = 0
            temp = {m: 0}
            Mark.update(temp)

            if self.words_flag:
                self.word_time = np.append(self.word_time, m / 100)
                self.words_time = np.append(self.words_time, [self.word_time], axis=0)
                self.word_time = np.array([])
                self.words_flag = False

        if num == 1 and self.T < self.speech_range:
            self.T += 1
            temp = {m: 0}
            Mark.update(temp)
        elif num == 1 and self.T == self.speech_range:
            for i in np.arange(m - self.T + 1, m - self.T + self.speech_range, 1):
                Mark[i] = 1
            self.T += 1

            self.word_time = np.append(self.word_time, (m - self.T + 1) / 100)
            self.words_flag = True
        elif num == 1 and self.T > self.speech_range:
            self.R = 0
            # self.T += 1
            temp = {m: 1}
            Mark.update(temp)

        if num == 2:
            temp = {m: 2}
            Mark.update(temp)
        return Mark

    def __add_edges(self, num, Mark, m):
        if num == 0:
            self.T = 0
            temp = {m: 0}
            Mark.update(temp)

        if num == 1 and self.T < self.speech_range:
            self.T += 1
        elif num == 1 and self.T == self.speech_range:
            for i in np.arange(m - self.T + 1, m - self.T + self.speech_range, 1):
                temp = {i: 1}
                Mark.update(temp)
            self.T += 1
        elif num == 1 and self.T > self.speech_range:
            self.R = 0
            temp = {m: 1}
            Mark.update(temp)

        if num == 2:
            temp = {m: 2}
            Mark.update(temp)
        return Mark

    def find_words_edges(self):
        lst = np.array([[0, 0]])
        flag = False
        temp = []
        for k, v in self.Mark.items():
            if v == 1 and flag == False:
                temp.append(k / 100)
                flag = True
            if v == 0 and flag == True:
                temp.append(k / 100)
                flag = False
                lst = np.append(lst, [temp], axis=0)
                temp = []
        lst = lst[1:]
        return lst

    def plot_voice_range(self, words_grid, title):
        plt.figure(figsize=(15, 10))
        plt.title(title)
        librosa.display.waveplot(self.arr, sr=self.sr)
        for edges in words_grid:
            plt.vlines(edges[0], -1, 1, colors='g', linewidth=2)
            plt.vlines(edges[1], -1, 1, colors='r', linewidth=2)
            plt.text((edges[0] + edges[1]) / 2 - 0.1, -1.05, 'Слово')
        plt.legend(['Sound', 'Start', 'End'])
        plt.savefig('../graphs_answers/' + title + '.png')
        plt.show()

    def export_words(self, words_grid, title):
        sound_file = AudioSegment.from_file('../test_audio_3.m4a')
        for x, y in words_grid:
            new_file = self.arr_original[int(x * self.sr):int(y * self.sr)]
            song = AudioSegment(new_file.tobytes(), frame_rate=sound_file.frame_rate,
                                sample_width=sound_file.sample_width, channels=1)
            song.export('../sounds_answers/' + title + ': ' + str(x) + ' - ' + str(y) + ".mp3", format='mp3')
