import logging

import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from pydub import AudioSegment
from scipy.stats.mstats import gmean

from Speech_grid.speech_info import sp_info

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class sp_grid(sp_info):
    def __init__(self, _arr, _sr, _frames_length, _hop_length):
        super().__init__(_arr, _sr, _frames_length, _hop_length)
        self.T = 0
        self.R = 0
        self.speech_range = 15
        self.noise_range = 5

    def speech_split_with_zero_crossing_rate(self):
        n = self.frames.shape[0]

        Mark = {frame: 0 for frame in np.arange(self.ind_noise_test)}  # FIXME добавить вариацию времени входной калибровки
        Noise = set([i for i in np.arange(self.ind_noise_test)])  # номер фреймов содержащие только шум
        e = self.__e_edge(Noise)
        z = self.__z_edge(Noise)

        for m in np.arange(self.ind_noise_test, n, 1):
            if self.frames_energy(self.frames[m]) < e:
                Mark = self.__add_edges(0, Mark, m)
                Noise.add(m)
                e = self.__e_edge(Noise)
                # z = self.__z_edge(Noise)
            else:
                if self.frames_rootmeansquare(self.frames[m])/self.frames_zero(self.frames[m]) > z: # Можно просто сравнивать zero cross с z, но точность меньше
                    Mark = self.__add_edges(1, Mark, m)
                else:
                    Mark = self.__add_edges(2, Mark, m)
                    # Noise.add(m)
                    # z = self.__z_edge(Noise)
        return Mark

    def speech_split_with_short_term_feature(self):
        energy_PrimTresh = 40
        F_PrimTresh = 185
        SF_PrimTresh = 5
        n = self.frames.shape[0]
        energy = np.array([])
        F = np.array([])
        SFM = np.array([])
        Mark = {}

        for m in np.arange(self.ind_noise_test):
            yf = fft(self.frames[m])
            xf = fftfreq(np.size(self.frames[m]), 1/self.sr)

            energy = np.append(energy, self.frames_energy(self.frames[m]))
            F = np.append(F, xf[np.argmax(yf)])
            SFM = np.append(SFM, 10*np.log(gmean(yf)/np.mean(yf)+1e2))
            temp = {m : 0}
            Mark.update(temp)

        Min_E, Min_F, Min_SF = np.min(energy), np.min(F), np.min(SFM)
        Tresh_E = energy_PrimTresh * np.log(Min_E+1e2)
        Tresh_F = F_PrimTresh
        Tresh_SF = SF_PrimTresh
        Silence_count = 0
        Speech_count = 0
        for m in np.arange(self.ind_noise_test, n, 1):
            yf = fft(self.frames[m])
            xf = fftfreq(np.size(self.frames[m]), 1/self.sr)
            E = self.frames_energy(self.frames[m])

            counter = 0
            if E - Min_E >= Tresh_E:
                counter += 1
            if xf[np.argmax(yf)] - Min_F >= Tresh_F:
                counter += 1
            if 10*np.log10(gmean(yf)/np.mean(yf)+1e10) - Min_SF >= Tresh_SF:
                counter += 1

            if counter > 1:
                Speech_count += 1
            else:
                Silence_count += 1
                Min_E = ((Silence_count * Min_E) + E) / (Silence_count + 1)

            if Silence_count >= 10:
                for i in range(10):
                    Mark[m - i] = 0
                Speech_count = 0
            if Speech_count >= 5:
                for i in range(5):
                    Mark[m - i] = 1
                Silence_count = 0

        Tresh_E = energy_PrimTresh * np.log(Min_E)
        return Mark

    def speech_split_with_entropy(self): # может чаще ошибаться
        if self.time < 1:
            return {}
        else:
            n = self.frames.shape[0]

            Mark = {frame: 0 for frame in range(self.ind_noise_test)}  # FIXME добавить вариацию времени входной калибровки
            Noise = set([i for i in range(self.ind_noise_test)])  # номер фреймов содержащие только шум
            e, h = self.__e_edge(Noise), self.__h_edge(Noise)

        for m in np.arange(self.ind_noise_test, n, 1):
            if self.frames_energy(self.frames[m]) < e:  # FIXME подумать над тем, чтобы делать пересчитывание только в том случае, если меняется систематический шум
                Mark = self.__add_edges(0, Mark, m)
                Noise.add(m)
                e = self.__e_edge(Noise)
                # h = self.__h_edge(Noise)
            else:
                if self.frames_entropy(self.frames[m]) < h:
                    Mark = self.__add_edges(1, Mark, m)
                else:
                    Mark = self.__add_edges(2, Mark, m)
                    Noise.add(m)
                    h = self.__h_edge(Noise)
        return Mark

    def __h_edge(self, Noise):
        temp_h = np.array([self.frames_entropy(self.frames[m]) for m in Noise])
        Mh = np.mean(temp_h)
        Dh = np.var(temp_h)
        return Mh + np.sqrt(Dh)

    def __e_edge(self, Noise):
        temp_e = np.array([self.frames_energy(self.frames[m]) for m in Noise])
        Me = np.mean(temp_e)
        return Me

    def __z_edge(self, Noise):
        temp_z = np.array([self.frames_zero(self.frames[m]) for m in Noise])
        temp_re = np.array([self.frames_rootmeansquare(self.frames[m]) for m in Noise])
        Mz = np.mean(temp_z)
        Dz = np.var(temp_z)
        Erms = np.mean(temp_re)
        return Erms/Mz

    def __add_edges(self, num, Mark, m):
        if num == 0:
            self.T = 0
            temp = {m : 0}
            Mark.update(temp)

        if num == 1 and self.T < self.speech_range:
            self.T += 1
        elif num == 1 and self.T == self.speech_range:
            for i in np.arange(m - self.T + 1, m - self.T + self.speech_range, 1):
                temp = {i : 1}
                Mark.update(temp)
            self.T += 1
        elif num == 1 and self.T > self.speech_range:
            self.R = 0
            temp = {m : 1}
            Mark.update(temp)

        if num == 2:
            temp = {m : 2}
            Mark.update(temp)
        return Mark

    def find_words_edges(self, mark):
        lst = np.array([[0, 0]])
        flag = False
        temp = []
        for k, v in mark.items():
            if v == 1 and flag == False:
                temp.append(k/100)
                flag = True
            if v == 0 and flag == True:
                temp.append(k/100)
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
        plt.savefig('../graphs_answers/'+title + '.png')
        plt.show()

    def export_words(self, words_grid, title):
        sound_file = AudioSegment.from_file('../test_audio_3.m4a')
        for x, y in words_grid:
            new_file = self.arr_original[int(x*self.sr):int(y*self.sr)]
            song = AudioSegment(new_file.tobytes(), frame_rate=sound_file.frame_rate,sample_width=sound_file.sample_width,channels=1)
            song.export('../sounds_answers/' + title + ': ' + str(x) + ' - ' + str(y) + ".mp3", format='mp3')