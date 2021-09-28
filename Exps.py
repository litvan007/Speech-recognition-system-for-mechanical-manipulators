import librosa
import librosa.display
import math
import numpy as np
from scipy.stats import entropy
from scipy.signal import spectrogram
import matplotlib.pyplot as plt
import scipy.signal as signal

arr, sr = librosa.load('sa.wav', sr=22050)

t = len(arr)/sr
# X = librosa.stft(arr, n_fft=512, hop_length=512)
# Xdb = librosa.amplitude_to_db(abs(X))

mfccs = librosa.feature.mfcc(arr) #(20, 301)
# print(mfccs.shape)
h = t/mfccs[0].size # time step
# print(mfccs)

def coef_of_regression(mfccs, I, i, n):
    sum1 = 0
    sum2 = 0
    for r in range(1, 2*I, 1):
        print(i, n)
        sum1 += (mfccs[i][n]*(r+n)*r)
        sum2 += r*r
    return sum1/sum2

def speed_of_sec(mfccs, D, n):
    sum = 0
    for i in range(1, D, 1):
        sum += coef_of_regression(mfccs, 3, i, n)/D
    return sum

plt.figure(figsize=(15, 4))
tt = np.linspace(0, mfccs[0].size-1, mfccs[0].size)
time = np.linspace(0, t, round(t/h))
print(mfccs[0].size)
xx = np.array([speed_of_sec(mfccs, 13, int(i)) for i in tt])
plt.plot(time, xx)
plt.show()

# f, t, Sxx = signal.spectrogram(arr, fs=10e3)
# Sxx = librosa.amplitude_to_db(np.abs(Sxx))
# plt.pcolormesh(t, f, Sxx, shading='gouraud')
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.show()





















#Xdb[t][Hz]
# print(Xdb[1][np.where(Xdb[1]>10)], np.where(Xdb[1]>10))
# print(Xdb.shape)

# plt.figure(figsize=(15, 10))
# librosa.display.waveplot(arr, sr=sr)
# plt.show()
#
# plt.figure(figsize=(15, 10))
# plt.hist(arr, color = 'blue', edgecolor = 'black', bins = int(180))
# plt.show()

# f, t, Zxx = signal.stft(arr, fs=5, window='tukey', nfft=512, nperseg=512)
# freq, time, Spec = spectrogram(arr, fs=3)
# Zxx = librosa.amplitude_to_db(abs(Zxx))
# print(f, t, Zxx)
# print(t.shape, f.shape, Zxx.shape)
# print(Xdb.shape)
# fig, ax = plt.subplots()
# img = librosa.display.specshow(Zxx, x_axis='time', y_axis='log', sr=sr, ax=ax)
# plt.colorbar(img, ax=ax, format="%+2.f dB")
# plt.show()
#
# def check(arr, entropy_edge, eps):
#     samples = librosa.samples_like(arr, hop_length=1)
#     wt = np.array([[0,0]])
#     print('samples = %s' % samples)
    # times = librosa.frames_to_time(samples, sr=sr, hop_length=1)
    # print(len(times))
    # print('times = %s' % times)
    # a = 0
    # b = 0
    # temp = 0
    # temp_old = 0
    # n = 0
    # for i in range(0, len(arr), 10):
    #     temp_old = temp
    #     temp = entropy(np.array([arr[j]+eps for j in range(i, i+10)]), base=2)
    #     print(temp)
    #     if i!=len(arr)-1 and temp > entropy_edge and n%2==0:
    #         temp_old = temp
    #         a = times[i]
    #         n =+ 1
    #     if i!=len(arr)-1 and temp <= entropy_edge and n%2!=0:
    #         b = times[i]
    #         n =+ 1
    #         wt = np.append(wt, [[a, b]], axis=0)
    #
    # return wt
dfs = np.array([0.01245, 59.36231])
def words_time_search(Xdb, a_hz, b_hz, db_edge, dfs):
    wt = np.array([[0, 0]])
    temp1 = 0
    temp2 = 0
    n = 0
    for i in range(len(Xdb)):
        hzs = np.array(np.where(Xdb[i]>db_edge))
        print(hzs)
        # hzs = np.array([hz for hz in hzs if (hzs.size > 0) and (hz > a_hz) and (hz < b_hz)])
        # if hzs.size>0 and n%2==0:
        #     temp1 = (i+1)*dfs[0]
        #     n=+1
        # if hzs.size==0 and n%2!=0:
        #     temp2 = (i+1)*dfs[0]
        #     wt = np.append(wt, [[temp1, temp2]], axis=0)
        #     n=+1
    return '0'



# fig, ax = plt.subplots()
# img = librosa.display.specshow(Xdb, x_axis='time', y_axis='log', sr=sr, ax=ax)
# plt.colorbar(img, ax=ax, format="%+2.f dB")
# plt.show()
# print(X.shape, Xdb.shape)
# print(X)
# print(Xdb)

