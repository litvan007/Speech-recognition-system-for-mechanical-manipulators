import librosa
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
from pydub import AudioSegment

arr, sr = librosa.load('test_audio_2.wav', sr=44100)
print('Общее количество кадров =% d, частота дискретизации =% d, продолжительность в секундах =% f'% (len (arr), sr, len (arr) / sr))
print(arr.shape)
samples = librosa.samples_like(arr, hop_length=1)
print('samples = %s' % samples)
times = librosa.frames_to_time(samples, sr=sr, hop_length=1)
print(len(times))
print('times = %s' % times)


def words_frame(arr, k):
    samples = librosa.samples_like(arr, hop_length=1)
    print('samples = %s' % samples)
    times = librosa.frames_to_time(samples, sr=sr, hop_length=1)
    print(len(times))
    print('times = %s' % times)
    wt = np.array([[0,0]])
    a = 0
    b = 0
    eps = 1e7
    # for i in range(round(len(arr)*0.3), round(len(arr)*0.5)):
    for i in range(len(arr)):
        if arr[i] > k:
            wt = np.append(wt, times[i])
    return wt
print(words_frame(arr, 0.1))

    # return [np.append(wt, times[i]) for i in range(len(arr)) if arr[i] > k and i%2==0]

plt.figure(figsize=(15, 4))
librosa.display.waveplot(arr, sr=sr, alpha=0.5)
plt.xlim([1.08, 1.14])
plt.show()