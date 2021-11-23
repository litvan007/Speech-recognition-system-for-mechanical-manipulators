import sklearn
import numpy as np
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import librosa.display
from pydub import AudioSegment

# Converting
audio_data = "./sa"
audio_format = 'mp3'
new_audio_format = 'wav'
AudioSegment.from_file(audio_data+'.'+audio_format).export(audio_data+'.wav', format='wav')
arr, sr = librosa.load('test_audio_2.wav', sr=44100)

# Creating samples
t = librosa.samples_to_time(arr)
print('Общее количество кадров =% d, частота дискретизации =% d, продолжительность в секундах =% f'% (len (arr), sr, len (arr) / sr))

yf = fft(arr)
xf = fftfreq(np.size(arr), 1/sr)

# Short Term Fourier Transform
X = librosa.stft(arr)
Xdb = librosa.amplitude_to_db(abs(X))

# Plot
plt.figure(figsize=(15, 10))
plt.subplot(2, 1, 1, ylabel="Sample's Amplitude(Power)", title='Samples')
librosa.display.waveplot(arr, sr=sr)

plt.subplot(2, 1, 2, title='Spectrogram heatmap visualizations')
librosa.display.specshow(Xdb, x_axis='time', y_axis='log', sr=sr)
plt.colorbar(format="%+2.f dB")
plt.show()

# Spectral centroids and spectral rolloff
spectral_centroids = librosa.feature.spectral_centroid(arr, sr=sr)[0]
spectral_rolloff = librosa.feature.spectral_rolloff(arr+0.01, sr=sr)[0]
frames = range(len(spectral_centroids))
t = librosa.frames_to_time(frames)
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)
# print(spectral_centroids.shape)

plt.figure(figsize=(15, 10))
plt.subplot(2, 1, 1, title='Spectral centroids')
librosa.display.waveplot(arr, alpha=0.4, sr=sr)
plt.plot(t, normalize(spectral_centroids), color='b')

plt.subplot(2, 1, 2, title='Spectral rolloff')
librosa.display.waveplot(arr, alpha=0.4, sr=sr)
plt.plot(t, normalize(spectral_rolloff), color='r')
plt.show()

# Spectral bandwidth
spectral_bandwidth_2 = librosa.feature.spectral_bandwidth(arr+0.01)[0]

plt.figure(figsize=(15, 9))
librosa.display.waveplot(arr, alpha=0.4, sr=sr)
plt.plot(t, normalize(spectral_bandwidth_2), color='r')
plt.title("Spectral bandwidth")
plt.show()

#MFCC
mfccs = librosa.feature.mfcc(arr) #(20, 301)
print(mfccs)
# print(mfccs.shape)
plt.figure(figsize=(15, 9))
plt.subplot(2, 1, 1, title='MFCC vector Visualisation')
librosa.display.specshow(mfccs, x_axis='time', sr=sr)
plt.subplot(2, 1, 2, xlabel='Hz', ylabel='Amplitude(Power)', title='Spectrogram')
plt.xlim([-2000, 2000])
plt.plot(xf, np.abs(yf)) # это нет смысла рассматривать. Можно упомянуть в курсовой
plt.show()

#chromagram (sound color)
chromagram = librosa.feature.chroma_stft(arr)
plt.figure(figsize=(15, 5))
librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', cmap='coolwarm', sr=sr)
plt.title('Chromagram')
plt.show()
