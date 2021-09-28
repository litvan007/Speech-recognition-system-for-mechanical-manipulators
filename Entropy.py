import librosa
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
import math

arr, sr = librosa.load('sa.wav', sr=22050)
arr = arr/max(abs(arr)) # norm arr
time = float(arr.size/sr)
print(" Sample's count: %d" % arr.size, "\n Frequency of discretization: %d" % sr, "\n Sound time: %s " % time, "\n delta t of samples: % s" % float(10**3/sr))

def frames_create():
    n = arr.size
    frame_length = 25
    hop_length = 10
    dt = time*1000/n
    step = int(hop_length/dt)
    frame_edge = int(frame_length/dt)
# arr[i:i+frame_edge+1]
    frames_matrix = np.array([])
    for i in range(0, n-2*step, step):
        temp = np.array([arr[j] for j in range(i, i+frame_edge, 1)])
        frames_matrix = np.append(frames_matrix, temp, axis=0)
    frames_matrix = np.reshape(frames_matrix, (int((n-2*step)/step)+1, frame_edge))
    return frames_matrix

def count_frames(frame, l=9):
    counts_frame = list()
    bins = round(len(frame)/l)
    print("bins size:", bins)
    for i in range(round(len(frame)/l)):
        counts_ampl = {}
        for amp in frame[i*l:i*l+l]:
            if amp not in dict.keys(counts_ampl):
                temp = {amp : 1}
                counts_ampl.update(temp)
            else:
                counts_ampl[amp] += 1
        counts_frame.append(counts_ampl)
    plt.figure(figsize=(15, 4))
    plt.hist(frame, edgecolor="black", bins=bins)
    plt.show()
    # proposition(ampl) = count of ampl divided by length of frame
    return counts_frame

def frame_entropy(frame):
    lst_counts = count_frames(frame)
    N = len(frame)
    sum = 0
    for interval in lst_counts:
        for count in interval.values():
            p = count/N
            sum =+ p * math.log(p)
    return -sum

def frame_energy(frame):
    N = len(frame)
    return 1/N*sum([(x-np.mean(frame))**2 for x in frame]) #biased variance estimate

def frame_MFCC(frame, sr=sr):
    return librosa.feature.mfcc(frame, sr=sr)


frames = frames_create()
print(frames.shape)
temp1 = frame_entropy(frames[100])
temp2 = frame_energy(frames[100])
temp3 = frame_MFCC(frames[100])
print("Frame's entropy:", temp1, "Frame's energy:", temp2, '\n', "Frame's MFCC's vector:\n", temp3)




