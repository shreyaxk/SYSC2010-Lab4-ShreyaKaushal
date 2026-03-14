import numpy as np
import matplotlib.pyplot as plt 
from scipy.signal import butter, filtfilt, find_peaks
from scipy.fft import fft, fftfreq 
from scipy.signal import find_peaks
import pandas as pd 
# 1. ECG Signal Generator Function
def generate_ecg(duration=10, fs=250, heart_rate=72):

    t = np.linspace(0, duration, duration * fs)

    # Heartbeat frequency
    f_hr = heart_rate / 60

    # Basic ECG waveform using Gaussian pulses
    ecg = np.zeros_like(t)

    beat_interval = int(fs / f_hr)

    for i in range(0, len(t), beat_interval):

        if i < len(t):
            ecg += np.exp(-((t - t[i])**2) / (2*0.01**2)) * 1.2   # R peak
            ecg += np.exp(-((t - t[i]-0.05)**2) / (2*0.02**2)) * -0.3  # S
            ecg += np.exp(-((t - t[i]-0.2)**2) / (2*0.04**2)) * 0.3   # T wave

    return t, ecg

# 2. Add Noise Sources
def add_noise(ecg, fs):

    t = np.arange(len(ecg)) / fs

    # Baseline drift (respiration)
    baseline = 0.3 * np.sin(2 * np.pi * 0.3 * t)

    # Muscle noise
    muscle_noise = 0.05 * np.random.randn(len(ecg))

    # Powerline interference
    powerline = 0.05 * np.sin(2 * np.pi * 60 * t)

    noisy_ecg = ecg + baseline + muscle_noise + powerline

    return noisy_ecg

# 3. Generate Dataset
fs = 250
duration = 10
heart_rate = 72

t, clean_ecg = generate_ecg(duration, fs, heart_rate)

noisy_ecg = add_noise(clean_ecg, fs)

# 4. Plot Example Signal
plt.figure(figsize=(10,4))

plt.plot(t, noisy_ecg, label="Noisy ECG")
plt.plot(t, clean_ecg, alpha=0.6, label="Clean ECG")

plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Synthetic ECG Signal")

plt.legend()
plt.show()

# 5. Save Dataset
data = np.column_stack((t, noisy_ecg))

df = pd.DataFrame(data, columns=["time","signal"])

df.to_csv("ecg_signal.csv", index=False)

#Part II: Low Pass Filtering 
def lowpass_filter(signal,fs,cutoff_frequency=40, order = 4):
    nyquist_condition=0.5*fs
    normal_cutoff=cutoff_frequency/nyquist_condition
    b,a = butter(order,normal_cutoff,btype ='low')
    filtered_signal=filtfilt(b,a,noisy_ecg)
    return filtered_signal 
filtered_ecg = lowpass_filter(noisy_ecg,fs=250)


plt.plot(t, clean_ecg, label="Clean ECG")
plt.plot(t, noisy_ecg, label="Noisy ECG")
plt.plot(t, filtered_ecg,label="Lowpass Filtered ECG")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Raw and Lowpass filtered signal ")
plt.legend()
plt.show()

#Part III: High Pass Filtering 
def highpass_filter(signal,fs,cutoff_frequencyh=0.5, order = 4):
    nyquist_condition=0.5*fs
    normal_cutoff=cutoff_frequencyh/nyquist_condition
    b,a = butter(order,normal_cutoff,btype ='high')
    filtered_signal=filtfilt(b,a,noisy_ecg)
    return filtered_signal 
filteredhigh_ecg = highpass_filter(noisy_ecg,fs=250)

plt.plot(t, filtered_ecg,label="Highpass Filtered ECG")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Highpass filtered signal ")
plt.legend()
plt.show()

#Part IV: Band Pass Filtering
lowcutoff = 0.5
highcutoff =40
order=4
nyquist = 0.5 * fs
low = lowcutoff/nyquist
high = highcutoff/nyquist
b, a = butter(order, [low, high], btype='band')
bandpass_ecg = filtfilt(b, a, noisy_ecg)

plt.plot(t, clean_ecg, label="Clean ECG")
plt.plot(t, noisy_ecg, label="Noisy ECG")
plt.plot(t, bandpass_ecg,label="Bandpass Filtered ECG")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Raw and Bandpass filtered signal ")
plt.legend()
plt.show()

#Part V: Feature Extraction 
#Detect peaks
min_height = 0.5
min_distance = int(0.25 * fs)
peaks, _ = find_peaks(bandpass_ecg, height=min_height, distance=min_distance)
#RR intervals 
rr_intervals = np.diff(t[peaks])
#Average heart rate 
heart_rate = 60 / rr_intervals
average_hr = np.mean(heart_rate)

plt.plot(t[peaks], bandpass_ecg[peaks],"ro",label="R-Peaks on ECG")
plt.plot(t, bandpass_ecg,label="Bandpass Filtered ECG")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("R-Peak Detection")
plt.legend()
plt.show()

#Part VI: Frequency Comparison
#FFT of raw signal 
total_samples = len(noisy_ecg)
fft_raw = fft(noisy_ecg)
fft_raw = np.abs(fft_raw)/total_samples 
freqs = fftfreq(total_samples, 1/fs)

#Compute FFT of the Filtered Signal 
fft_filtered = fft(bandpass_ecg)
fft_filtered = np.abs(fft_filtered)/total_samples

#Plot Spectra
mask = freqs >= 0
plt.figure(figsize=(10,4))
plt.plot(freqs[mask], fft_raw[mask],label="Raw ECG")
plt.plot(freqs[mask], fft_filtered[mask], label="Filtered ECG ")

plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.title("Spectrum Before and After Filtering")
plt.legend()
plt.show()