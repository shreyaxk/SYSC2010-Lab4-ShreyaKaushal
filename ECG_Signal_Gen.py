import numpy as np
import matplotlib.pyplot as plt
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
