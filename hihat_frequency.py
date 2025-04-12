#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import scipy.signal
import sys

# File to analyze
if len(sys.argv) > 1:
    filename = sys.argv[1]
else:
    filename = 'music_sample.wav'

print(f"Analyzing hi-hat frequency in {filename}...")

# Load the audio file
y, sr = librosa.load(filename, sr=None)

# Time point we want to analyze (0.68s)
time_point = 0.68
frame_length = 2048  # FFT window size

# Convert time to sample index
sample_idx = int(time_point * sr)

# Extract audio segment around the time point (±0.1s for context)
context_samples = int(0.1 * sr)
start_idx = max(0, sample_idx - context_samples)
end_idx = min(len(y), sample_idx + context_samples)
segment = y[start_idx:end_idx]

# Create a window function to avoid spectral leakage
window = np.hanning(frame_length)

# Perform FFT on the precise moment where the hi-hat occurs
center_idx = sample_idx - start_idx
start_frame = max(0, center_idx - frame_length // 2)
end_frame = min(len(segment), start_frame + frame_length)

# If we don't have enough samples, pad with zeros
if end_frame - start_frame < frame_length:
    frame_data = np.zeros(frame_length)
    frame_data[:end_frame-start_frame] = segment[start_frame:end_frame]
else:
    frame_data = segment[start_frame:end_frame]

# Apply window function
frame_data = frame_data * window

# Compute the FFT
fft_data = np.abs(np.fft.rfft(frame_data))
fft_freq = np.fft.rfftfreq(frame_length, 1/sr)

# Detect peaks in the spectrum
peak_indices, _ = scipy.signal.find_peaks(fft_data, height=0.01*np.max(fft_data), distance=10)
peak_freqs = fft_freq[peak_indices]
peak_mags = fft_data[peak_indices]

# Sort peaks by magnitude (descending)
sort_idx = np.argsort(-peak_mags)
peak_freqs = peak_freqs[sort_idx]
peak_mags = peak_mags[sort_idx]

# Create visualization
plt.figure(figsize=(14, 10))

# Plot 1: Waveform around time point
plt.subplot(3, 1, 1)
segment_time = np.linspace(start_idx/sr, end_idx/sr, len(segment))
plt.plot(segment_time, segment)
plt.axvline(x=time_point, color='r', linestyle='--', label='Hi-hat at 0.68s')
plt.title(f'Waveform at {time_point:.2f}s ± 0.1s')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True, alpha=0.3)
plt.legend()

# Plot 2: Spectrogram around the time point
plt.subplot(3, 1, 2)
D = librosa.amplitude_to_db(np.abs(librosa.stft(segment)), ref=np.max)
librosa.display.specshow(D, y_axis='log', x_axis='time', sr=sr)
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram around the hi-hat')
plt.axvline(x=0.1, color='r', linestyle='--')  # Mark the exact time point

# Plot 3: FFT at the specific time point
plt.subplot(3, 1, 3)
plt.plot(fft_freq, fft_data)
plt.plot(peak_freqs, peak_mags, 'rx', markersize=10, label='Peaks')
plt.xscale('log')
plt.xlim(500, 22000)  # Focus on frequencies relevant to hi-hats
plt.grid(True, alpha=0.3)
plt.title(f'Frequency Spectrum at {time_point:.2f}s')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.legend()

plt.tight_layout()
plt.savefig('hihat_frequency_analysis.png')

# Print the top frequency peaks
print(f"\nTop frequency components at {time_point:.2f}s:")
for i, (freq, mag) in enumerate(zip(peak_freqs[:10], peak_mags[:10])):
    print(f"{i+1}. {freq:.1f} Hz (magnitude: {mag:.2f})")

print(f"\nHi-hat frequency analysis is complete. See hihat_frequency_analysis.png for visualization.")

# Create a zoomed visualization of the top frequency bands for hi-hats
plt.figure(figsize=(12, 6))

# Filter for typical hi-hat frequencies (3kHz - 16kHz)
hihat_freq_mask = (fft_freq >= 3000) & (fft_freq <= 16000)
hihat_freqs = fft_freq[hihat_freq_mask]
hihat_mags = fft_data[hihat_freq_mask]

plt.plot(hihat_freqs, hihat_mags)
plt.title('Hi-hat Frequency Profile (3kHz - 16kHz)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid(True, alpha=0.3)

# Add annotations for major peaks
hihat_peak_indices, _ = scipy.signal.find_peaks(hihat_mags, height=0.05*np.max(hihat_mags), distance=20)
for idx in hihat_peak_indices:
    freq = hihat_freqs[idx]
    mag = hihat_mags[idx]
    plt.annotate(f"{freq:.0f} Hz", 
                 xy=(freq, mag),
                 xytext=(0, 10),
                 textcoords='offset points',
                 ha='center',
                 arrowprops=dict(arrowstyle='->', color='red'))

plt.savefig('hihat_frequency_profile.png')
print("Hi-hat frequency profile saved to hihat_frequency_profile.png")