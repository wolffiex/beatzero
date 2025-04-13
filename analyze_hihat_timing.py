#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import aubio
import sys
from scipy import signal

# File to analyze
if len(sys.argv) > 1:
    filename = sys.argv[1]
else:
    filename = "music_sample.wav"

print(f"Analyzing hi-hat timing in {filename}...")

# Load the audio file
y, sr = librosa.load(filename, sr=None)

# Parameters
hop_size = 512
win_size = 1024
sample_duration = len(y) / sr

# Create bandpass filter to isolate hi-hat frequencies (4000-8000 Hz)
sos = signal.butter(6, [4000, 8000], "bandpass", fs=sr, output="sos")
y_filtered = signal.sosfilt(sos, y)

# Create onset detector focused on high frequency content (good for hi-hats)
onset_detector = aubio.onset("hfc", win_size, hop_size, sr)
onset_detector.set_threshold(0.3)  # Adjust based on sensitivity needed

# Process the filtered audio to detect hi-hat onsets
total_frames = len(y_filtered)
onsets = []
onset_strengths = []

for i in range(0, total_frames - hop_size, hop_size):
    # Extract a chunk of audio from the filtered signal
    chunk = y_filtered[i : i + hop_size]

    # If the chunk is shorter than hop_size, pad it
    if len(chunk) < hop_size:
        chunk = np.pad(chunk, (0, hop_size - len(chunk)))

    # Convert chunk to float32 (required by aubio)
    chunk = chunk.astype(np.float32)

    # Detect onset
    if onset_detector(chunk):
        time = i / sr
        strength = onset_detector.get_descriptor()
        onsets.append(time)
        onset_strengths.append(strength)

# Calculate inter-onset intervals
if len(onsets) > 1:
    intervals = np.diff(onsets)
    # Convert intervals to BPM
    bpms = 60 / intervals

    # Filter out unreasonable BPMs
    valid_bpms = bpms[(bpms > 40) & (bpms < 220)]

    if len(valid_bpms) > 0:
        median_bpm = np.median(valid_bpms)
        mean_bpm = np.mean(valid_bpms)
        std_bpm = np.std(valid_bpms)

        print(f"Detected {len(onsets)} hi-hat onsets")
        print(f"Median hi-hat BPM: {median_bpm:.1f}")
        print(f"Mean hi-hat BPM: {mean_bpm:.1f} ± {std_bpm:.1f}")

        # Find mode/most common BPM using histogram
        hist, bins = np.histogram(valid_bpms, bins=20)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        most_common_idx = np.argmax(hist)
        most_common_bpm = bin_centers[most_common_idx]
        print(f"Most common hi-hat BPM: {most_common_bpm:.1f}")

        # Calculate beat consistency (lower std = more consistent)
        consistency = 1 - min(1, std_bpm / mean_bpm)
        print(f"Beat consistency: {consistency:.1%}")
    else:
        print("No valid BPM values detected.")
else:
    print("Not enough onsets detected to calculate timing.")

# Create visualizations
plt.figure(figsize=(14, 10))

# Plot 1: Waveform and detected hi-hat onsets
plt.subplot(3, 1, 1)
librosa.display.waveshow(y, sr=sr, alpha=0.6)
for onset in onsets:
    plt.axvline(x=onset, color="yellow", alpha=0.7)
plt.title("Waveform with Hi-hat Detections")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

# Plot 2: Filtered signal (hi-hat frequencies only)
plt.subplot(3, 1, 2)
librosa.display.waveshow(y_filtered, sr=sr, alpha=0.6)
plt.title("Filtered Signal (4000-8000 Hz)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

# Plot 3: Histogram of intervals
if len(onsets) > 1:
    plt.subplot(3, 1, 3)
    plt.hist(valid_bpms, bins=20, alpha=0.7)
    plt.axvline(
        x=median_bpm, color="red", linestyle="--", label=f"Median: {median_bpm:.1f} BPM"
    )
    plt.axvline(
        x=most_common_bpm,
        color="green",
        linestyle="-",
        label=f"Mode: {most_common_bpm:.1f} BPM",
    )
    plt.title("Hi-hat BPM Distribution")
    plt.xlabel("BPM")
    plt.ylabel("Count")
    plt.legend()

plt.tight_layout()
plt.savefig("hihat_timing_analysis.png")
print("Created visualization: hihat_timing_analysis.png")

# Plot the periodic pattern more clearly
if len(onsets) > 1:
    # Create a visualization that clearly shows the hi-hat pattern
    plt.figure(figsize=(14, 6))

    # Create a spectrogram focusing on hi-hat range
    D = librosa.stft(y)
    D_highpass = D.copy()

    # Zero out frequencies below 3kHz
    freq_bins = librosa.fft_frequencies(sr=sr, n_fft=2048)
    low_freq_bins = freq_bins < 3000
    D_highpass[low_freq_bins, :] = 0

    # Convert to decibels for better visualization
    D_highpass_db = librosa.amplitude_to_db(np.abs(D_highpass), ref=np.max)

    # Plot the filtered spectrogram
    librosa.display.specshow(D_highpass_db, sr=sr, y_axis="log", x_axis="time")
    plt.colorbar(format="%+2.0f dB")

    # Overlay the detected onsets
    for onset in onsets:
        plt.axvline(x=onset, color="white", alpha=0.7)

    plt.title("Hi-hat Pattern (High Frequency Components)")
    plt.ylim(3000, 10000)  # Focus on hi-hat frequencies
    plt.savefig("hihat_pattern.png")
    print("Created visualization: hihat_pattern.png")

print("\nAnalysis complete!")
