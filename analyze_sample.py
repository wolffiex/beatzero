#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import aubio
import sys

# File to analyze
if len(sys.argv) > 1:
    filename = sys.argv[1]
else:
    filename = 'music_sample.wav'

print(f"Analyzing {filename}...")

# Load the audio file
y, sr = librosa.load(filename, sr=None)

# Basic file info
duration = librosa.get_duration(y=y, sr=sr)
print(f"Sample rate: {sr} Hz")
print(f"Duration: {duration:.2f} seconds")
print(f"Total samples: {len(y)}")

# Create a figure with multiple subplots
plt.figure(figsize=(14, 10))

# Plot 1: Waveform
plt.subplot(3, 1, 1)
librosa.display.waveshow(y, sr=sr)
plt.title('Waveform')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

# Plot 2: Spectrogram
plt.subplot(3, 1, 2)
D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
librosa.display.specshow(D, y_axis='log', x_axis='time', sr=sr)
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram')

# Plot 3: Chromagram (musical notes)
plt.subplot(3, 1, 3)
chroma = librosa.feature.chroma_stft(y=y, sr=sr)
librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', sr=sr)
plt.colorbar()
plt.title('Chromagram (Musical Notes)')

plt.tight_layout()
plt.savefig('music_analysis.png')
print(f"Created visualization: music_analysis.png")

# Detect beats using aubio
hop_size = 512
win_size = 1024

# Onset/beat detection
onset_detector_hfc = aubio.onset("hfc", win_size, hop_size, sr)
onset_detector_energy = aubio.onset("energy", win_size, hop_size, sr)

# Note detection
note_detector = aubio.notes("default", win_size, hop_size, sr)

# Process the file in chunks
total_frames = len(y)
hfc_onsets = []
energy_onsets = []
detected_notes = []

for i in range(0, total_frames - hop_size, hop_size):
    # Extract a chunk of audio
    chunk = y[i:i+hop_size]
    
    # If the chunk is shorter than hop_size, pad it
    if len(chunk) < hop_size:
        chunk = np.pad(chunk, (0, hop_size - len(chunk)))
    
    # Convert chunk to float32 (required by aubio)
    chunk = chunk.astype(np.float32)
    
    # Detect onsets (HFC - good for hi-hats)
    if onset_detector_hfc(chunk):
        time = i / sr
        hfc_onsets.append(time)
    
    # Detect onsets (Energy - good for kicks)
    if onset_detector_energy(chunk):
        time = i / sr
        energy_onsets.append(time)
    
    # Detect notes
    note = note_detector(chunk)
    if note.size > 0 and note[0] > 0:
        time = i / sr
        detected_notes.append(time)

# Print detection results
print(f"\nDetection results:")
print(f"HFC onsets (hi-hats): {len(hfc_onsets)}")
print(f"Energy onsets (kicks): {len(energy_onsets)}")
print(f"Notes detected: {len(detected_notes)}")

# Plot the detections on the waveform
plt.figure(figsize=(14, 6))
librosa.display.waveshow(y, sr=sr, alpha=0.6)

# Plot HFC onsets (hi-hats)
for onset in hfc_onsets:
    plt.axvline(x=onset, color='yellow', alpha=0.7, linestyle='--', label='HFC (hi-hat)')

# Plot energy onsets (kicks)
for onset in energy_onsets:
    plt.axvline(x=onset, color='red', alpha=0.7, linestyle='-', label='Energy (kick)')

# Plot notes
for note_time in detected_notes:
    plt.axvline(x=note_time, color='blue', alpha=0.5, linestyle=':', label='Note')

# Remove duplicate labels
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

plt.title('Detected Events in Audio')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.savefig('event_detection.png')
print(f"Created visualization: event_detection.png")

print("\nAnalysis complete!")