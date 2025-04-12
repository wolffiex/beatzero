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

print(f"Analyzing {filename} - Last Dance with Mary Jane intro...")

# Load the audio file
y, sr = librosa.load(filename, sr=None)

# Basic file info
duration = librosa.get_duration(y=y, sr=sr)
print(f"Sample rate: {sr} Hz")
print(f"Duration: {duration:.2f} seconds")
print(f"Total samples: {len(y)}")

# Create a figure for the full intro
plt.figure(figsize=(16, 10))

# Plot detailed waveform with zoomed view
plt.subplot(3, 1, 1)
librosa.display.waveshow(y, sr=sr, alpha=0.8)
plt.title('Waveform (Full Intro)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(alpha=0.3)

# Create a spectral contrast plot to highlight drum events
plt.subplot(3, 1, 2)
spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
librosa.display.specshow(spec_contrast, x_axis='time', sr=sr)
plt.colorbar(format='%+2.0f dB')
plt.title('Spectral Contrast (Helps Identify Percussion/Beat Entry)')

# Create high-frequency energy to identify hi-hats specifically
plt.subplot(3, 1, 3)
# Create a custom high-pass filter to isolate high frequencies (hi-hats)
y_highpass = librosa.effects.hpss(y)[0]  # Use harmonic-percussive source separation to isolate percussion
# Calculate energy in the high-frequency band
hop_length = 512
n_fft = 2048
S_high = np.abs(librosa.stft(y_highpass, n_fft=n_fft, hop_length=hop_length))
high_energy = np.sum(S_high, axis=0)
frames = np.arange(len(high_energy))
times = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)

plt.plot(times, high_energy, color='yellow', alpha=0.8)
plt.title('High Frequency Energy (Isolates Hi-Hats)')
plt.xlabel('Time (s)')
plt.ylabel('Energy')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('song_intro_overview.png')
print(f"Created visualization: song_intro_overview.png")

# Detailed element isolation
hop_size = 512
win_size = 1024

# Initialize detectors
onset_detector_hfc = aubio.onset("hfc", win_size, hop_size, sr)
onset_detector_energy = aubio.onset("energy", win_size, hop_size, sr)
note_detector = aubio.notes("default", win_size, hop_size, sr)

# Process the file in chunks to identify events
total_frames = len(y)
hfc_onsets = []  # Hi-hats
energy_onsets = []  # Kicks/beats
notes = []  # Potential vocals/melody

for i in range(0, total_frames - hop_size, hop_size):
    # Extract a chunk of audio
    chunk = y[i:i+hop_size]
    
    # If the chunk is shorter than hop_size, pad it
    if len(chunk) < hop_size:
        chunk = np.pad(chunk, (0, hop_size - len(chunk)))
    
    # Convert chunk to float32 (required by aubio)
    chunk = chunk.astype(np.float32)
    
    # Detect HFC onsets (hi-hats)
    if onset_detector_hfc(chunk):
        time = i / sr
        confidence = onset_detector_hfc.get_descriptor() / onset_detector_hfc.get_threshold()
        hfc_onsets.append((time, confidence))
    
    # Detect energy onsets (kicks/beats)
    if onset_detector_energy(chunk):
        time = i / sr
        energy_confidence = onset_detector_energy.get_descriptor() / onset_detector_energy.get_threshold()
        energy_onsets.append((time, energy_confidence))
    
    # Detect notes (potential vocals)
    note = note_detector(chunk)
    if note.size > 0 and note[0] > 0:
        time = i / sr
        notes.append(time)

# Find the first significant events for each element
if hfc_onsets:
    first_hihat = min([time for time, _ in hfc_onsets])
    # Find the moment where hi-hats become consistent
    hihat_times = [time for time, _ in hfc_onsets]
    hihat_intervals = np.diff(hihat_times)
    
    # Look for pattern establishment (several hi-hats with similar intervals)
    consistent_hihat_time = first_hihat
    for i in range(len(hihat_intervals) - 2):
        if abs(hihat_intervals[i] - hihat_intervals[i+1]) < 0.05:  # within 50ms tolerance
            consistent_hihat_time = hihat_times[i]
            break
else:
    first_hihat = None
    consistent_hihat_time = None

if energy_onsets:
    # Sort by confidence to find the first significant beat (likely kick)
    sorted_beats = sorted(energy_onsets, key=lambda x: x[1], reverse=True)
    first_significant_beat = sorted_beats[0][0]
    first_beat = min([time for time, _ in energy_onsets])
else:
    first_significant_beat = None
    first_beat = None

if notes:
    first_note = min(notes)
    # Identify potential vocal entry (look for multiple notes within a short time frame)
    notes_array = np.array(notes)
    potential_vocal = first_note
    for i in range(len(notes) - 3):
        if notes[i+3] - notes[i] < 1.0:  # Multiple notes within 1 second
            potential_vocal = notes[i]
            break
else:
    first_note = None
    potential_vocal = None

# Print timing information
print("\nDetected Entry Points:")
print(f"First hi-hat detected: {first_hihat:.2f}s" if first_hihat is not None else "No hi-hats detected")
print(f"Consistent hi-hat pattern established: {consistent_hihat_time:.2f}s" if consistent_hihat_time is not None else "No consistent pattern found")
print(f"First beat detected: {first_beat:.2f}s" if first_beat is not None else "No beats detected")
print(f"First significant beat (kick drum): {first_significant_beat:.2f}s" if first_significant_beat is not None else "No significant beat detected")
print(f"First note detected: {first_note:.2f}s" if first_note is not None else "No notes detected")
print(f"Potential vocal entry: {potential_vocal:.2f}s" if potential_vocal is not None else "No potential vocal entry detected")

# Create a visualization of entry points
plt.figure(figsize=(16, 8))
librosa.display.waveshow(y, sr=sr, alpha=0.6)
plt.title('Song Elements Entry Points - Last Dance with Mary Jane')

colors = plt.cm.tab10.colors

# Mark hi-hat entry
if first_hihat is not None:
    plt.axvline(x=first_hihat, color=colors[0], linestyle='--', linewidth=2, alpha=0.8, label=f'First Hi-hat ({first_hihat:.2f}s)')
if consistent_hihat_time is not None:
    plt.axvline(x=consistent_hihat_time, color=colors[0], linestyle='-', linewidth=2, alpha=0.8, label=f'Hi-hat Pattern ({consistent_hihat_time:.2f}s)')

# Mark beat entries
if first_beat is not None:
    plt.axvline(x=first_beat, color=colors[1], linestyle='--', linewidth=2, alpha=0.8, label=f'First Beat ({first_beat:.2f}s)')
if first_significant_beat is not None:
    plt.axvline(x=first_significant_beat, color=colors[1], linestyle='-', linewidth=2, alpha=0.8, label=f'Main Beat Entry ({first_significant_beat:.2f}s)')

# Mark potential vocal entry
if potential_vocal is not None:
    plt.axvline(x=potential_vocal, color=colors[2], linestyle='-', linewidth=2, alpha=0.8, label=f'Vocal Entry ({potential_vocal:.2f}s)')
elif first_note is not None:
    plt.axvline(x=first_note, color=colors[2], linestyle='--', linewidth=2, alpha=0.8, label=f'First Note ({first_note:.2f}s)')

plt.grid(alpha=0.3)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.tight_layout()
plt.savefig('song_element_entries.png')
print(f"Created visualization: song_element_entries.png")

# Create a zoomed version with element markers for each section
# Divide the intro into segments based on element entries
fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

# Time points to analyze (start, before vocals, after vocals, full beat)
times = [0.0]
if first_hihat is not None:
    times.append(max(0, first_hihat - 0.2))
if potential_vocal is not None:
    times.append(max(0, potential_vocal - 0.2))
if first_significant_beat is not None:
    times.append(max(0, first_significant_beat - 0.2))
times.append(duration)

# Ensure times are unique and sorted
times = sorted(list(set(times)))

# Define a function to draw a segment
def plot_segment(ax, start, end, title):
    segment_duration = end - start
    segment_samples = int(segment_duration * sr)
    segment_start_sample = int(start * sr)
    
    if segment_samples > 0:
        segment = y[segment_start_sample:segment_start_sample + segment_samples]
        times_segment = np.linspace(start, end, len(segment))
        ax.plot(times_segment, segment)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.set_ylabel("Amplitude")
        
        # Mark events in this segment
        for t, _ in hfc_onsets:
            if start <= t <= end:
                ax.axvline(x=t, color='yellow', linestyle='--', alpha=0.5)
        for t, _ in energy_onsets:
            if start <= t <= end:
                ax.axvline(x=t, color='red', linestyle='-', alpha=0.5)
        for t in notes:
            if start <= t <= end:
                ax.axvline(x=t, color='blue', linestyle=':', alpha=0.5)

# Plot segments based on detected entry points
segment_titles = []
if len(times) >= 4:
    segment_titles = [
        "Intro Silence",
        "Hi-hat Section",
        "Vocal Entry",
        "Full Beat"
    ]
else:
    # Generic titles if not all elements were detected
    segment_titles = ["Section " + str(i+1) for i in range(len(times)-1)]

for i in range(min(3, len(times)-1)):
    plot_segment(axes[i], times[i], times[i+1], segment_titles[i])

plt.xlabel("Time (s)")
plt.tight_layout()
plt.savefig('song_sections.png')
print(f"Created visualization: song_sections.png")

print("\nDetection results:")
print(f"HFC onsets (hi-hats): {len(hfc_onsets)}")
print(f"Energy onsets (beats/kicks): {len(energy_onsets)}")
print(f"Notes detected: {len(notes)}")

print("\nAnalysis complete!")