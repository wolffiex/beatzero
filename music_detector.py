import time
import sys
from datetime import datetime
from collections import deque
import numpy as np
import pyaudio
import aubio
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.prompt import Prompt

# Audio parameters
BUFFER_SIZE = 512  # Changed to match what aubio onset expects
SAMPLE_RATE = 44100
CHANNELS = 1

# Set up Rich console for real-time display
console = Console()

# Initialize PyAudio
p = pyaudio.PyAudio()

# List available devices
print("Available audio devices:")
for i in range(p.get_device_count()):
    device_info = p.get_device_info_by_index(i)
    print(f"Device {i}: {device_info['name']}")

# Open input stream from USB microphone
# You may need to specify the device_index based on the output above
stream = p.open(
    format=pyaudio.paFloat32,
    channels=CHANNELS,
    rate=SAMPLE_RATE,
    input=True,
    frames_per_buffer=BUFFER_SIZE
)

# Initialize multiple onset detectors with different methods
# Available onset detection methods: energy, hfc, complex, phase, wphase, mkl, kl, specflux
ONSET_METHODS = ["energy", "hfc", "complex", "phase", "specflux"]

# Method descriptions
METHOD_DESCRIPTIONS = {
    "energy": "Energy (loudness)",
    "hfc": "High Frequency (treble)",
    "complex": "Complex (tonal changes)",
    "phase": "Phase (timing changes)",
    "specflux": "Spectral Flux (overall)"
}

# Aubio classes reference
# aubio.notes(method="default", buf_size=1024, hop_size=512, samplerate=44100)
# Note detection

onset_detectors = {}

for method in ONSET_METHODS:
    detector = aubio.onset(method, BUFFER_SIZE, BUFFER_SIZE, SAMPLE_RATE)
    detector.set_threshold(0.1)  # Lower threshold for more sensitivity
    detector.set_silence(-70)    # Even lower silence threshold
    detector.set_minioi_ms(40)   # Slightly shorter minimum interval between onsets
    onset_detectors[method] = detector

# Initialize aubio pitch detection
# Note: hop_size is set to BUFFER_SIZE (512)
pitch_detector = aubio.pitch("yin", BUFFER_SIZE, BUFFER_SIZE, SAMPLE_RATE)
pitch_detector.set_unit("Hz")
pitch_detector.set_silence(-40)

# Initialize tempo detection (BPM)
tempo_detector = aubio.tempo("specdiff", BUFFER_SIZE, BUFFER_SIZE, SAMPLE_RATE)
tempo_detector.set_threshold(0.2)

# Initialize note detection
note_detector = aubio.notes("default", BUFFER_SIZE, BUFFER_SIZE, SAMPLE_RATE)
note_detector.set_silence(-40)
note_detector.set_minioi_ms(50)  # Minimum interval between notes (ms)

# Initialize FFT for spectral analysis
fft = aubio.fft(BUFFER_SIZE)

# Store beat timestamps for BPM calculation
beat_times = deque(maxlen=20)  # Store last 20 beats for calculation

# Calculate BPM from beat timestamps
def calculate_bpm():
    if len(beat_times) < 4:  # Need more beats for reliable calculation
        return 0.0
    
    # Calculate time differences between consecutive beats
    time_diffs = []
    for i in range(1, len(beat_times)):
        diff = beat_times[i] - beat_times[i-1]
        if 0.2 < diff < 2.0:  # Only consider reasonable intervals (between 30 and 300 BPM)
            time_diffs.append(diff)
    
    if not time_diffs:
        return 0.0
    
    # Calculate median time between beats to avoid outliers
    time_diffs.sort()
    if len(time_diffs) % 2 == 0:
        median_time = (time_diffs[len(time_diffs)//2] + time_diffs[len(time_diffs)//2-1]) / 2
    else:
        median_time = time_diffs[len(time_diffs)//2]
    
    # Convert to BPM
    if median_time > 0:
        raw_bpm = 60.0 / median_time
        
        # Apply sanity check - most music is between 60-180 BPM
        if raw_bpm > 180:
            # Probably double-tempo, halve it
            return raw_bpm / 2
        elif raw_bpm < 60:
            # Probably half-tempo, double it if it makes sense
            if raw_bpm * 2 <= 180:
                return raw_bpm * 2
        return raw_bpm
    return 0.0

# Simple function to create a real-time display table
def create_audio_table(onset_results, pitch, pitch_confidence, note, low, mid, high, volume, bpm, aubio_bpm):
    table = Table(title=f"Real-time Audio Analysis")
    
    table.add_column("Feature", style="cyan")
    table.add_column("Value", style="green", width=15, justify="right")
    
    # Beat detection from all methods
    for method in ONSET_METHODS:
        is_beat, descriptor, threshold = onset_results[method]
        beat_display = "[bold green]YES[/bold green]" if is_beat else "[dim]no[/dim]"
        ratio = descriptor / threshold if threshold > 0 else 0
        
        # Use color coding for the method name based on whether it detected a beat
        method_color = "green" if is_beat else "cyan"
        method_desc = METHOD_DESCRIPTIONS[method]
        
        table.add_row(
            f"[{method_color}]{method_desc:<15}[/{method_color}]", 
            f"{beat_display:>15}"
        )
    
    # Pitch detection
    pitch_color = "yellow" if pitch_confidence > 0.5 else "dim"
    table.add_row("Pitch".ljust(15), f"[{pitch_color}]{pitch:6.1f} Hz[/{pitch_color}]".rjust(15))
    
    # Note detection
    note_detected = note.size > 0 and note[0] > 0
    
    # Create a simple visualization with ASCII blocks based on pitch
    # Map the pitch to our 8 blocks (we know pitch is reliable)
    note_blocks = ""
    
    # Use pitch frequency for the visualization
    # Map expanded musical range (roughly 20Hz-5000Hz) to 8 blocks
    # Each block represents a range of frequencies
    freq_ranges = [
        (20, 80),      # Sub-bass (very low)
        (80, 250),     # Bass
        (250, 500),    # Low-mids
        (500, 1000),   # Mids
        (1000, 2000),  # Upper-mids
        (2000, 3000),  # Presence
        (3000, 4000),  # Brilliance
        (4000, 8000)   # Air/Ultra high
    ]
    
    # Light up the block corresponding to the current pitch
    # Also consider the note detection for coloring
    highlight_note = note_detected and pitch_confidence > 0.4
    for i, (min_freq, max_freq) in enumerate(freq_ranges):
        if min_freq <= pitch < max_freq:
            if highlight_note:
                # Musical note detected in this range - use filled block with bright color
                note_blocks += "■ "  # Filled block
            else:
                # Pitch detected but not confirmed as musical note
                note_blocks += "▣ "  # Half-filled block
        else:
            note_blocks += "□ "  # Empty block
    
    note_status = "[bold blue]YES[/bold blue]" if note_detected else "[dim]no[/dim]"
    table.add_row("Note Detected".ljust(15), f"{note_status:>15}")
    
    # Add frequency visualization - just the blocks without labels
    table.add_row("Freq Bands".ljust(15), f"[blue]{note_blocks}[/blue]".rjust(15))
    
    # Frequency bands with visual meter
    low_meter = "▓" * int(low * 10)
    mid_meter = "▓" * int(mid * 10)
    high_meter = "▓" * int(high * 10)
    
    table.add_row("Low Band".ljust(15), f"[green]{low_meter:10}[/green]".rjust(15))
    table.add_row("Mid Band".ljust(15), f"[yellow]{mid_meter:10}[/yellow]".rjust(15))
    table.add_row("High Band".ljust(15), f"[blue]{high_meter:10}[/blue]".rjust(15))
    
    # Volume
    vol_color = "green" if volume > 0.1 else "yellow" if volume > 0.01 else "red"
    table.add_row("Volume".ljust(15), f"[{vol_color}]{volume:6.4f}[/{vol_color}]".rjust(15))
    
    # BPM - our calculated value
    bpm_color = "green" if bpm > 10 else "dim"
    table.add_row("BPM (calc)".ljust(15), f"[{bpm_color}]{bpm:6.1f}[/{bpm_color}]".rjust(15))
    
    # BPM - aubio's estimation
    aubio_bpm_color = "green" if aubio_bpm > 10 else "dim"
    table.add_row("BPM (aubio)".ljust(15), f"[{aubio_bpm_color}]{aubio_bpm:6.1f}[/{aubio_bpm_color}]".rjust(15))
    
    return table

# Main processing loop
console.print("[bold green]BeatZero Music Detector[/bold green]")
console.print("[bold]Listening for music... Press Ctrl+C to stop[/bold]")

# Use Rich's Live display for real-time updates
try:
    with Live(refresh_per_second=10) as live:
        while True:
            # Read audio data
            audiobuffer = stream.read(BUFFER_SIZE, exception_on_overflow=False)
            signal = np.frombuffer(audiobuffer, dtype=np.float32)
            
            # Detect beat/onset with all methods
            onset_results = {}
            any_beat_detected = False
            for method, detector in onset_detectors.items():
                is_beat = detector(signal)
                descriptor = detector.get_descriptor()
                threshold = detector.get_threshold()
                onset_results[method] = (is_beat, descriptor, threshold)
                if is_beat:
                    any_beat_detected = True
            
            # Record beat time for BPM calculation if any method detected a beat
            if any_beat_detected:
                beat_times.append(time.time())
            
            # Also check tempo detector
            is_tempo_beat = tempo_detector(signal)
            if is_tempo_beat:
                beat_times.append(time.time())
            
            # Calculate current BPM
            bpm = calculate_bpm()
            # Final sanity check
            if bpm > 200:
                bpm = bpm / 2  # Halve if too fast
                
            # Get aubio's built-in tempo estimation
            aubio_bpm = tempo_detector.get_bpm()
            
            # Detect pitch
            pitch = pitch_detector(signal)[0]
            pitch_confidence = pitch_detector.get_confidence()
            
            # Detect notes
            note = note_detector(signal)
            
            # Spectral analysis
            spectrum = fft(signal)
            spectrum_magnitude = spectrum.norm
            
            # Calculate energy in different frequency bands
            bin_size = SAMPLE_RATE / BUFFER_SIZE
            low_band = np.sum(spectrum_magnitude[int(20/bin_size):int(150/bin_size)])
            mid_band = np.sum(spectrum_magnitude[int(150/bin_size):int(2000/bin_size)])
            high_band = np.sum(spectrum_magnitude[int(2000/bin_size):int(10000/bin_size)])
            
            # Normalize energy values
            max_energy = max(low_band, mid_band, high_band)
            if max_energy > 0:
                low_band = low_band / max_energy
                mid_band = mid_band / max_energy
                high_band = high_band / max_energy

            # Calculate overall volume level
            volume = np.sqrt(np.mean(signal**2))
            
            # Always update the display in real-time
            table = create_audio_table(
                onset_results,
                pitch, pitch_confidence,
                note,
                low_band, mid_band, high_band, 
                volume,
                bpm,
                aubio_bpm
            )
            live.update(table)
            
            # Small delay to reduce CPU usage
            time.sleep(0.01)

except KeyboardInterrupt:
    console.print("[bold red]Stopping...[/bold red]")
finally:
    # Clean up
    stream.stop_stream()
    stream.close()
    p.terminate()
    console.print("[bold green]Stopped[/bold green]")
