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
    frames_per_buffer=BUFFER_SIZE,
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
    "specflux": "Spectral Flux (overall)",
}

# Aubio classes reference
# aubio.notes(method="default", buf_size=1024, hop_size=512, samplerate=44100)
# Note detection

onset_detectors = {}

for method in ONSET_METHODS:
    detector = aubio.onset(method, BUFFER_SIZE, BUFFER_SIZE, SAMPLE_RATE)
    detector.set_threshold(0.1)  # Lower threshold for more sensitivity
    detector.set_silence(-70)  # Even lower silence threshold
    detector.set_minioi_ms(40)  # Slightly shorter minimum interval between onsets
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

# We no longer need to store beat timestamps as we use aubio's BPM detection

# This function has been removed as we're now using aubio's BPM detection


# Simple function to create a real-time display table
def create_audio_table(onset_results, pitch, pitch_confidence, note, volume, aubio_bpm):
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
            f"[{method_color}]{method_desc:<15}[/{method_color}]", f"{beat_display:>15}"
        )

    # Note detection
    note_detected = note.size > 0 and note[0] > 0

    # Create a simple visualization with ASCII blocks based on pitch
    # Map the pitch to our 8 blocks (we know pitch is reliable)
    note_blocks = ""

    # Use pitch frequency for the visualization
    # Map expanded musical range (roughly 20Hz-5000Hz) to 8 blocks
    # Each block represents a range of frequencies
    freq_ranges = [
        (20, 80),  # Sub-bass (very low)
        (80, 250),  # Bass
        (250, 500),  # Low-mids
        (500, 1000),  # Mids
        (1000, 2000),  # Upper-mids
        (2000, 3000),  # Presence
        (3000, 4000),  # Brilliance
        (4000, 8000),  # Air/Ultra high
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

    # Add frequency visualization - just the blocks without the note detection status
    table.add_row("Freq Bands".ljust(15), f"[blue]{note_blocks}[/blue]".rjust(15))

    # Kick drum detection - using energy detector
    # Energy is good at detecting low frequency transients like kick drums
    kick_detected = False
    if "energy" in onset_results:
        is_energy_beat = onset_results["energy"][0]
        if is_energy_beat:
            kick_detected = True

    # Create a visual indicator for kick drum detection
    if kick_detected:
        kick_indicator = "[bold red]⚫ KICK ⚫[/bold red]"
    else:
        kick_indicator = "[dim]----------[/dim]"
    table.add_row("Kick".ljust(15), f"{kick_indicator}".rjust(15))

    # Hi-hat detection - using expanded 4500-6000 Hz frequency range
    # This targets the primary hi-hat frequencies we identified (centered on 5211 Hz)
    # But expanded to catch the broader hi-hat spectrum (4716 Hz - 5642 Hz)

    # Check if the current pitch is in the hi-hat frequency range
    # Lower confidence threshold based on our sample analysis
    hihat_detected = 4000 <= pitch <= 8000 and pitch_confidence > 0.07

    # Create a more visual indicator for hi-hat detection
    if hihat_detected:
        hihat_indicator = f"[bold yellow]✧✧ +++ ✧✧[/bold yellow]"
    else:
        hihat_indicator = "[dim]----------[/dim]"
    table.add_row("Hi-Hat".ljust(15), f"{hihat_indicator}".rjust(15))

    # BPM from aubio's estimation
    aubio_bpm_color = "green" if aubio_bpm > 10 else "dim"
    table.add_row(
        "BPM".ljust(15),
        f"[{aubio_bpm_color}]{aubio_bpm:6.1f}[/{aubio_bpm_color}]".rjust(15),
    )

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

            # Check tempo detector
            is_tempo_beat = tempo_detector(signal)

            # Get aubio's built-in tempo estimation
            aubio_bpm = tempo_detector.get_bpm()

            # Detect pitch
            pitch = pitch_detector(signal)[0]
            pitch_confidence = pitch_detector.get_confidence()

            # Detect notes
            note = note_detector(signal)

            # We no longer need spectral analysis or energy band calculations
            # as we're only using the note detection and pitch information

            # Calculate overall volume level (used for hi-hat detection criteria)
            volume = np.sqrt(np.mean(signal**2))

            # Always update the display in real-time
            table = create_audio_table(
                onset_results, pitch, pitch_confidence, note, volume, aubio_bpm
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
