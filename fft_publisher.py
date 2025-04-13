import time
import json
import os
import paho.mqtt.client as mqtt
from datetime import datetime
import numpy as np
import pyaudio
import aubio
from rich.console import Console
import scipy.fftpack

# Audio parameters
BUFFER_SIZE = 512
SAMPLE_RATE = 44100
CHANNELS = 1

# MQTT parameters
MQTT_BROKER = os.environ.get("MQTT_BROKER", "0.0.0.0")
MQTT_PORT = 1883
MQTT_TOPIC = "beatzero/spectrum_data"
MQTT_CLIENT_ID = f"beatzero-fft-publisher-{int(time.time())}"

# Define frequency bands for analysis
FREQ_BANDS = [
    (20, 80),  # Sub-bass (very low)
    (80, 250),  # Bass
    (250, 500),  # Low-mids
    (500, 1000),  # Mids
    (1000, 2000),  # Upper-mids
    (2000, 3000),  # Presence
    (3000, 4000),  # Brilliance
    (4000, 8000),  # Air/Ultra high
]

# Set up Rich console
console = Console()

# Initialize MQTT client
client = mqtt.Client(client_id=MQTT_CLIENT_ID)

# Initialize PyAudio
p = pyaudio.PyAudio()

# List available devices
console.print("[bold]Available audio devices:[/bold]")
for i in range(p.get_device_count()):
    device_info = p.get_device_info_by_index(i)
    console.print(f"Device {i}: {device_info['name']}")

# Open input stream from microphone
stream = p.open(
    format=pyaudio.paFloat32,
    channels=CHANNELS,
    rate=SAMPLE_RATE,
    input=True,
    frames_per_buffer=BUFFER_SIZE,
)

# Initialize onset detectors (keeping these for beat detection alongside FFT)
ONSET_METHODS = ["energy", "hfc", "complex", "phase", "specflux"]

onset_detectors = {}
for method in ONSET_METHODS:
    detector = aubio.onset(method, BUFFER_SIZE, BUFFER_SIZE, SAMPLE_RATE)
    detector.set_threshold(0.5)  # Higher threshold to reduce false positives
    detector.set_silence(-50)  # Less sensitive to quiet sounds
    detector.set_minioi_ms(100)  # Larger minimum interval between onsets (100ms)
    onset_detectors[method] = detector

# Initialize tempo detection
tempo_detector = aubio.tempo("specdiff", BUFFER_SIZE, BUFFER_SIZE, SAMPLE_RATE)
tempo_detector.set_threshold(0.5)  # Higher threshold for tempo detection

# Initialize note detection
note_detector = aubio.notes("default", BUFFER_SIZE, BUFFER_SIZE, SAMPLE_RATE)
note_detector.set_silence(-30)  # Less sensitive to quiet notes
note_detector.set_minioi_ms(100)  # Larger minimum interval between notes

# Smoothing for frequency band energies
smoothed_band_energy = np.zeros(len(FREQ_BANDS))
smoothing_factor = 0.7  # Higher = more smoothing, must be < 1.0


def connect_mqtt():
    """Connect to MQTT broker"""
    try:
        client.connect(MQTT_BROKER, MQTT_PORT)
        client.loop_start()
        console.print(
            f"[bold green]Connected to MQTT broker at {MQTT_BROKER}:{MQTT_PORT}[/bold green]"
        )
        return True
    except Exception as e:
        console.print(f"[bold red]Failed to connect to MQTT broker: {e}[/bold red]")
        return False


def publish_data(data):
    """Publish data to MQTT topic"""
    try:
        msg = json.dumps(data)
        result = client.publish(MQTT_TOPIC, msg)
        status = result[0]
        if status == 0:
            return True
        else:
            console.print(
                f"[bold red]Failed to send message to topic {MQTT_TOPIC}[/bold red]"
            )
            return False
    except Exception as e:
        console.print(f"[bold red]MQTT publish error: {e}[/bold red]")
        return False


def calculate_band_energy(fft_data, freqs):
    """Calculate energy in each frequency band"""
    band_energy = []

    for low_freq, high_freq in FREQ_BANDS:
        # Find indices corresponding to this frequency band
        indices = np.where((freqs >= low_freq) & (freqs <= high_freq))[0]

        if len(indices) > 0:
            # Calculate average energy in this band
            energy = np.mean(np.abs(fft_data[indices]))
            # Apply some normalization to get a reasonable range (this might need tuning)
            energy = min(1.0, energy * 5.0)  # Assuming energy typically < 0.2
            band_energy.append(float(energy))
        else:
            band_energy.append(0.0)

    return band_energy


# Main processing loop
console.print("[bold green]BeatZero FFT Spectrum Publisher[/bold green]")
console.print(
    "[bold]Analyzing audio spectrum and publishing to MQTT... Press Ctrl+C to stop[/bold]"
)

# Connect to MQTT broker
if not connect_mqtt():
    console.print("[bold red]Exiting due to MQTT connection failure[/bold red]")
    exit(1)

# Apply a window function to reduce spectral leakage
hann_window = np.hanning(BUFFER_SIZE)

try:
    while True:
        # Read audio data
        audiobuffer = stream.read(BUFFER_SIZE, exception_on_overflow=False)
        signal = np.frombuffer(audiobuffer, dtype=np.float32)

        # Apply window function to the signal
        windowed_signal = signal * hann_window

        # Perform FFT
        fft_data = scipy.fftpack.fft(windowed_signal)

        # Calculate frequency bins
        freqs = scipy.fftpack.fftfreq(len(fft_data), 1.0 / SAMPLE_RATE)

        # We only need the positive half of the spectrum (due to symmetry)
        positive_freq_indices = np.where(freqs >= 0)[0]
        freqs = freqs[positive_freq_indices]
        fft_data = fft_data[positive_freq_indices]

        # Calculate energy in each frequency band
        band_energy = calculate_band_energy(fft_data, freqs)

        # Apply smoothing to band energy
        smoothed_band_energy = smoothing_factor * smoothed_band_energy + (
            1 - smoothing_factor
        ) * np.array(band_energy)

        # Process the audio data
        timestamp = datetime.now().isoformat()

        # Detect onsets with all methods (keeping for beat detection)
        onset_data = {}
        for method, detector in onset_detectors.items():
            is_beat = bool(detector(signal))
            descriptor = float(detector.get_descriptor())
            threshold = float(detector.get_threshold())
            onset_data[method] = {
                "is_beat": is_beat,
                "descriptor": descriptor,
                "threshold": threshold,
            }

        # Check tempo detector
        is_tempo_beat = bool(tempo_detector(signal))
        bpm = float(tempo_detector.get_bpm())

        # Detect notes
        note_array = note_detector(signal)
        has_note = bool(note_array.size > 0 and note_array[0] > 0)

        # Calculate volume
        volume = float(np.sqrt(np.mean(signal**2)))

        # Detect kick drum (using energy detector)
        kick_detected = bool(
            onset_data.get("energy", {}).get("is_beat", False)
            and smoothed_band_energy[0] > 0.5  # Higher threshold for kick detection
        )

        # Detect hi-hat (using high frequency energy)
        hihat_detected = bool(
            smoothed_band_energy[7] > 0.5
            and onset_data.get("hfc", {}).get("is_beat", False)
        )

        # Create data packet
        data_packet = {
            "timestamp": timestamp,
            "onsets": onset_data,
            "tempo": {"is_beat": is_tempo_beat, "bpm": bpm},
            "note_detected": has_note,
            "volume": volume,
            "kick_detected": kick_detected,
            "hihat_detected": hihat_detected,
            "spectrum": {
                "band_energy": [float(e) for e in smoothed_band_energy],
                "band_ranges": FREQ_BANDS,
            },
        }

        # Publish to MQTT
        publish_data(data_packet)

        # Console feedback (minimal to reduce CPU usage)
        if (
            kick_detected
            or hihat_detected
            or any(data["is_beat"] for data in onset_data.values())
            or any(
                e > 0.7 for e in smoothed_band_energy
            )  # Only report higher energy events
        ):
            # Format spectrum data for display
            spectrum_str = " ".join(f"{e:.2f}" for e in smoothed_band_energy)
            console.print(
                f"[{timestamp}] BPM={bpm:.1f}, Kick={kick_detected}, HiHat={hihat_detected}, Spectrum=[{spectrum_str}]"
            )

        # Small delay to reduce CPU usage
        time.sleep(0.01)

except KeyboardInterrupt:
    console.print("[bold red]Stopping...[/bold red]")
finally:
    # Clean up
    client.loop_stop()
    client.disconnect()
    stream.stop_stream()
    stream.close()
    p.terminate()
    console.print("[bold green]Stopped[/bold green]")
