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

# Define frequency bands for analysis (split high frequencies into two bands)
FREQ_BANDS = [
    (80, 250),  # Bass
    (250, 500),  # Low-mids
    (500, 1000),  # Mids
    (1000, 2000),  # Upper-mids
    (2000, 3000),  # Presence
    (3000, 4000),  # Brilliance
    (4000, 5000),  # High (4-5kHz)
    (5000, 8000),  # Ultra high (5-8kHz)
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
smoothed_band_energy = np.zeros(len(FREQ_BANDS))  # Now 8 bands
smoothing_factor = 0.2  # Higher = more smoothing, must be < 1.0

# Volume-based gain adjustment
volume_history = []
MAX_VOLUME_HISTORY = 100  # ~1 second at ~100 frames per second
GAIN_THRESHOLD = 0.03  # Base threshold for gain adjustment
MAX_GAIN = 0.8  # Maximum gain to prevent pegging at 1.0
MIN_GAIN = 0.2  # Minimum gain level
gain_multiplier = 0.5  # Start with a moderate gain


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
    """Calculate energy in each frequency band with adaptive scaling and bass attenuation"""
    band_energy = []
    # Calculate raw energies first to determine adaptive scaling
    raw_energies = []

    for low_freq, high_freq in FREQ_BANDS:
        # Find indices corresponding to this frequency band
        indices = np.where((freqs >= low_freq) & (freqs <= high_freq))[0]

        if len(indices) > 0:
            # Calculate average energy in this band
            energy = np.mean(np.abs(fft_data[indices]))
            raw_energies.append(energy)
        else:
            raw_energies.append(0.0)

    # Attenuate bass before calculating max energy for better balance
    # First band is bass (80-250Hz) - attenuate it
    if len(raw_energies) > 0:
        # Store original bass energy
        original_bass = raw_energies[0]
        # Apply attenuation to bass for max energy calculation
        raw_energies[0] *= 0.5  # Reduce bass by 50%

    # Calculate adaptive scaling factor based on maximum energy
    max_energy = max(raw_energies) if raw_energies else 0
    if max_energy > 0:
        # Scale so that the max value will be around 0.7-0.8 but not saturate
        adaptive_scale = 0.8 / max_energy
    else:
        adaptive_scale = 1.0

    # Restore original bass energy before final scaling
    if len(raw_energies) > 0:
        raw_energies[0] = original_bass

    # Apply the adaptive scaling to all bands
    for i, energy in enumerate(raw_energies):
        # Apply extra attenuation to bass (80-250Hz)
        if i == 0:
            # Apply scaling and bass attenuation but ensure we don't exceed 1.0
            scaled_energy = min(1.0, energy * adaptive_scale * 0.5)
        else:
            # Apply normal scaling for other bands
            scaled_energy = min(1.0, energy * adaptive_scale)
        band_energy.append(float(scaled_energy))

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

# Track publish rate
publish_count = 0
last_report_time = time.time()

try:
    while True:
        # Read audio data
        audiobuffer = stream.read(BUFFER_SIZE, exception_on_overflow=False)
        signal = np.frombuffer(audiobuffer, dtype=np.float32)

        # Calculate volume for gain adjustment
        volume = float(np.sqrt(np.mean(signal**2)))

        # Update volume history
        volume_history.append(volume)
        if len(volume_history) > MAX_VOLUME_HISTORY:
            volume_history.pop(0)

        # Calculate average volume over the last ~1 second
        avg_volume = sum(volume_history) / len(volume_history)

        # Dynamically adjust gain multiplier based on average volume
        # Lower volume = lower gain multiplier (makes display less active)
        # Higher volume = higher gain multiplier (makes display more responsive)
        if avg_volume < GAIN_THRESHOLD:
            # Calculate a gain that scales with volume (approaches MIN_GAIN at very low volumes)
            target_gain = MIN_GAIN + (avg_volume / GAIN_THRESHOLD) * (
                MAX_GAIN - MIN_GAIN
            )
            # Apply a faster reduction rate when volume is very low
            reduction_factor = 0.80 if avg_volume < (GAIN_THRESHOLD * 0.5) else 0.95
            # Smooth transition to avoid sudden changes
            gain_multiplier = (
                reduction_factor * gain_multiplier
                + (1 - reduction_factor) * target_gain
            )
        else:
            # For normal/loud volumes, gradually approach MAX_GAIN
            # but at a slower rate to prevent rapid jumps
            old_gain = gain_multiplier

            # Calculate how close to MAX_GAIN we want to be based on volume
            volume_factor = min(
                1.0, (avg_volume - GAIN_THRESHOLD) / (GAIN_THRESHOLD * 2)
            )
            target_gain = MIN_GAIN + (MAX_GAIN - MIN_GAIN) * (0.5 + volume_factor * 0.5)

            # Limit the maximum change per cycle
            max_change = 0.01
            if target_gain > old_gain:
                new_gain = min(target_gain, old_gain + max_change)
            else:
                new_gain = max(target_gain, old_gain - max_change)

            # Ensure gain stays within limits
            gain_multiplier = max(MIN_GAIN, min(MAX_GAIN, new_gain))

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

        # We no longer apply gain multiplier to the actual data
        # Gain multiplier is now only used to control visualization state

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

            # No longer applying gain multiplier to descriptors
            # Let visualizer handle display control based on gain

            onset_data[method] = {
                "is_beat": is_beat,
                "descriptor": descriptor,  # Using unadjusted descriptor
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

        # Detect kick drum (using energy detector and bass band)
        kick_detected = bool(
            onset_data.get("energy", {}).get("is_beat", False)
            and smoothed_band_energy[0] > 0.4  # Lower threshold due to bass attenuation
        )

        # Detect hi-hat (using the 4-8kHz ultra high frequency band)
        hihat_detected = bool(
            smoothed_band_energy[6] > 0.7  # Ultra high band (4-8kHz)
            and onset_data.get("hfc", {}).get("is_beat", False)
        )

        # Create data packet
        data_packet = {
            "timestamp": timestamp,
            "onsets": onset_data,
            "tempo": {"is_beat": is_tempo_beat, "bpm": bpm},
            "note_detected": has_note,
            "volume": volume,
            "avg_volume": avg_volume,
            "gain_multiplier": gain_multiplier,
            "kick_detected": kick_detected,
            "hihat_detected": hihat_detected,
            "spectrum": {
                "band_energy": [float(e) for e in smoothed_band_energy],
                "band_ranges": FREQ_BANDS,
            },
        }

        # Publish to MQTT
        publish_data(data_packet)
        publish_count += 1

        # Log publish rate every 5 seconds
        current_time = time.time()
        if current_time - last_report_time >= 5:
            rate = publish_count / (current_time - last_report_time)
            console.print(f"Publishing rate: {rate:.2f} messages/second")
            publish_count = 0
            last_report_time = current_time

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
