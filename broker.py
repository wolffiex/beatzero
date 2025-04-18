import time
import json
import os
from datetime import datetime
import numpy as np
import pyaudio
import aubio
import paho.mqtt.client as mqtt
from rich.console import Console
from collections import deque

# Audio parameters
BUFFER_SIZE = 1024  # Larger buffer for better frequency resolution
HOP_SIZE = 512  # Smaller hop for responsive timing
SAMPLE_RATE = 44100
CHANNELS = 1

# MQTT parameters
MQTT_BROKER = os.environ.get("MQTT_BROKER", "localhost")
MQTT_PORT = 1883
MQTT_TOPIC = "beatzero/music_detection"
MQTT_CLIENT_ID = f"beatzero-broker-{int(time.time())}"

# 30 FPS publishing rate
FRAME_TIME = 1 / 30  # 33.3ms for 30fps

# Minimum volume threshold (baseline noise floor)
MIN_VOLUME_THRESHOLD = 0.01

# Set up Rich console
console = Console()

# Initialize MQTT client with API version 2
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=MQTT_CLIENT_ID)

# Initialize PyAudio
p = pyaudio.PyAudio()

# Open input stream from microphone
stream = p.open(
    format=pyaudio.paFloat32,
    channels=CHANNELS,
    rate=SAMPLE_RATE,
    input=True,
    frames_per_buffer=HOP_SIZE,
)

# Initialize onset detectors
# Each method specializes in different aspects of audio detection:
# - energy: Detects loudness changes, good for percussion
# - hfc: High Frequency Content, good for cymbals/hi-hats
# - complex: Detects tonal changes, better for harmonic instruments
# - phase: Detects timing/phase changes, good for transients
# - specflux: Overall spectral changes, general-purpose
# - wphase: Weighted phase deviation
# - mkl: Modified Kullback-Liebler divergence
# - kl: Kullback-Liebler divergence
ONSET_METHODS = ["energy", "hfc", "complex", "phase", "specflux", "wphase", "mkl", "kl"]

onset_detectors = {}
for method in ONSET_METHODS:
    detector = aubio.onset(method, BUFFER_SIZE, HOP_SIZE, SAMPLE_RATE)
    detector.set_threshold(0.1)
    detector.set_silence(-70)
    detector.set_minioi_ms(40)
    onset_detectors[method] = detector

# Initialize pitch detection
pitch_detector = aubio.pitch("yin", BUFFER_SIZE, HOP_SIZE, SAMPLE_RATE)
pitch_detector.set_unit("Hz")
pitch_detector.set_silence(-60)

# Initialize tempo detection
tempo_detector = aubio.tempo("specdiff", BUFFER_SIZE, HOP_SIZE, SAMPLE_RATE)
tempo_detector.set_threshold(0.2)
tempo_detector.set_silence(-40)

# Initialize note detection
note_detector = aubio.notes("default", BUFFER_SIZE, HOP_SIZE, SAMPLE_RATE)
note_detector.set_silence(-40)
note_detector.set_minioi_ms(50)


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


def process_audio_buffer(signal):
    """Process a single audio buffer and return detection data"""
    # Process current audio buffer
    data = {}

    # Detect onsets with all methods
    onset_data = {}
    for method, detector in onset_detectors.items():
        onset_data[method] = bool(detector(signal))
    data["onsets"] = onset_data

    # Check tempo detector
    is_tempo_beat = bool(tempo_detector(signal))
    bpm = float(tempo_detector.get_bpm())
    data["bpm"] = bpm
    data["tempo_beat"] = is_tempo_beat

    # Detect pitch
    pitch = float(pitch_detector(signal)[0])
    pitch_confidence = float(pitch_detector.get_confidence())
    data["pitch"] = {"value": pitch, "confidence": pitch_confidence}

    # Detect notes
    note_array = note_detector(signal)
    # Filter out zeros using NumPy and convert to a list
    data["notes"] = note_array[note_array != 0].tolist()

    # Calculate volume
    volume = float(np.sqrt(np.mean(signal**2)))
    data["volume"] = volume

    # Always add to volume history
    volume_history.append(volume)

    return data


def combine_packets(packets, volume_history):
    """Combine multiple data packets into a single packet for publishing"""
    if not packets:
        return None

    result = {}

    for method in ONSET_METHODS:
        # OR together all is_beat flags
        is_beat = any(packet["onsets"][method] for packet in packets)
        result[method] = is_beat

    # Combine tempo data (OR the tempo_beat, average recent BPM values)
    is_tempo_beat = any(packet["tempo_beat"] for packet in packets)
    # Calculate average BPM to smooth out fluctuations
    # Normalize each BPM value individually (halve values over 120)
    normalized_bpm_values = [
        bpm / 2 if bpm > 120 else bpm
        for packet in packets
        if (bpm := packet["bpm"]) > 0
    ]

    if normalized_bpm_values:
        result["bpm"] = float(sum(normalized_bpm_values) / len(normalized_bpm_values))
    else:
        result["bpm"] = 0.0
    result["tempo_beat"] = is_tempo_beat

    # Find the pitch with highest confidence from all packets
    best_pitch = 0
    best_confidence = 0

    for packet in packets:
        pitch_value = packet["pitch"]["value"]
        pitch_confidence = packet["pitch"]["confidence"]

        if pitch_confidence > best_confidence:
            best_confidence = pitch_confidence
            best_pitch = pitch_value

    # Store the best pitch data
    result["pitch"] = {"value": best_pitch, "confidence": best_confidence}

    # Combine all detected notes from all packets
    combined_notes = set()
    for packet in packets:
        combined_notes.update(packet["notes"])
    result["notes"] = list(combined_notes)

    # Get the maximum volume from all packets in this buffer
    raw_volume = max(packet["volume"] for packet in packets)

    # Get the max volume from history with a minimum floor
    max_vol = max(*volume_history, MIN_VOLUME_THRESHOLD)

    # If volume is below threshold, set to zero
    if raw_volume < MIN_VOLUME_THRESHOLD:
        result["volume"] = 0.0
    else:
        # Otherwise normalize between 0 and 1
        normalized = raw_volume / max_vol
        result["volume"] = min(1.0, max(0.0, normalized))

    return result


# Main processing loop
console.print("[bold green]BeatZero MQTT Broker[/bold green]")
console.print(
    "[bold]Listening for music and publishing to MQTT at 30fps... Press Ctrl+C to stop[/bold]"
)

# Connect to MQTT broker
if not connect_mqtt():
    console.print("[bold red]Exiting due to MQTT connection failure[/bold red]")
    exit(1)

try:
    # Initialize frame timing
    next_frame = time.time()

    # Buffer to store data packets between publishes
    buffer = deque()

    # Buffer for volume normalization (stores last 100 volume samples)
    volume_history = deque(maxlen=100)

    while True:
        # Read audio data
        audiobuffer = stream.read(HOP_SIZE, exception_on_overflow=False)
        signal = np.frombuffer(audiobuffer, dtype=np.float32)

        # Process the audio data
        data_packet = process_audio_buffer(signal)

        # Store the packet in our buffer
        buffer.append(data_packet)

        # Check if it's time to publish
        current_time = time.time()
        if current_time >= next_frame:
            # Combine all buffered packets
            combined_packet = combine_packets(list(buffer), volume_history)

            # Clear the buffer
            buffer.clear()

            # Publish the combined packet
            if combined_packet:
                publish_data(combined_packet)

            # Calculate next frame time (maintain even 30fps)
            next_frame += FRAME_TIME

        # Small delay to reduce CPU usage
        time.sleep(0.001)

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
