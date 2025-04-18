import time
import json
import os
import paho.mqtt.client as mqtt
from datetime import datetime
import numpy as np
import pyaudio
import aubio
from rich.console import Console

# Audio parameters
BUFFER_SIZE = 512
SAMPLE_RATE = 44100
CHANNELS = 1

# MQTT parameters
MQTT_BROKER = os.environ.get("MQTT_BROKER", "localhost")
MQTT_PORT = 1883
MQTT_TOPIC = "beatzero/music_detection"
MQTT_CLIENT_ID = f"beatzero-publisher-{int(time.time())}"

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

# Initialize onset detectors
ONSET_METHODS = ["energy", "hfc", "complex", "phase", "specflux"]

onset_detectors = {}
for method in ONSET_METHODS:
    detector = aubio.onset(method, BUFFER_SIZE, BUFFER_SIZE, SAMPLE_RATE)
    detector.set_threshold(0.1)
    detector.set_silence(-70)
    detector.set_minioi_ms(40)
    onset_detectors[method] = detector

# Initialize pitch detection
pitch_detector = aubio.pitch("yin", BUFFER_SIZE, BUFFER_SIZE, SAMPLE_RATE)
pitch_detector.set_unit("Hz")
pitch_detector.set_silence(-40)

# Initialize tempo detection
tempo_detector = aubio.tempo("specdiff", BUFFER_SIZE, BUFFER_SIZE, SAMPLE_RATE)
tempo_detector.set_threshold(0.2)

# Initialize note detection
note_detector = aubio.notes("default", BUFFER_SIZE, BUFFER_SIZE, SAMPLE_RATE)
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


# Main processing loop
console.print("[bold green]BeatZero MQTT Publisher[/bold green]")
console.print(
    "[bold]Listening for music and publishing to MQTT... Press Ctrl+C to stop[/bold]"
)

# Connect to MQTT broker
if not connect_mqtt():
    console.print("[bold red]Exiting due to MQTT connection failure[/bold red]")
    exit(1)

try:
    while True:
        # Read audio data
        audiobuffer = stream.read(BUFFER_SIZE, exception_on_overflow=False)
        signal = np.frombuffer(audiobuffer, dtype=np.float32)

        # Process the audio data
        timestamp = datetime.now().isoformat()

        # Detect onsets with all methods
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

        # Detect pitch
        pitch = float(pitch_detector(signal)[0])
        pitch_confidence = float(pitch_detector.get_confidence())

        # Detect notes
        note_array = note_detector(signal)
        has_note = bool(note_array.size > 0 and note_array[0] > 0)

        # Calculate volume
        volume = float(np.sqrt(np.mean(signal**2)))

        # Detect kick drum (using energy detector)
        kick_detected = bool(onset_data.get("energy", {}).get("is_beat", False))

        # Detect hi-hat (using pitch frequency range)
        hihat_detected = bool(4000 <= pitch <= 8000 and pitch_confidence > 0.07)

        # Create data packet
        data_packet = {
            "timestamp": timestamp,
            "onsets": onset_data,
            "tempo": {"is_beat": is_tempo_beat, "bpm": bpm},
            "pitch": {"value": pitch, "confidence": pitch_confidence},
            "note_detected": has_note,
            "volume": volume,
            "kick_detected": kick_detected,
            "hihat_detected": hihat_detected,
        }

        # Publish to MQTT
        publish_data(data_packet)

        # Console feedback (minimal to reduce CPU usage)
        if (
            kick_detected
            or hihat_detected
            or any(data["is_beat"] for data in onset_data.values())
        ):
            console.print(
                f"[{timestamp}] Published beat detection data: BPM={bpm:.1f}, Kick={kick_detected}, HiHat={hihat_detected}"
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
