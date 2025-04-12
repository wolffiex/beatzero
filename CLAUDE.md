# BeatZero: Real-time Music Analysis and Visualization

BeatZero is a project that performs real-time audio analysis for beat detection, frequency analysis, and music element recognition. It uses PyAudio for audio capture, Aubio for audio processing, and provides multiple visualization options.

## Project Components

- **music_detector.py**: CLI-based music analysis with rich text visualization
- **mqtt_publisher.py**: Publishes audio analysis data to MQTT
- **mqtt_subscriber.py**: Subscribes to audio data for text-based visualization
- **pygame_visualizer.py**: Graphical visualization of music elements

## Setup Requirements

```bash
# Install required Python packages
pip install -r requirements.txt

# Install and start Mosquitto MQTT broker (required for MQTT features)
sudo apt-get install -y mosquitto mosquitto-clients
sudo systemctl start mosquitto
```

## Running the Applications

### CLI Music Detector
```bash
python music_detector.py
```

### MQTT Publisher & Subscriber
```bash
# Terminal 1: Start the MQTT publisher
python mqtt_publisher.py

# Terminal 2: Start the MQTT subscriber
python mqtt_subscriber.py
```

### Pygame Visualizer
```bash
# Start the Pygame visualizer (requires MQTT publisher running)
python pygame_visualizer.py
```

## Key Features

- Real-time onset (beat) detection using multiple methods
- Pitch and note detection
- BPM (tempo) estimation
- Frequency band analysis
- Kick drum and hi-hat detection
- Multiple visualization options:
  - Rich text-based console display
  - MQTT-based distributed visualization
  - Pygame graphical visualization with animations

## Code Overview

- Audio is captured in real-time using PyAudio
- Audio processing uses Aubio library for:
  - Onset detection (various methods: energy, HFC, complex, phase, specflux)
  - Pitch and note detection
  - Tempo (BPM) analysis
- Visualization systems process the analyzed data for display

## Development Guidelines

- Keep consistent audio parameters across components (BUFFER_SIZE, SAMPLE_RATE)
- Ensure MQTT broker is running before starting publisher/subscriber
- For new visualizations, follow the same data structure used in existing components