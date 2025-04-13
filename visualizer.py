import pygame
import time
import json
import os
import paho.mqtt.client as mqtt
from datetime import datetime

# Window setup
WIDTH, HEIGHT = 600, 450  # Increased height to accommodate pitch visualizer
FPS = 30
BACKGROUND_COLOR = (10, 10, 20)

# MQTT parameters
MQTT_BROKER = os.environ.get("MQTT_BROKER", "localhost")
MQTT_PORT = 1883
MQTT_TOPIC = "beatzero/music_detection"
MQTT_CLIENT_ID = f"beatzero-visualizer-{int(time.time())}"

# Global variable to store latest data from MQTT
latest_data = None


# MQTT callbacks
def on_connect(client, userdata, flags, rc, properties=None):
    """Callback for when the client connects to the broker"""
    if rc == 0:
        print(f"Connected to MQTT broker at {MQTT_BROKER}")
        client.subscribe(MQTT_TOPIC)
        print(f"Subscribed to topic: {MQTT_TOPIC}")
    else:
        print(f"Failed to connect to MQTT broker with code: {rc}")


def on_message(client, userdata, msg):
    """Callback for when a message is received from the broker"""
    global latest_data
    try:
        latest_data = json.loads(msg.payload.decode())
    except Exception as e:
        print(f"Error parsing message: {e}")


class OnsetDetector:
    def __init__(self, label):
        self.label = label

        # State management
        self.active_time = 0

        # Used to track the original detector name in MQTT data
        if label.lower() == "high freq":
            self.detector_name = "hfc"
        elif label.lower() == "spectral":
            self.detector_name = "specflux"
        else:
            self.detector_name = label.lower()

    def activate(self, current_time):
        self.active_time = current_time

    def render(self, surface, font, x, y, width, current_time):
        """Draw the detector at the given coordinates"""
        # Draw label
        label_surface = font.render(self.label, True, (200, 200, 200))
        surface.blit(label_surface, (x, y))

        # Calculate color based on intensity
        active = current_time - self.active_time < 60
        if active:
            box_color = (200, 200, 200)
        else:
            box_color = (10, 10, 20)  # Dark gray when inactive

        # Draw indicator box
        box_x = x + width - 30
        box_y = y
        box_size = 20

        pygame.draw.rect(surface, box_color, (box_x, box_y, box_size, box_size))

        # Add border to box
        pygame.draw.rect(
            surface, (100, 100, 120), (box_x, box_y, box_size, box_size), 1
        )
        self.active = False


class NoteVisualizer:
    def __init__(self):
        # Store the 12 pitch classes with their activation times
        self.active_pitch_classes = {}  # {pitch_class (0-11): activation_time}
        self.activation_duration = 200
        self.note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    def update(self, notes):
        # Add current time for each note, mapped to pitch class (0-11)
        current_time = pygame.time.get_ticks()
        for note in notes:
            # Convert MIDI note to pitch class (C=0, C#=1, ..., B=11)
            pitch_class = int(note) % 12
            self.active_pitch_classes[pitch_class] = current_time

    def render(self, surface, font, x, y, width, height):
        """Draw the note visualizer - 12 boxes for the 12 pitch classes"""
        # Get current time to check which notes are still active
        current_time = pygame.time.get_ticks()

        # Draw section title
        title_surface = font.render("Notes", True, (200, 200, 200))
        surface.blit(title_surface, (x, y))

        # Draw background
        panel_y = y + 30
        panel_height = 40
        pygame.draw.rect(surface, (20, 20, 30), (x, panel_y, width, panel_height))
        pygame.draw.rect(surface, (50, 50, 60), (x, panel_y, width, panel_height), 1)

        # Remove pitch classes that have been visible for longer than activation_duration
        pitch_classes_to_remove = []
        for pitch_class, timestamp in self.active_pitch_classes.items():
            if current_time - timestamp > self.activation_duration:
                pitch_classes_to_remove.append(pitch_class)

        for pitch_class in pitch_classes_to_remove:
            self.active_pitch_classes.pop(pitch_class)

        # Use fixed box dimensions
        box_width = 30  # Fixed width for each box
        box_height = 25
        box_spacing = 5  # Space between boxes
        
        # Calculate total width needed for all boxes
        total_boxes_width = (box_width * 12) + (box_spacing * 11)
        
        # Calculate starting x to center all boxes in the panel
        start_x = x + ((width - total_boxes_width) // 2)
        box_y = panel_y + 7

        # Draw all 12 pitch class boxes
        for pitch_class in range(12):
            box_x = start_x + (pitch_class * (box_width + box_spacing))
            
            # Check if this pitch class is active
            is_active = pitch_class in self.active_pitch_classes
            
            # Set box color based on activity
            if is_active:
                box_color = (255, 255, 255)  # White when active
            else:
                box_color = (40, 40, 50)     # Dark gray when inactive
            
            # Draw the box
            pygame.draw.rect(surface, box_color, (box_x, box_y, box_width, box_height))


class PitchVisualizer:
    def __init__(self):
        # Store pitches with their activation times
        self.active_pitches = {}  # {pitch_value: activation_time}
        self.min_pitch = 50  # Min frequency in Hz - widened to prevent low-end clipping
        self.max_pitch = 500  # Max frequency in Hz - widened to prevent high-end clipping
        self.fade_duration = 350  # Fade out duration in ms

    def activate(self, pitch, confidence, current_time):
        if confidence > 0.2 and pitch > 0:
            # Add current pitch with its activation time
            self.active_pitches[pitch] = current_time

    def render(self, surface, font, x, y, width, current_time):
        """Draw the pitch visualizer with boxes that fade out over time"""
        # Draw label
        label_surface = font.render("Pitch", True, (200, 200, 200))
        surface.blit(label_surface, (x, y))

        # Draw background panel
        panel_y = y + 30
        panel_height = 30
        pygame.draw.rect(surface, (20, 20, 30), (x, panel_y, width, panel_height))
        pygame.draw.rect(surface, (50, 50, 60), (x, panel_y, width, panel_height), 1)

        # Find pitches that have expired and should be removed
        pitches_to_remove = []
        for pitch, activation_time in self.active_pitches.items():
            # Check if this pitch's fade duration has expired
            if current_time - activation_time >= self.fade_duration:
                pitches_to_remove.append(pitch)

        # Remove expired pitches
        for pitch in pitches_to_remove:
            self.active_pitches.pop(pitch)

        # Create a separate surface for additive blending of pitch boxes
        pitch_surface = pygame.Surface((width, panel_height), pygame.SRCALPHA)
        pitch_surface.fill((0, 0, 0, 0))  # Transparent background

        # Draw all active pitches with fading onto the pitch surface
        for pitch, activation_time in self.active_pitches.items():
            # Calculate fade factor based on how long since activation
            time_elapsed = current_time - activation_time
            fade_factor = 1.0 - (time_elapsed / self.fade_duration)

            # Clip pitch to our range
            clipped_pitch = max(self.min_pitch, min(self.max_pitch, pitch))

            # Map pitch to position in the bar
            position_ratio = (clipped_pitch - self.min_pitch) / (
                self.max_pitch - self.min_pitch
            )
            box_x = int((position_ratio * (width - 20)))  # Relative to pitch_surface

            # Draw the pitch indicator box
            box_size = 20
            box_y = 5  # Relative to pitch_surface

            # Apply fade to color (255 -> 0 as fade progresses)
            color_value = int(
                255 * fade_factor
            )  # Lower base value for better additive effect
            alpha_value = int(255 * fade_factor)
            box_color = (color_value, color_value, color_value, alpha_value)

            # Draw rectangle to the surface (draw.rect doesn't support special_flags)
            pygame.draw.rect(
                pitch_surface, box_color, (box_x, box_y, box_size, box_size)
            )

        # Blit the combined pitch surface onto the main surface with additive blending
        surface.blit(pitch_surface, (x, panel_y), special_flags=pygame.BLEND_ADD)


def main():
    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("BeatZero Visualizer")
    clock = pygame.time.Clock()

    # Setup font for labels
    font = pygame.font.Font(None, 24)
    small_font = pygame.font.Font(None, 18)

    # Initialize MQTT client with API version 2
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=MQTT_CLIENT_ID)
    client.on_connect = on_connect
    client.on_message = on_message

    # Connect to MQTT broker
    try:
        client.connect(MQTT_BROKER, MQTT_PORT)
        client.loop_start()
        print(f"Connecting to MQTT broker at {MQTT_BROKER}...")
    except Exception as e:
        print(f"Failed to connect to MQTT broker: {e}")
        # We'll continue without MQTT and use default values

    # Default values when no MQTT data is available
    bpm = 58
    volume = 0.7
    is_tempo_beat = False

    # BPM blinker configuration
    blink_state = False  # Start with blinker off
    next_transition_time = 0

    # Initialize onset detectors with different colors
    onset_detectors = {
        "energy": OnsetDetector("Energy"),
        "hfc": OnsetDetector("High Freq"),
        "complex": OnsetDetector("Complex"),
        "phase": OnsetDetector("Phase"),
        "specflux": OnsetDetector("Spectral"),
        "wphase": OnsetDetector("W-Phase"),
        "mkl": OnsetDetector("MKL"),
        "kl": OnsetDetector("KL"),
    }

    # Initialize note visualizer
    note_viz = NoteVisualizer()

    # Initialize pitch visualizer
    pitch_viz = PitchVisualizer()

    # Track time for smooth updates
    last_time = pygame.time.get_ticks()

    # Main game loop
    running = True
    while running:
        current_time = pygame.time.get_ticks()
        last_time = current_time

        # Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        # Update data if new MQTT message received
        if latest_data:
            # Update BPM from MQTT data
            if "bpm" in latest_data:
                bpm = latest_data["bpm"]

            # Update onset detectors from MQTT data
            for method in onset_detectors:
                if method in latest_data and latest_data[method]:
                    onset_detectors[method].activate(current_time)

            # Update volume from MQTT data
            if "volume" in latest_data:
                volume = latest_data["volume"]

            # Update note visualizer from MQTT data
            if "notes" in latest_data:
                notes = latest_data["notes"]
                note_viz.update(notes)

            # Update pitch visualizer from MQTT data
            if "pitch" in latest_data:
                pitch_data = latest_data["pitch"]
                if "value" in pitch_data and "confidence" in pitch_data:
                    pitch_viz.activate(
                        pitch_data["value"], pitch_data["confidence"], current_time
                    )
                    # print(pitch_data["value"], pitch_data["confidence"])

        # Check if it's time for a blink transition
        if current_time >= next_transition_time:
            ms_per_beat = 60000 / bpm  # Convert BPM to milliseconds
            ms_per_transition = ms_per_beat / 2  # Half-beat for on->off or off->on
            # Toggle blink state
            blink_state = not blink_state

            # Set next transition time
            next_transition_time = current_time + ms_per_transition

        # Clear the screen
        screen.fill(BACKGROUND_COLOR)

        # Draw BPM blinker in top left - showing as integer
        bpm_text = font.render(f"BPM {int(bpm)}", True, (255, 255, 255))
        screen.blit(bpm_text, (20, 20))

        # Draw blinking box
        if blink_state:
            blink_color = (255, 255, 255)  # White when on
        else:
            blink_color = (50, 50, 70)  # Dark gray when off

        pygame.draw.rect(screen, blink_color, (90, 17, 20, 20))

        # Draw Volume indicator
        vol_text = font.render("Volume", True, (255, 255, 255))
        screen.blit(vol_text, (120, 20))

        # Draw volume as a single horizontal bar
        volume_bar_width = 200  # 200px total width
        volume_bar_height = 20
        volume_bar_x = 190
        volume_bar_y = 17

        # Draw background volume bar (empty)
        pygame.draw.rect(
            screen,
            (50, 50, 70),  # Dark gray background
            (volume_bar_x, volume_bar_y, volume_bar_width, volume_bar_height),
        )

        # Draw filled portion based on volume (0-1 directly maps to 0-200px)
        filled_width = int(volume * volume_bar_width)
        if filled_width > 0:
            pygame.draw.rect(
                screen,
                (255, 255, 255),  # White for filled portion
                (volume_bar_x, volume_bar_y, filled_width, volume_bar_height),
            )

        # Draw onset detectors section title
        title_y = 60
        onset_title = font.render("Onset Detectors", True, (200, 200, 200))
        screen.blit(onset_title, (20, title_y))

        # Draw onset detectors in two columns
        detector_y = title_y + 30
        detector_height = 25
        col_width = WIDTH // 2

        # Define column layout
        left_col = [
            "energy",
            "complex",
            "specflux",
            "wphase",
        ]
        right_col = ["hfc", "phase", "mkl", "kl"]

        # Draw left column detectors
        for i, key in enumerate(left_col):
            y_pos = detector_y + i * detector_height
            onset_detectors[key].render(
                screen, small_font, 40, y_pos, col_width - 80, current_time
            )

        # Draw right column detectors
        for i, key in enumerate(right_col):
            y_pos = detector_y + i * detector_height
            onset_detectors[key].render(
                screen, small_font, col_width + 40, y_pos, col_width - 80, current_time
            )

        # Draw note visualizer in the bottom part of the screen
        note_viz_y = (
            detector_y + max(len(left_col), len(right_col)) * detector_height + 20
        )
        note_viz.render(screen, font, 40, note_viz_y, WIDTH - 80, 200)

        # Draw pitch visualizer below the note visualizer
        pitch_viz_y = note_viz_y + 100
        pitch_viz.render(screen, font, 40, pitch_viz_y, WIDTH - 80, current_time)

        # Display MQTT connection status
        status_text = "Connected" if latest_data else "No MQTT data"
        status_color = (100, 255, 100) if latest_data else (255, 100, 100)
        status_display = small_font.render(status_text, True, status_color)
        screen.blit(status_display, (20, HEIGHT - 30))

        # Update the display
        pygame.display.flip()

        # Cap the frame rate
        clock.tick(FPS)

    # Clean up
    try:
        client.loop_stop()
        client.disconnect()
    except:
        pass
    pygame.quit()


if __name__ == "__main__":
    main()
