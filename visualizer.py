import pygame
import time
import json
import os
import paho.mqtt.client as mqtt
from datetime import datetime

# Window setup
WIDTH, HEIGHT = 600, 600
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
            box_color = (50, 50, 70)  # Dark gray when inactive

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
    }

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

        # Calculate how many volume bars to light based on volume level
        volume_bars = 10
        active_bars = int(volume * volume_bars)

        # Draw volume bars
        bar_width = 10
        bar_height = 20
        bar_spacing = 2
        bar_x_start = 190

        for i in range(volume_bars):
            # Determine if this bar should be lit
            if i < active_bars:
                bar_color = (255, 255, 255)
            else:
                bar_color = (50, 50, 70)

            # Draw the bar
            pygame.draw.rect(
                screen,
                bar_color,
                (
                    bar_x_start + i * (bar_width + bar_spacing),
                    17,
                    bar_width,
                    bar_height,
                ),
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
        ]
        right_col = ["hfc", "phase"]

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
