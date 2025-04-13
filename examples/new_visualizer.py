import json
import time
import os
import pygame
import paho.mqtt.client as mqtt
from datetime import datetime
import numpy as np

# Window setup
WIDTH, HEIGHT = 1024, 600
FPS = 60
BACKGROUND_COLOR = (10, 10, 20)
GRID_COLOR = (30, 30, 40)

# MQTT parameters
MQTT_BROKER = os.environ.get("MQTT_BROKER", "localhost")
MQTT_PORT = 1883
MQTT_TOPIC = "beatzero/music_detection"
MQTT_CLIENT_ID = f"beatzero-new-visualizer-{int(time.time())}"

# Colors
COLORS = {
    "energy": (255, 50, 50),  # Red
    "hfc": (255, 165, 0),  # Orange
    "complex": (255, 255, 50),  # Yellow
    "phase": (50, 255, 50),  # Green
    "specflux": (50, 50, 255),  # Blue
    "kick": (255, 0, 128),  # Pink
    "hihat": (255, 255, 255),  # White
    "pitch": (128, 0, 255),  # Purple
    "note": (0, 255, 255),  # Cyan
    "bpm": (200, 200, 200),  # Light gray
}

# Define frequency bands
FREQ_RANGES = [
    (20, 80),  # Sub-bass (very low)
    (80, 250),  # Bass
    (250, 500),  # Low-mids
    (500, 1000),  # Mids
    (1000, 2000),  # Upper-mids
    (2000, 3000),  # Presence
    (3000, 4000),  # Brilliance
    (4000, 8000),  # Air/Ultra high
]

# Initialize data storage
latest_data = None
MAX_HISTORY = 200


class FrequencyBandVisualizer:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.font = pygame.font.Font(None, 24)
        self.block_width = width // len(FREQ_RANGES)

        # Store active bands with smoothing
        self.active_bands = [0] * len(FREQ_RANGES)  # Activity level for each band (0-1)
        self.decay_rate = 0.05  # How quickly inactive bands fade out
        self.rise_rate = 0.3  # How quickly active bands light up

        self.note_detected = False
        self.confidence = 0
        self.active_idx = -1

    def update(self, pitch, pitch_confidence, note_detected):
        self.note_detected = note_detected
        self.confidence = pitch_confidence
        old_active = self.active_idx
        self.active_idx = -1

        # Find which frequency band the pitch falls into
        for i, (min_freq, max_freq) in enumerate(FREQ_RANGES):
            if min_freq <= pitch < max_freq:
                self.active_idx = i
                # Gradually increase active band brightness
                self.active_bands[i] += self.rise_rate
                if self.active_bands[i] > 1:
                    self.active_bands[i] = 1
                break

        # Apply decay to all inactive bands
        for i in range(len(self.active_bands)):
            if i != self.active_idx:  # Not the active band
                self.active_bands[i] -= self.decay_rate
                if self.active_bands[i] < 0:
                    self.active_bands[i] = 0

    def draw(self, surface):
        # Draw background
        pygame.draw.rect(
            surface, (20, 20, 30), (self.x, self.y, self.width, self.height)
        )
        pygame.draw.rect(
            surface, (50, 50, 60), (self.x, self.y, self.width, self.height), 1
        )

        # Draw title
        title = "Frequency Bands"
        title_surface = self.font.render(title, True, (200, 200, 200))
        surface.blit(title_surface, (self.x + 10, self.y + 10))

        # Calculate dimensions for square blocks
        block_size = min(self.block_width - 10, (self.height - 80) // 2)
        block_y = self.y + 50  # Position after title

        # Draw frequency band labels
        for i, (min_freq, max_freq) in enumerate(FREQ_RANGES):
            x = self.x + i * self.block_width + (self.block_width - block_size) // 2

            # Draw frequency label (only for some bands to avoid clutter)
            if i % 2 == 0:
                label = f"{min_freq}"
                label_surface = pygame.font.Font(None, 18).render(
                    label, True, (150, 150, 150)
                )
                label_width = label_surface.get_width()
                # Center the label under the block
                surface.blit(
                    label_surface,
                    (x + (block_size - label_width) // 2, block_y + block_size + 5),
                )

        # Draw the blocks
        for i, (min_freq, max_freq) in enumerate(FREQ_RANGES):
            x = self.x + i * self.block_width + (self.block_width - block_size) // 2

            # Base color for this band
            r, g, b = 50, 100, 200  # Base blue color

            # Calculate brightness based on smoothed activity level
            activity = self.active_bands[i]

            # Change color if note detected for this band
            if i == self.active_idx and self.note_detected and self.confidence > 0.4:
                r, g, b = 50, 200, 255  # Bright cyan for notes

            # Scale RGB based on activity (keeping some minimum brightness)
            min_brightness = 0.2
            scaled_r = int(r * (min_brightness + activity * (1 - min_brightness)))
            scaled_g = int(g * (min_brightness + activity * (1 - min_brightness)))
            scaled_b = int(b * (min_brightness + activity * (1 - min_brightness)))

            # Draw the square block
            pygame.draw.rect(
                surface,
                (scaled_r, scaled_g, scaled_b),
                (x, block_y, block_size, block_size),
            )

            # Draw a border around the block
            pygame.draw.rect(
                surface, (100, 100, 120), (x, block_y, block_size, block_size), 1
            )


class OnsetDetectionVisualizer:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.font = pygame.font.Font(None, 24)

        # Define onset methods
        self.onset_methods = [
            "energy",
            "hfc",
            "complex",
            "phase",
            "specflux",
            "kick",
            "hihat",
        ]
        self.block_width = width // len(self.onset_methods)

        # Store active methods with smoothing
        self.active_levels = {method: 0 for method in self.onset_methods}
        self.decay_rate = 0.05
        self.rise_rate = 0.3

        # Labels and positions
        self.labels = {
            "energy": "Energy",
            "hfc": "HFC",
            "complex": "Complex",
            "phase": "Phase",
            "specflux": "Specflux",
            "kick": "Kick",
            "hihat": "Hi-hat",
        }

    def update(self, data):
        # Update onset detection methods
        for method in ["energy", "hfc", "complex", "phase", "specflux"]:
            if method in data["onsets"] and data["onsets"][method]["is_beat"]:
                # Detected beat - increase activity
                self.active_levels[method] += self.rise_rate
                if self.active_levels[method] > 1:
                    self.active_levels[method] = 1
            else:
                # Decay activity
                self.active_levels[method] -= self.decay_rate
                if self.active_levels[method] < 0:
                    self.active_levels[method] = 0

        # Update kick and hihat
        if data["kick_detected"]:
            self.active_levels["kick"] += self.rise_rate
            if self.active_levels["kick"] > 1:
                self.active_levels["kick"] = 1
        else:
            self.active_levels["kick"] -= self.decay_rate
            if self.active_levels["kick"] < 0:
                self.active_levels["kick"] = 0

        if data["hihat_detected"]:
            self.active_levels["hihat"] += self.rise_rate
            if self.active_levels["hihat"] > 1:
                self.active_levels["hihat"] = 1
        else:
            self.active_levels["hihat"] -= self.decay_rate
            if self.active_levels["hihat"] < 0:
                self.active_levels["hihat"] = 0

    def draw(self, surface):
        # Draw background
        pygame.draw.rect(
            surface, (20, 20, 30), (self.x, self.y, self.width, self.height)
        )
        pygame.draw.rect(
            surface, (50, 50, 60), (self.x, self.y, self.width, self.height), 1
        )

        # Draw title
        title = "Onset Detection Methods"
        title_surface = self.font.render(title, True, (200, 200, 200))
        surface.blit(title_surface, (self.x + 10, self.y + 10))

        # Calculate dimensions for blocks
        block_size = min(self.block_width - 10, (self.height - 80) // 2)
        block_y = self.y + 50  # Position after title

        # Draw blocks for each detection method
        for i, method in enumerate(self.onset_methods):
            x = self.x + i * self.block_width + (self.block_width - block_size) // 2

            # Get base color for this method
            base_color = COLORS.get(
                method, (200, 200, 200)
            )  # Default to light gray if not found

            # Calculate brightness based on activity level
            activity = self.active_levels[method]
            min_brightness = 0.2
            scaled_r = int(
                base_color[0] * (min_brightness + activity * (1 - min_brightness))
            )
            scaled_g = int(
                base_color[1] * (min_brightness + activity * (1 - min_brightness))
            )
            scaled_b = int(
                base_color[2] * (min_brightness + activity * (1 - min_brightness))
            )

            # Draw the square block
            pygame.draw.rect(
                surface,
                (scaled_r, scaled_g, scaled_b),
                (x, block_y, block_size, block_size),
            )

            # Draw border
            pygame.draw.rect(
                surface, (100, 100, 120), (x, block_y, block_size, block_size), 1
            )

            # Draw label
            label = self.labels.get(method, method)
            label_surface = pygame.font.Font(None, 18).render(
                label, True, (150, 150, 150)
            )
            label_width = label_surface.get_width()

            # Center the label under the block
            surface.blit(
                label_surface,
                (x + (block_size - label_width) // 2, block_y + block_size + 5),
            )


# MQTT callbacks
def on_connect(client, userdata, flags, rc):
    """Callback for when the client connects to the broker"""
    if rc == 0:
        print(f"Connected to MQTT broker")
        client.subscribe(MQTT_TOPIC)
        print(f"Subscribed to topic: {MQTT_TOPIC}")
    else:
        print(f"Failed to connect to MQTT broker with code: {rc}")


def on_message(client, userdata, msg):
    """Callback for when a message is received from the broker"""
    global latest_data

    try:
        data = json.loads(msg.payload.decode())
        latest_data = data
    except Exception as e:
        print(f"Error parsing message: {e}")


def main():
    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("BeatZero New Visualizer")
    clock = pygame.time.Clock()

    # Initialize MQTT client
    client = mqtt.Client(client_id=MQTT_CLIENT_ID)
    client.on_connect = on_connect
    client.on_message = on_message

    # Connect to MQTT broker
    try:
        client.connect(MQTT_BROKER, MQTT_PORT)
        client.loop_start()
    except Exception as e:
        print(f"Failed to connect to MQTT broker: {e}")
        pygame.quit()
        return

    # Initialize visualization elements
    row_height = HEIGHT // 2 - 20  # Height for each row with some margin

    # Row 1: Frequency band visualizer
    freq_band_viz = FrequencyBandVisualizer(20, 20, WIDTH - 40, row_height)

    # Row 2: Onset detection visualizer
    onset_viz = OnsetDetectionVisualizer(20, row_height + 40, WIDTH - 40, row_height)

    # Font for on-screen info
    font = pygame.font.Font(None, 32)
    small_font = pygame.font.Font(None, 24)

    # Main game loop
    running = True
    while running:
        # Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        # Clear the screen
        screen.fill(BACKGROUND_COLOR)

        # Update visualizations if new data is available
        if latest_data:
            # Update frequency band visualizer
            freq_band_viz.update(
                latest_data["pitch"]["value"],
                latest_data["pitch"]["confidence"],
                latest_data["note_detected"],
            )

            # Update onset detection visualizer
            onset_viz.update(latest_data)

            # Display current BPM if available
            bpm_text = f"BPM: {latest_data['tempo']['bpm']:.1f}"
            bpm_surface = font.render(bpm_text, True, COLORS["bpm"])
            screen.blit(bpm_surface, (WIDTH - 150, 20))

            # Display timestamp
            timestamp = datetime.fromisoformat(latest_data["timestamp"]).strftime(
                "%H:%M:%S"
            )
            ts_surface = small_font.render(f"Time: {timestamp}", True, (150, 150, 150))
            screen.blit(ts_surface, (WIDTH - 150, 60))

        # Draw visualization elements
        freq_band_viz.draw(screen)
        onset_viz.draw(screen)

        # Update the display
        pygame.display.flip()

        # Cap the frame rate
        clock.tick(FPS)

    # Clean up
    client.loop_stop()
    client.disconnect()
    pygame.quit()


if __name__ == "__main__":
    main()
