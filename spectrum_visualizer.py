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
MQTT_TOPIC = "beatzero/spectrum_data"
MQTT_CLIENT_ID = f"beatzero-spectrum-visualizer-{int(time.time())}"

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

# Initialize data storage
latest_data = None
MAX_HISTORY = 200


class SpectrumVisualizer:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.font = pygame.font.Font(None, 24)

        # Store the FFT data
        self.band_energy = [0.0] * 8  # Assuming 8 frequency bands
        self.band_ranges = [
            (20, 80),  # Sub-bass (very low)
            (80, 250),  # Bass
            (250, 500),  # Low-mids
            (500, 1000),  # Mids
            (1000, 2000),  # Upper-mids
            (2000, 3000),  # Presence
            (3000, 4000),  # Brilliance
            (4000, 8000),  # Air/Ultra high
        ]

        # Color gradient for visualization
        self.color_gradient = [
            (50, 50, 200),  # Deep blue for low frequencies
            (100, 100, 255),  # Blue
            (50, 200, 255),  # Cyan
            (50, 255, 150),  # Green-cyan
            (100, 255, 50),  # Green
            (255, 255, 50),  # Yellow
            (255, 150, 50),  # Orange
            (255, 50, 50),  # Red for high frequencies
        ]

        self.block_width = width // len(self.band_ranges)
        self.max_bar_height = height - 80  # Leave room for labels

    def update(self, spectrum_data):
        if "band_energy" in spectrum_data:
            self.band_energy = spectrum_data["band_energy"]
        if "band_ranges" in spectrum_data:
            self.band_ranges = spectrum_data["band_ranges"]

    def draw(self, surface):
        # Draw background
        pygame.draw.rect(
            surface, (20, 20, 30), (self.x, self.y, self.width, self.height)
        )
        pygame.draw.rect(
            surface, (50, 50, 60), (self.x, self.y, self.width, self.height), 1
        )

        # Draw title
        title = "Frequency Spectrum Analyzer"
        title_surface = self.font.render(title, True, (200, 200, 200))
        surface.blit(title_surface, (self.x + 10, self.y + 10))

        # Draw grid lines (horizontal)
        for i in range(5):
            y_pos = self.y + 40 + (i * self.max_bar_height // 4)
            pygame.draw.line(
                surface,
                GRID_COLOR,
                (self.x + 5, y_pos),
                (self.x + self.width - 5, y_pos),
                1,
            )

            # Draw level label (0.0 - 1.0)
            level = 1.0 - (i / 4)  # 1.0, 0.75, 0.5, 0.25, 0.0
            level_label = self.font.render(f"{level:.1f}", True, (150, 150, 150))
            surface.blit(level_label, (self.x + 5, y_pos - 15))

        # Draw frequency bands
        for i, energy in enumerate(self.band_energy):
            # Calculate bar dimensions
            bar_width = self.block_width - 10
            bar_height = int(energy * self.max_bar_height)

            # Position
            x = self.x + (i * self.block_width) + 5
            y = self.y + 40 + self.max_bar_height - bar_height

            # Get color from gradient with intensity based on energy
            if energy < 0.05:
                # Nearly black for very low energy
                color = (5, 5, 10)
            else:
                base_color = (
                    self.color_gradient[i]
                    if i < len(self.color_gradient)
                    else (200, 200, 200)
                )

                # Scale color based on energy level
                color = (
                    int(base_color[0] * energy),
                    int(base_color[1] * energy),
                    int(base_color[2] * energy),
                )

            # Draw bar
            pygame.draw.rect(surface, color, (x, y, bar_width, bar_height))

            # Draw bar border
            pygame.draw.rect(surface, (100, 100, 120), (x, y, bar_width, bar_height), 1)

            # Draw frequency label
            min_freq, max_freq = (
                self.band_ranges[i] if i < len(self.band_ranges) else (0, 0)
            )
            if min_freq > 999:
                label = f"{min_freq // 1000}k"
            else:
                label = f"{min_freq}"

            label_surface = pygame.font.Font(None, 18).render(
                label, True, (150, 150, 150)
            )
            surface.blit(
                label_surface,
                (
                    x + (bar_width - label_surface.get_width()) // 2,
                    self.y + self.height - 20,
                ),
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
        self.decay_rate = 0.1  # Faster decay so lights go out quicker
        self.rise_rate = 0.5  # Faster rise rate for more responsive visualization

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
        # Update onset detection methods with continuous scaling
        for method in ["energy", "hfc", "complex", "phase", "specflux"]:
            if method in data["onsets"]:
                # Get the raw descriptor value and threshold
                descriptor = data["onsets"][method]["descriptor"]
                threshold = data["onsets"][method]["threshold"]
                is_beat = data["onsets"][method]["is_beat"]

                # Calculate normalized intensity - scale it relative to threshold
                # Avoid division by very small values
                if threshold > 0.01:
                    normalized_intensity = descriptor / threshold
                else:
                    normalized_intensity = 0

                if is_beat:
                    # On beat detection, go to full brightness immediately
                    self.active_levels[method] = 1.0
                else:
                    # For non-beats, only show if intensity is above 70% of threshold
                    # This creates more contrast between active and inactive
                    if normalized_intensity > 0.7:
                        target_level = min(0.5, normalized_intensity - 0.7)
                        self.active_levels[method] = max(
                            self.active_levels[method], target_level
                        )
                    else:
                        # Quickly fade out low levels
                        self.active_levels[method] -= self.decay_rate * 2

                # Apply decay - always fade out over time
                self.active_levels[method] -= self.decay_rate

                # Clamp values
                self.active_levels[method] = max(
                    0, min(1.0, self.active_levels[method])
                )
            else:
                # Decay activity
                self.active_levels[method] -= self.decay_rate
                if self.active_levels[method] < 0:
                    self.active_levels[method] = 0

        # Update kick and hihat with direct data (now based on spectrum energy)
        if data["kick_detected"]:
            self.active_levels["kick"] = 1.0  # Full brightness on detection
        else:
            # Very fast decay for kicks - they should be short and punchy
            self.active_levels["kick"] -= self.decay_rate * 4
            if self.active_levels["kick"] < 0:
                self.active_levels["kick"] = 0

        if data["hihat_detected"]:
            self.active_levels["hihat"] = 1.0  # Full brightness on detection
        else:
            # Extremely fast decay for hi-hats - they should be very short
            self.active_levels["hihat"] -= self.decay_rate * 6
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

            if activity < 0.05:
                # Nearly black when not active
                block_color = (5, 5, 10)
            else:
                # Scale color based on activity
                scaled_r = int(base_color[0] * activity)
                scaled_g = int(base_color[1] * activity)
                scaled_b = int(base_color[2] * activity)
                block_color = (scaled_r, scaled_g, scaled_b)

            # Draw the square block
            pygame.draw.rect(
                surface,
                block_color,
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
    pygame.display.set_caption("BeatZero Spectrum Visualizer")
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

    # Row 1: Spectrum visualizer
    spectrum_viz = SpectrumVisualizer(20, 20, WIDTH - 40, row_height)

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
            # Update spectrum visualizer
            if "spectrum" in latest_data:
                spectrum_viz.update(latest_data["spectrum"])

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
        spectrum_viz.draw(screen)
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
