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
MQTT_BROKER = os.environ.get(
    "MQTT_BROKER", "localhost"
)  # Use environment variable with fallback
MQTT_PORT = 1883
MQTT_TOPIC = "beatzero/music_detection"
MQTT_CLIENT_ID = f"beatzero-signal-display-{int(time.time())}"

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

# Define onset methods
ONSET_METHODS = ["energy", "hfc", "complex", "phase", "specflux"]

# Initialize data storage
latest_data = None
history = {
    "energy": [],
    "hfc": [],
    "complex": [],
    "phase": [],
    "specflux": [],
    "pitch": [],
    "confidence": [],
    "kick": [],
    "hihat": [],
    "bpm": [],
}
MAX_HISTORY = 200  # Store maximum 200 data points for each signal


class TimeSeriesDisplay:
    def __init__(
        self,
        x,
        y,
        width,
        height,
        max_points=100,
        line_color=(255, 255, 255),
        label="Signal",
        y_min=0,
        y_max=1,
        show_grid=True,
        threshold_display=False,
        threshold_color=(100, 100, 100),
    ):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.max_points = max_points
        self.line_color = line_color
        self.label = label
        self.y_min = y_min
        self.y_max = y_max
        self.show_grid = show_grid
        self.threshold_display = threshold_display
        self.threshold_color = threshold_color
        self.threshold_value = 0.5  # Default threshold value

    def draw(self, surface, data_points, threshold_values=None):
        # Draw background
        pygame.draw.rect(
            surface, (20, 20, 30), (self.x, self.y, self.width, self.height)
        )
        pygame.draw.rect(
            surface, (50, 50, 60), (self.x, self.y, self.width, self.height), 1
        )

        # Draw grid
        if self.show_grid:
            # Vertical grid lines
            for i in range(10):
                line_x = self.x + (i / 10) * self.width
                pygame.draw.line(
                    surface,
                    GRID_COLOR,
                    (line_x, self.y),
                    (line_x, self.y + self.height),
                    1,
                )

            # Horizontal grid lines
            for i in range(5):
                line_y = self.y + (i / 5) * self.height
                pygame.draw.line(
                    surface,
                    GRID_COLOR,
                    (self.x, line_y),
                    (self.x + self.width, line_y),
                    1,
                )

        # Draw label and beat count if applicable
        font = pygame.font.Font(None, 22)
        beat_count = sum(
            1 for point in data_points if isinstance(point, tuple) and point[0]
        )
        label_text = (
            f"{self.label} (Beats: {beat_count})" if beat_count > 0 else self.label
        )
        label_surface = font.render(label_text, True, self.line_color)
        surface.blit(label_surface, (self.x + 5, self.y + 5))

        # Draw min/max values
        min_val_surface = font.render(f"{self.y_min}", True, (150, 150, 150))
        max_val_surface = font.render(f"{self.y_max}", True, (150, 150, 150))
        surface.blit(min_val_surface, (self.x + 5, self.y + self.height - 20))
        surface.blit(max_val_surface, (self.x + 5, self.y + 20))

        # Draw time series data
        if len(data_points) > 1:
            # Scale data to fit the graph
            scaled_points = []
            display_points = (
                data_points[-self.max_points :]
                if len(data_points) > self.max_points
                else data_points
            )

            # Draw threshold line if available
            if (
                self.threshold_display
                and threshold_values
                and len(threshold_values) > 0
            ):
                # Get the most recent threshold
                threshold = threshold_values[-1]
                # Normalize threshold
                norm_threshold = (threshold - self.y_min) / (self.y_max - self.y_min)
                norm_threshold = max(0, min(1, norm_threshold))

                # Calculate y position for threshold
                threshold_y = self.y + self.height - (norm_threshold * self.height)

                # Draw threshold line
                pygame.draw.line(
                    surface,
                    self.threshold_color,
                    (self.x, threshold_y),
                    (self.x + self.width, threshold_y),
                    1,
                )

            # Process and draw main data points
            for i, value in enumerate(display_points):
                # Check if value is a tuple (is_beat, descriptor)
                if isinstance(value, tuple):
                    is_beat, descriptor = value
                    value = descriptor  # Use descriptor for the graph

                # Normalize value between y_min and y_max
                normalized = (value - self.y_min) / (self.y_max - self.y_min)
                normalized = max(0, min(1, normalized))  # Clamp between 0 and 1

                # Calculate x,y position
                point_x = self.x + (i / len(display_points)) * self.width
                point_y = self.y + self.height - (normalized * self.height)

                # If it's a beat detection point, draw a circle
                if isinstance(display_points[i], tuple) and display_points[i][0]:
                    pygame.draw.circle(
                        surface, (255, 255, 255), (int(point_x), int(point_y)), 4
                    )

                scaled_points.append((point_x, point_y))

            # Draw line connecting the points
            if len(scaled_points) > 1:
                pygame.draw.lines(surface, self.line_color, False, scaled_points, 2)


class SignalTable:
    def __init__(self, x, y, width, height, row_height=30):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.row_height = row_height
        self.font = pygame.font.Font(None, 24)

    def draw(self, surface, data_dict):
        # Draw background
        pygame.draw.rect(
            surface, (20, 20, 30), (self.x, self.y, self.width, self.height)
        )
        pygame.draw.rect(
            surface, (50, 50, 60), (self.x, self.y, self.width, self.height), 1
        )

        # Draw table header
        header_color = (200, 200, 200)
        pygame.draw.rect(
            surface, (30, 30, 40), (self.x, self.y, self.width, self.row_height)
        )

        header_labels = ["Signal", "Value", "Min", "Max", "Avg"]
        col_width = self.width / len(header_labels)

        for i, label in enumerate(header_labels):
            header_surface = self.font.render(label, True, header_color)
            surface.blit(header_surface, (self.x + i * col_width + 10, self.y + 5))

        # Draw rows
        row_y = self.y + self.row_height
        for i, (key, values) in enumerate(data_dict.items()):
            # Skip if no values
            if not values:
                continue

            if i % 2 == 0:
                pygame.draw.rect(
                    surface, (25, 25, 35), (self.x, row_y, self.width, self.row_height)
                )

            # Get signal color
            signal_color = COLORS.get(key, (200, 200, 200))

            # Process values based on type
            # For onset methods, values are tuples of (is_beat, descriptor)
            if key in ["energy", "hfc", "complex", "phase", "specflux"]:
                # Extract descriptor values for statistical calculations
                descriptors = [v[1] for v in values]
                current = descriptors[-1] if descriptors else 0
                min_val = min(descriptors) if descriptors else 0
                max_val = max(descriptors) if descriptors else 0
                avg_val = sum(descriptors) / len(descriptors) if descriptors else 0

                # Add beat indication to name
                is_current_beat = values[-1][0] if values else False
                display_key = f"{key} {'★' if is_current_beat else ''}"
            else:
                # For other metrics, values are simple numbers
                current = values[-1] if values else 0
                min_val = min(values) if values else 0
                max_val = max(values) if values else 0
                avg_val = sum(values) / len(values) if values else 0
                display_key = key

            # Column 1: Signal name
            name_surface = self.font.render(display_key, True, signal_color)
            surface.blit(name_surface, (self.x + 10, row_y + 5))

            # Column 2: Current value
            value_surface = self.font.render(f"{current:.3f}", True, (255, 255, 255))
            surface.blit(value_surface, (self.x + col_width + 10, row_y + 5))

            # Column 3: Min value
            min_surface = self.font.render(f"{min_val:.3f}", True, (150, 150, 150))
            surface.blit(min_surface, (self.x + col_width * 2 + 10, row_y + 5))

            # Column 4: Max value
            max_surface = self.font.render(f"{max_val:.3f}", True, (150, 150, 150))
            surface.blit(max_surface, (self.x + col_width * 3 + 10, row_y + 5))

            # Column 5: Avg value
            avg_surface = self.font.render(f"{avg_val:.3f}", True, (150, 150, 150))
            surface.blit(avg_surface, (self.x + col_width * 4 + 10, row_y + 5))

            row_y += self.row_height

            # Stop if we run out of space
            if row_y > self.y + self.height:
                break


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
    global history

    try:
        data = json.loads(msg.payload.decode())
        latest_data = data

        # Update history for onset detection methods
        for method in ["energy", "hfc", "complex", "phase", "specflux"]:
            if method in data["onsets"]:
                onset_data = data["onsets"][method]
                # Store both beat detection and descriptor value as a tuple
                is_beat = onset_data["is_beat"]
                descriptor = onset_data["descriptor"]
                threshold = onset_data["threshold"]

                # Store as tuple (is_beat, descriptor)
                history[method].append((is_beat, descriptor))
                if len(history[method]) > MAX_HISTORY:
                    history[method] = history[method][-MAX_HISTORY:]

        # Update pitch and confidence
        history["pitch"].append(data["pitch"]["value"])
        history["confidence"].append(data["pitch"]["confidence"])
        if len(history["pitch"]) > MAX_HISTORY:
            history["pitch"] = history["pitch"][-MAX_HISTORY:]
            history["confidence"] = history["confidence"][-MAX_HISTORY:]

        # Update kick/hihat/bpm
        history["kick"].append(1.0 if data["kick_detected"] else 0.0)
        history["hihat"].append(1.0 if data["hihat_detected"] else 0.0)
        history["bpm"].append(data["tempo"]["bpm"])
        if len(history["kick"]) > MAX_HISTORY:
            history["kick"] = history["kick"][-MAX_HISTORY:]
            history["hihat"] = history["hihat"][-MAX_HISTORY:]
            history["bpm"] = history["bpm"][-MAX_HISTORY:]

    except Exception as e:
        print(f"Error parsing message: {e}")


def main():
    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("BeatZero Signal Display")
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

    # Onset detection graphs
    onset_width = WIDTH // 2 - 20
    onset_height = 100
    onset_displays = {
        "energy": TimeSeriesDisplay(
            10,
            10,
            onset_width,
            onset_height,
            line_color=COLORS["energy"],
            label="Energy Onset",
            y_min=0,
            y_max=1,
            threshold_display=True,
        ),
        "hfc": TimeSeriesDisplay(
            10,
            10 + onset_height + 10,
            onset_width,
            onset_height,
            line_color=COLORS["hfc"],
            label="HFC Onset",
            y_min=0,
            y_max=2,
            threshold_display=True,
        ),
        "complex": TimeSeriesDisplay(
            10,
            10 + (onset_height + 10) * 2,
            onset_width,
            onset_height,
            line_color=COLORS["complex"],
            label="Complex Onset",
            y_min=0,
            y_max=1,
            threshold_display=True,
        ),
        "phase": TimeSeriesDisplay(
            10,
            10 + (onset_height + 10) * 3,
            onset_width,
            onset_height,
            line_color=COLORS["phase"],
            label="Phase Onset",
            y_min=0,
            y_max=1,
            threshold_display=True,
        ),
        "specflux": TimeSeriesDisplay(
            10,
            10 + (onset_height + 10) * 4,
            onset_width,
            onset_height,
            line_color=COLORS["specflux"],
            label="Specflux Onset",
            y_min=0,
            y_max=1,
            threshold_display=True,
        ),
    }

    # Threshold history for each method
    threshold_history = {method: [] for method in ONSET_METHODS}

    # Pitch and confidence graph
    pitch_display = TimeSeriesDisplay(
        WIDTH // 2 + 10,
        10,
        onset_width,
        onset_height,
        line_color=COLORS["pitch"],
        label="Pitch (Hz)",
        y_min=0,
        y_max=1000,
    )

    confidence_display = TimeSeriesDisplay(
        WIDTH // 2 + 10,
        10 + onset_height + 10,
        onset_width,
        onset_height,
        line_color=COLORS["note"],
        label="Pitch Confidence",
        y_min=0,
        y_max=1,
    )

    # Kick and hihat detection
    percussion_display = TimeSeriesDisplay(
        WIDTH // 2 + 10,
        10 + (onset_height + 10) * 2,
        onset_width,
        onset_height,
        line_color=(200, 200, 200),
        label="Percussion Detection",
    )

    # BPM tracking
    bpm_display = TimeSeriesDisplay(
        WIDTH // 2 + 10,
        10 + (onset_height + 10) * 3,
        onset_width,
        onset_height,
        line_color=COLORS["bpm"],
        label="BPM",
        y_min=60,
        y_max=200,
    )

    # Signal data table
    signal_table = SignalTable(
        WIDTH // 2 + 10, 10 + (onset_height + 10) * 4, onset_width, onset_height
    )

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

        # Update threshold history if new data available
        if latest_data:
            for method in ONSET_METHODS:
                if method in latest_data["onsets"]:
                    threshold = latest_data["onsets"][method]["threshold"]
                    threshold_history[method].append(threshold)
                    if len(threshold_history[method]) > MAX_HISTORY:
                        threshold_history[method] = threshold_history[method][
                            -MAX_HISTORY:
                        ]

        # Draw all time series displays with thresholds
        for method, display in onset_displays.items():
            display.draw(screen, history[method], threshold_history.get(method, []))

        # Draw pitch and confidence displays
        pitch_display.draw(screen, history["pitch"])
        confidence_display.draw(screen, history["confidence"])

        # Draw percussion display with both kick and hihat
        if history["kick"] and history["hihat"]:
            # Draw background first
            percussion_display.draw(screen, [0] * len(history["kick"]))

            # Draw kick detection
            kick_display = TimeSeriesDisplay(
                percussion_display.x,
                percussion_display.y,
                percussion_display.width,
                percussion_display.height,
                line_color=COLORS["kick"],
                label="Kick (Red) & Hihat (White)",
                show_grid=False,
            )
            kick_display.draw(screen, history["kick"])

            # Draw hihat detection (with same dimensions)
            hihat_display = TimeSeriesDisplay(
                percussion_display.x,
                percussion_display.y,
                percussion_display.width,
                percussion_display.height,
                line_color=COLORS["hihat"],
                show_grid=False,
            )
            hihat_display.draw(screen, history["hihat"])

        # Draw BPM display
        bpm_display.draw(screen, history["bpm"])

        # Draw signal data table
        table_data = {key: values for key, values in history.items() if values}
        signal_table.draw(screen, table_data)

        # Display current BPM if available
        if latest_data:
            bpm_text = f"Current BPM: {latest_data['tempo']['bpm']:.1f}"
            bpm_surface = font.render(bpm_text, True, COLORS["bpm"])
            screen.blit(bpm_surface, (WIDTH - 250, 10))

            # Display timestamp
            timestamp = datetime.fromisoformat(latest_data["timestamp"]).strftime(
                "%H:%M:%S"
            )
            ts_surface = small_font.render(f"Time: {timestamp}", True, (150, 150, 150))
            screen.blit(ts_surface, (WIDTH - 250, 50))

            # Display note detection status
            if latest_data["note_detected"]:
                note_text = f"Note Detected: Yes"
                note_surface = font.render(note_text, True, COLORS["note"])
                screen.blit(note_surface, (WIDTH - 250, 90))

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
