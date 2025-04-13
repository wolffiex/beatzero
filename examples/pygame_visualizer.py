import json
import time
import math
import os
import pygame
import paho.mqtt.client as mqtt
from datetime import datetime
import random

# Window setup
WIDTH, HEIGHT = 1024, 768
FPS = 60
BACKGROUND_COLOR = (0, 0, 0)

# MQTT parameters
MQTT_BROKER = os.environ.get("MQTT_BROKER", "localhost")
MQTT_PORT = 1883
MQTT_TOPIC = "beatzero/music_detection"
MQTT_CLIENT_ID = f"beatzero-pygame-visualizer-{int(time.time())}"

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
history_data = {
    "timestamps": [],
    "beat_intensity": [],
    "notes": [],
    "kick_times": [],
    "hihat_times": [],
}
HISTORY_LENGTH = 100  # Number of data points to keep


# Visualization elements
class PulseCircle:
    def __init__(
        self,
        color,
        max_radius,
        x=WIDTH // 2,
        y=HEIGHT // 2,
        decay_rate=0.05,
        pulsate=True,
    ):
        self.x = x
        self.y = y
        self.color = color
        self.current_radius = 0
        self.max_radius = max_radius
        self.active = False
        self.decay_rate = decay_rate
        self.opacity = 255
        self.pulsate = pulsate
        self.pulsate_size = 0

    def trigger(self):
        self.active = True
        self.current_radius = 10  # Starting radius
        self.opacity = 255

    def update(self):
        if not self.active and not self.pulsate:
            return

        if self.active:
            self.current_radius += (self.max_radius - self.current_radius) * 0.2
            self.opacity -= self.decay_rate * 255

            if self.opacity <= 0:
                self.active = False
                self.opacity = 0

        if self.pulsate:
            self.pulsate_size = 5 + 3 * math.sin(time.time() * 3)

    def draw(self, surface):
        if self.active or self.pulsate:
            if self.pulsate and not self.active:
                draw_radius = self.max_radius * 0.5 + self.pulsate_size
                alpha = 100
            else:
                draw_radius = self.current_radius
                alpha = max(0, min(255, int(self.opacity)))

            color_with_alpha = (*self.color, alpha)

            # Create a surface with per-pixel alpha
            circle_surface = pygame.Surface(
                (draw_radius * 2, draw_radius * 2), pygame.SRCALPHA
            )
            pygame.draw.circle(
                circle_surface,
                color_with_alpha,
                (draw_radius, draw_radius),
                draw_radius,
            )

            # Blit the surface onto the main surface
            surface.blit(circle_surface, (self.x - draw_radius, self.y - draw_radius))


class FrequencyBands:
    def __init__(self):
        self.freq_ranges = [
            (20, 80),  # Sub-bass
            (80, 250),  # Bass
            (250, 500),  # Low-mids
            (500, 1000),  # Mids
            (1000, 2000),  # Upper-mids
            (2000, 3000),  # Presence
            (3000, 4000),  # Brilliance
            (4000, 8000),  # Air/Ultra high
        ]
        self.heights = [0] * len(self.freq_ranges)
        self.target_heights = [0] * len(self.freq_ranges)
        self.band_width = WIDTH // len(self.freq_ranges)
        self.spacing = 5
        self.max_height = HEIGHT // 2

    def update(self, pitch, pitch_confidence, note_detected):
        # Reset target heights
        self.target_heights = [0] * len(self.freq_ranges)

        # Set the target height for the band containing the current pitch
        for i, (min_freq, max_freq) in enumerate(self.freq_ranges):
            if min_freq <= pitch < max_freq:
                # Higher confidence and note detection means higher bars
                height_factor = pitch_confidence
                if note_detected:
                    height_factor *= 1.5
                self.target_heights[i] = self.max_height * min(1.0, height_factor)

        # Smoothly animate current heights toward target heights
        for i in range(len(self.heights)):
            self.heights[i] += (self.target_heights[i] - self.heights[i]) * 0.3

    def draw(self, surface):
        for i in range(len(self.freq_ranges)):
            x = i * self.band_width + self.spacing
            y = HEIGHT - self.heights[i]
            width = self.band_width - 2 * self.spacing
            height = self.heights[i]

            # Choose color based on frequency band
            hue = int(255 * i / len(self.freq_ranges))
            color = pygame.Color(0)
            color.hsla = (hue, 100, 50, 100)

            # Draw the frequency band
            pygame.draw.rect(surface, color, (x, y, width, height))


class ParticleSystem:
    def __init__(self, color, x, y, lifetime=1.0):
        self.particles = []
        self.color = color
        self.x = x
        self.y = y
        self.lifetime = lifetime

    def emit(self, count=10, velocity_range=(1, 5), size_range=(2, 8)):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            velocity = random.uniform(*velocity_range)
            vx = math.cos(angle) * velocity
            vy = math.sin(angle) * velocity
            size = random.uniform(*size_range)
            lifetime = random.uniform(0.5, self.lifetime)

            self.particles.append(
                {
                    "x": self.x,
                    "y": self.y,
                    "vx": vx,
                    "vy": vy,
                    "size": size,
                    "lifetime": lifetime,
                    "age": 0,
                }
            )

    def update(self, dt):
        # Update existing particles
        for particle in self.particles[:]:
            particle["x"] += particle["vx"]
            particle["y"] += particle["vy"]
            particle["age"] += dt

            # Remove particles that have exceeded their lifetime
            if particle["age"] > particle["lifetime"]:
                self.particles.remove(particle)

    def draw(self, surface):
        for particle in self.particles:
            # Calculate opacity based on particle age
            alpha = 255 * (1 - particle["age"] / particle["lifetime"])
            color_with_alpha = (*self.color, alpha)

            # Create a surface with per-pixel alpha
            size = int(particle["size"])
            if size < 1:
                continue

            circle_surface = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
            pygame.draw.circle(circle_surface, color_with_alpha, (size, size), size)

            # Blit the surface onto the main surface
            surface.blit(circle_surface, (particle["x"] - size, particle["y"] - size))


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
        latest_data = json.loads(msg.payload.decode())
    except Exception as e:
        print(f"Error parsing message: {e}")


def main():
    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("BeatZero Pygame Visualizer")
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
    pulse_circles = {
        "energy": PulseCircle(COLORS["energy"], 150, WIDTH // 2, HEIGHT // 2),
        "hfc": PulseCircle(COLORS["hfc"], 200, WIDTH // 2, HEIGHT // 2),
        "complex": PulseCircle(COLORS["complex"], 250, WIDTH // 2, HEIGHT // 2),
        "phase": PulseCircle(COLORS["phase"], 300, WIDTH // 2, HEIGHT // 2),
        "specflux": PulseCircle(COLORS["specflux"], 350, WIDTH // 2, HEIGHT // 2),
        "kick": PulseCircle(COLORS["kick"], 400, WIDTH // 2, HEIGHT // 2),
        "hihat": PulseCircle(COLORS["hihat"], 100, WIDTH // 2, HEIGHT // 2),
        "bpm": PulseCircle(
            COLORS["bpm"], 450, WIDTH // 2, HEIGHT // 2, decay_rate=0.02, pulsate=True
        ),
    }

    frequency_bands = FrequencyBands()

    # Particle systems for kick and hihat
    kick_particles = ParticleSystem(
        COLORS["kick"], WIDTH // 2, HEIGHT // 2, lifetime=0.8
    )
    hihat_particles = ParticleSystem(
        COLORS["hihat"], WIDTH // 2, HEIGHT // 2, lifetime=0.5
    )

    font = pygame.font.Font(None, 36)
    small_font = pygame.font.Font(None, 24)

    # Main game loop
    running = True
    last_time = time.time()

    while running:
        # Calculate delta time
        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time

        # Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        # Clear the screen
        screen.fill(BACKGROUND_COLOR)

        # Update and draw visualization elements based on latest data
        if latest_data:
            # Process onset detections
            for method in pulse_circles:
                if method in ["energy", "hfc", "complex", "phase", "specflux"]:
                    if (
                        method in latest_data["onsets"]
                        and latest_data["onsets"][method]["is_beat"]
                    ):
                        pulse_circles[method].trigger()
                elif method == "kick" and latest_data["kick_detected"]:
                    pulse_circles[method].trigger()
                    kick_particles.emit(30, velocity_range=(2, 8), size_range=(3, 12))
                elif method == "hihat" and latest_data["hihat_detected"]:
                    pulse_circles[method].trigger()
                    hihat_particles.emit(20, velocity_range=(5, 10), size_range=(1, 5))
                elif method == "bpm" and latest_data["tempo"]["is_beat"]:
                    pulse_circles[method].trigger()

            # Update frequency bands
            frequency_bands.update(
                latest_data["pitch"]["value"],
                latest_data["pitch"]["confidence"],
                latest_data["note_detected"],
            )

            # Display BPM
            bpm_text = f"BPM: {latest_data['tempo']['bpm']:.1f}"
            bpm_surface = font.render(bpm_text, True, COLORS["bpm"])
            screen.blit(bpm_surface, (20, 20))

            # Display timestamp
            timestamp = datetime.fromisoformat(latest_data["timestamp"]).strftime(
                "%H:%M:%S"
            )
            ts_surface = small_font.render(f"Time: {timestamp}", True, (150, 150, 150))
            screen.blit(ts_surface, (20, 60))

        # Update and draw all visualization elements
        for circle in pulse_circles.values():
            circle.update()
            circle.draw(screen)

        frequency_bands.draw(screen)

        # Update and draw particle systems
        kick_particles.update(dt)
        kick_particles.draw(screen)

        hihat_particles.update(dt)
        hihat_particles.draw(screen)

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
