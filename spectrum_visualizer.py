import json
import time
import os
import pygame
import paho.mqtt.client as mqtt
from datetime import datetime
import numpy as np

# Window setup
WIDTH, HEIGHT = 1024, 680  # Increased height for the BPM visualizer row
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
            (20, 80),     # Sub-bass (very low)
            (80, 250),    # Bass
            (250, 500),   # Low-mids
            (500, 1000),  # Mids
            (1000, 2000), # Upper-mids
            (2000, 3000), # Presence
            (3000, 4000), # Brilliance
            (4000, 8000), # Air/Ultra high
        ]
        
        # Color gradient for visualization
        self.color_gradient = [
            (50, 50, 200),     # Deep blue for low frequencies
            (100, 100, 255),   # Blue
            (50, 200, 255),    # Cyan
            (50, 255, 150),    # Green-cyan
            (100, 255, 50),    # Green
            (255, 255, 50),    # Yellow
            (255, 150, 50),    # Orange
            (255, 50, 50),     # Red for high frequencies
        ]
        
        self.block_width = width // len(self.band_ranges)
    
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
        
        # Calculate block size for frequency boxes (similar to onset detection)
        block_size = min(self.block_width - 10, (self.height - 80) // 2)
        block_y = self.y + 50  # Position after title
        
        # Draw frequency bands as illuminated blocks (similar to onset detection)
        for i, energy in enumerate(self.band_energy):
            # Position the block
            x = self.x + i * self.block_width + (self.block_width - block_size) // 2
            
            # Get base color from gradient
            base_color = (
                self.color_gradient[i]
                if i < len(self.color_gradient)
                else (200, 200, 200)
            )
            
            # Determine color based on energy level
            if energy < 0.1:
                # Much darker when inactive - almost completely black
                color = (2, 2, 5)
            else:
                # Scale color based on energy level but with stronger contrast
                # Apply a non-linear scaling to make high values brighter and low values darker
                energy_scaled = energy ** 2  # Square the value to increase contrast
                color = (
                    int(base_color[0] * energy_scaled),
                    int(base_color[1] * energy_scaled),
                    int(base_color[2] * energy_scaled)
                )
            
            # Draw the frequency block
            pygame.draw.rect(
                surface,
                color,
                (x, block_y, block_size, block_size)
            )
            
            # Draw border around block
            pygame.draw.rect(
                surface, (100, 100, 120), (x, block_y, block_size, block_size), 1
            )
            
            # Draw frequency label under block
            min_freq, max_freq = self.band_ranges[i] if i < len(self.band_ranges) else (0, 0)
            if min_freq > 999:
                label = f"{min_freq//1000}k"
            else:
                label = f"{min_freq}"
                
            label_surface = pygame.font.Font(None, 18).render(
                label, True, (150, 150, 150)
            )
            label_width = label_surface.get_width()
            
            # Center the label under the block
            surface.blit(
                label_surface,
                (x + (block_size - label_width) // 2, block_y + block_size + 5)
            )

class BPMVisualizer:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.font = pygame.font.Font(None, 36)  # Larger font for BPM
        
        # BPM tracking
        self.current_bpm = 120.0
        self.scaled_bpm = 30.0  # Target a much lower frequency for visualization (30 BPM)
        self.is_beat = False
        self.last_beat_time = 0
        self.beat_interval = 2000  # in milliseconds (for 30 BPM)
        
        # Fixed flash duration (100ms)
        self.flash_duration = 100  # milliseconds
        
        # For the BPM label
        self.label_y = self.y + 10
        
        # For the blinking rectangle
        self.blink_rect_y = self.y + 10
        self.blink_rect_height = self.height - 20
        
        # For debugging
        self.frame_count = 0
        self.last_frame_time = 0
        
    def update(self, bpm_data):
        current_time = pygame.time.get_ticks()
        
        # Update actual BPM from data
        self.current_bpm = bpm_data["bpm"]
        
        # Calculate a scaled down BPM to make visualization more manageable
        # Scale down to target ~30 BPM for visualization
        bpm_scale_factor = 1
        if self.current_bpm > 80:
            bpm_scale_factor = 4  # Quarter time for fast tempos
        elif self.current_bpm > 40:
            bpm_scale_factor = 2  # Half time for moderate tempos
            
        self.scaled_bpm = self.current_bpm / bpm_scale_factor
        
        # Calculate beat interval in milliseconds based on scaled BPM
        self.beat_interval = 60000 / self.scaled_bpm if self.scaled_bpm > 0 else 2000
        
        # If this is the first update or we don't have a last beat time
        if self.last_beat_time == 0:
            self.last_beat_time = current_time
            
        # Calculate time since last beat
        time_since_last = current_time - self.last_beat_time
        
        # Debug frame rate
        self.frame_count += 1
        if current_time - self.last_frame_time > 1000:  # Every second
            # Print frame rate for debugging
            # print(f"FPS: {self.frame_count}, Beat interval: {self.beat_interval}ms")
            self.frame_count = 0
            self.last_frame_time = current_time
        
        # Check if we're due for a new beat based on the scaled BPM
        if time_since_last >= self.beat_interval:
            # Calculate how many beats we've missed (should generally be just 1)
            beats_missed = int(time_since_last / self.beat_interval)
            
            # Update last beat time to be exactly on the beat grid
            self.last_beat_time += beats_missed * self.beat_interval
            
            # We're on a beat
            self.is_beat = True
        else:
            # Check if we're within the flash duration
            self.is_beat = (time_since_last < self.flash_duration)
            
    def draw(self, surface):
        # Draw background
        pygame.draw.rect(
            surface, (20, 20, 30), (self.x, self.y, self.width, self.height)
        )
        pygame.draw.rect(
            surface, (50, 50, 60), (self.x, self.y, self.width, self.height), 1
        )
        
        # Draw BPM label showing both actual and scaled BPM
        bpm_text = f"BPM: {self.current_bpm:.1f} (Scaled: {self.scaled_bpm:.1f})"
        bpm_label = self.font.render(bpm_text, True, (200, 200, 200))
        surface.blit(bpm_label, (self.x + 10, self.label_y))
        
        # Draw blinking rectangle (right side of the BPM label)
        label_width = bpm_label.get_width()
        blink_rect_x = self.x + label_width + 40
        blink_rect_width = self.width - label_width - 60
        
        if self.is_beat:
            # Fully bright rectangle when on beat
            blink_color = COLORS["bpm"]
            
            # Draw filled rectangle
            pygame.draw.rect(
                surface, 
                blink_color, 
                (blink_rect_x, self.blink_rect_y, blink_rect_width, self.blink_rect_height)
            )
            
            # Draw a white border when blinking
            pygame.draw.rect(
                surface, 
                (255, 255, 255), 
                (blink_rect_x, self.blink_rect_y, blink_rect_width, self.blink_rect_height), 
                2
            )
        else:
            # Make this very dark when not blinking
            pygame.draw.rect(
                surface, 
                (5, 5, 8), 
                (blink_rect_x, self.blink_rect_y, blink_rect_width, self.blink_rect_height)
            )
            
            # Very subtle dark outline
            pygame.draw.rect(
                surface, 
                (10, 10, 15), 
                (blink_rect_x, self.blink_rect_y, blink_rect_width, self.blink_rect_height), 
                1
            )
            
        # Draw beat timing info
        ms_per_beat = f"{self.beat_interval:.0f}ms/beat"
        timing_label = pygame.font.Font(None, 20).render(ms_per_beat, True, (150, 150, 150))
        surface.blit(timing_label, (blink_rect_x + 5, self.blink_rect_y + 5))


class OnsetDetectionVisualizer:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.font = pygame.font.Font(None, 24)
        
        # Define onset methods
        self.onset_methods = ["energy", "hfc", "complex", "phase", "specflux", "kick", "hihat"]
        self.block_width = width // len(self.onset_methods)
        
        # Store active methods with smoothing
        self.active_levels = {method: 0 for method in self.onset_methods}
        self.decay_rate = 0.1   # Faster decay so lights go out quicker
        self.rise_rate = 0.5    # Faster rise rate for more responsive visualization
        
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
                        self.active_levels[method] = max(self.active_levels[method], target_level)
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
            
            if activity < 0.1:
                # Almost completely black when not active (even darker than before)
                block_color = (1, 1, 3)
            else:
                # Apply non-linear scaling for stronger contrast
                activity_scaled = activity ** 1.5  # Power of 1.5 for contrast
                scaled_r = int(base_color[0] * activity_scaled)
                scaled_g = int(base_color[1] * activity_scaled)
                scaled_b = int(base_color[2] * activity_scaled)
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
                (x + (block_size - label_width) // 2, block_y + block_size + 5)
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
    bpm_row_height = 60  # Height for the BPM visualizer row
    main_row_height = (HEIGHT - bpm_row_height - 60) // 2  # Height for the main rows
    
    # Row 0: BPM visualizer (top row)
    bpm_viz = BPMVisualizer(20, 20, WIDTH - 40, bpm_row_height)
    
    # Row 1: Spectrum visualizer (middle row)
    spectrum_viz = SpectrumVisualizer(20, bpm_row_height + 30, WIDTH - 40, main_row_height)
    
    # Row 2: Onset detection visualizer (bottom row)
    onset_viz = OnsetDetectionVisualizer(20, bpm_row_height + main_row_height + 40, WIDTH - 40, main_row_height)

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
            # Update BPM visualizer
            if "tempo" in latest_data:
                bpm_viz.update(latest_data["tempo"])
                
            # Update spectrum visualizer
            if "spectrum" in latest_data:
                spectrum_viz.update(latest_data["spectrum"])
            
            # Update onset detection visualizer
            onset_viz.update(latest_data)
            
            # Display timestamp in top right corner
            timestamp = datetime.fromisoformat(latest_data["timestamp"]).strftime(
                "%H:%M:%S"
            )
            ts_surface = small_font.render(f"Time: {timestamp}", True, (150, 150, 150))
            screen.blit(ts_surface, (WIDTH - 150, 20))

        # Draw visualization elements
        bpm_viz.draw(screen)
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