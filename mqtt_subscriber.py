import json
import time
from datetime import datetime
import paho.mqtt.client as mqtt
from rich.console import Console
from rich.live import Live
from rich.table import Table

# MQTT parameters
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC = "beatzero/music_detection"
MQTT_CLIENT_ID = f"beatzero-subscriber-{int(time.time())}"

# Set up Rich console for real-time display
console = Console()

# Method descriptions for display
METHOD_DESCRIPTIONS = {
    "energy": "Energy (loudness)",
    "hfc": "High Frequency (treble)",
    "complex": "Complex (tonal changes)",
    "phase": "Phase (timing changes)",
    "specflux": "Spectral Flux (overall)"
}

# Initialize data storage for display
latest_data = None

def on_connect(client, userdata, flags, rc):
    """Callback for when the client connects to the broker"""
    if rc == 0:
        console.print(f"[bold green]Connected to MQTT broker[/bold green]")
        client.subscribe(MQTT_TOPIC)
        console.print(f"[bold green]Subscribed to topic: {MQTT_TOPIC}[/bold green]")
    else:
        console.print(f"[bold red]Failed to connect to MQTT broker with code: {rc}[/bold red]")

def on_message(client, userdata, msg):
    """Callback for when a message is received from the broker"""
    global latest_data
    try:
        latest_data = json.loads(msg.payload.decode())
    except Exception as e:
        console.print(f"[bold red]Error parsing message: {e}[/bold red]")

def create_display_table(data):
    """Create a table for displaying the latest music detection data"""
    if not data:
        table = Table(title="Waiting for music detection data...")
        return table
    
    table = Table(title=f"Music Detection Data - {data['timestamp']}")
    
    table.add_column("Feature", style="cyan")
    table.add_column("Value", style="green", width=15, justify="right")
    
    # Beat detection from all methods
    for method, method_desc in METHOD_DESCRIPTIONS.items():
        if method in data["onsets"]:
            onset_data = data["onsets"][method]
            is_beat = onset_data["is_beat"]
            beat_display = "[bold green]YES[/bold green]" if is_beat else "[dim]no[/dim]"
            
            # Use color coding based on beat detection
            method_color = "green" if is_beat else "cyan"
            
            table.add_row(
                f"[{method_color}]{method_desc:<15}[/{method_color}]", 
                f"{beat_display:>15}"
            )
    
    # Frequency visualization
    pitch = data["pitch"]["value"]
    pitch_confidence = data["pitch"]["confidence"]
    note_detected = data["note_detected"]
    
    # Create frequency band visualization
    freq_ranges = [
        (20, 80),      # Sub-bass
        (80, 250),     # Bass
        (250, 500),    # Low-mids
        (500, 1000),   # Mids
        (1000, 2000),  # Upper-mids
        (2000, 3000),  # Presence
        (3000, 4000),  # Brilliance
        (4000, 8000)   # Air/Ultra high
    ]
    
    note_blocks = ""
    highlight_note = note_detected and pitch_confidence > 0.4
    for i, (min_freq, max_freq) in enumerate(freq_ranges):
        if min_freq <= pitch < max_freq:
            if highlight_note:
                note_blocks += "■ "  # Filled block
            else:
                note_blocks += "▣ "  # Half-filled block
        else:
            note_blocks += "□ "  # Empty block
    
    table.add_row("Freq Bands".ljust(15), f"[blue]{note_blocks}[/blue]".rjust(15))
    
    # Kick drum detection
    kick_detected = data["kick_detected"]
    if kick_detected:
        kick_indicator = "[bold red]⚫ KICK ⚫[/bold red]"
    else:
        kick_indicator = "[dim]----------[/dim]"
    table.add_row("Kick".ljust(15), f"{kick_indicator}".rjust(15))
    
    # Hi-hat detection
    hihat_detected = data["hihat_detected"]
    if hihat_detected:
        hihat_indicator = f"[bold yellow]✧✧ +++ ✧✧[/bold yellow]"
    else:
        hihat_indicator = "[dim]----------[/dim]"
    table.add_row("Hi-Hat".ljust(15), f"{hihat_indicator}".rjust(15))
    
    # BPM from tempo estimation
    bpm = data["tempo"]["bpm"]
    bpm_color = "green" if bpm > 10 else "dim"
    table.add_row("BPM".ljust(15), f"[{bpm_color}]{bpm:6.1f}[/{bpm_color}]".rjust(15))
    
    return table

def main():
    """Main function to run the MQTT subscriber"""
    console.print("[bold green]BeatZero MQTT Subscriber[/bold green]")
    console.print("[bold]Waiting for music detection data... Press Ctrl+C to stop[/bold]")
    
    # Initialize MQTT client
    client = mqtt.Client(client_id=MQTT_CLIENT_ID)
    client.on_connect = on_connect
    client.on_message = on_message
    
    # Connect to MQTT broker
    try:
        client.connect(MQTT_BROKER, MQTT_PORT)
    except Exception as e:
        console.print(f"[bold red]Failed to connect to MQTT broker: {e}[/bold red]")
        return
    
    # Start the MQTT client loop in a background thread
    client.loop_start()
    
    # Use Rich's Live display for real-time updates
    try:
        with Live(refresh_per_second=10) as live:
            while True:
                table = create_display_table(latest_data)
                live.update(table)
                time.sleep(0.1)
    except KeyboardInterrupt:
        console.print("[bold red]Stopping...[/bold red]")
    finally:
        # Clean up
        client.loop_stop()
        client.disconnect()
        console.print("[bold green]Stopped[/bold green]")

if __name__ == "__main__":
    main()