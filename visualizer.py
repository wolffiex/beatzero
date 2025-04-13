import pygame
import time

# Window setup
WIDTH, HEIGHT = 600, 600
FPS = 30
BACKGROUND_COLOR = (10, 10, 20)


def main():
    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("BeatZero Visualizer")
    clock = pygame.time.Clock()

    # Setup font for labels
    font = pygame.font.Font(None, 24)

    # BPM blinker configuration
    bpm = 58

    blink_state = False  # Start with blinker off
    next_transition_time = 0

    # Volume level (fixed for now)
    volume = 0.7  # Range 0-1

    # Main game loop
    running = True
    while running:
        current_time = pygame.time.get_ticks()

        # Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

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

        # Draw BPM blinker in top left
        bpm_text = font.render("BPM", True, (255, 255, 255))
        screen.blit(bpm_text, (20, 20))

        # Draw blinking box
        if blink_state:
            blink_color = (255, 255, 255)  # White when on
        else:
            blink_color = (50, 50, 70)  # Dark gray when off

        pygame.draw.rect(screen, blink_color, (75, 17, 20, 20))

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

        # Update the display
        pygame.display.flip()

        # Cap the frame rate
        clock.tick(FPS)

    # Clean up
    pygame.quit()


if __name__ == "__main__":
    main()
