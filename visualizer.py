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

    # Setup font for BPM label
    font = pygame.font.Font(None, 24)

    bpm = 58

    blink_state = False  # Start with blinker off
    next_transition_time = 0

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

        # Update the display
        pygame.display.flip()

        # Cap the frame rate
        clock.tick(FPS)

    # Clean up
    pygame.quit()


if __name__ == "__main__":
    main()
