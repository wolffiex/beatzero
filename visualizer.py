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
        
        # Draw BPM blinker in top right
        bpm_text = font.render("BPM", True, (255, 255, 255))
        screen.blit(bpm_text, (20, 20))
        
        # Draw white box that's always lit
        pygame.draw.rect(
            screen,
            (255, 255, 255),
            (75, 17, 20, 20)
        )
        
        # Update the display
        pygame.display.flip()
        
        # Cap the frame rate
        clock.tick(FPS)
    
    # Clean up
    pygame.quit()

if __name__ == "__main__":
    main()
