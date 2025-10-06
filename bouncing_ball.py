import sys
import math
import pygame
import numpy as np

# -----------------------------------
# Config
# -----------------------------------
WIDTH, HEIGHT = 900, 600
FPS = 120  # high FPS for smoother physics
BG_COLOR = (18, 18, 24)
BALL_COLOR = (200, 230, 255)
FLOOR_COLOR = (35, 40, 55)
TEXT_COLOR = (220, 220, 220)

RADIUS = 22
REST_COEFF = 0.80    # coefficient of restitution (bounciness)
AIR_DRAG = 0.000     # optional air drag (0 = off)
GROUND_FRICTION = 5.0  # horizontal damping when on ground

GRAVITY_ON = True
GRAVITY = 1400.0     # px/s^2 downward

# -----------------------------------
# Helpers
# -----------------------------------
def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def draw_text(surface, text, x, y, font):
    img = font.render(text, True, TEXT_COLOR)
    surface.blit(img, (x, y))

# -----------------------------------
# Main
# -----------------------------------
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Day 1 Task 2: Bouncing Ball (Pygame + NumPy)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 18)

    # State
    pos = np.array([WIDTH * 0.25, HEIGHT * 0.25], dtype=float)
    vel = np.array([280.0, 20.0], dtype=float)
    paused = False
    gravity_on = GRAVITY_ON

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0  # seconds since last frame

        # -----------------------------------
        # Events / Controls
        # -----------------------------------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # left click: reposition ball
                    pos = np.array(pygame.mouse.get_pos(), dtype=float)
                    vel = np.array([0.0, 0.0], dtype=float)

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    pos = np.array([WIDTH * 0.25, HEIGHT * 0.25], dtype=float)
                    vel = np.array([280.0, 20.0], dtype=float)
                    gravity_on = True
                elif event.key == pygame.K_g:
                    gravity_on = not gravity_on
                elif event.key == pygame.K_LEFT:
                    vel[0] -= 200.0
                elif event.key == pygame.K_RIGHT:
                    vel[0] += 200.0
                elif event.key == pygame.K_UP:
                    # launch if close to ground
                    if abs((HEIGHT - RADIUS) - pos[1]) < 2.5:
                        vel[1] = -700.0

        # -----------------------------------
        # Physics
        # -----------------------------------
        if not paused:
            # Gravity
            if gravity_on:
                vel[1] += GRAVITY * dt

            # Air drag (very light)
            if AIR_DRAG > 0.0:
                vel -= vel * AIR_DRAG * dt

            # Integrate
            pos += vel * dt

            # Collisions: walls
            if pos[0] - RADIUS < 0:
                pos[0] = RADIUS
                vel[0] = -vel[0] * REST_COEFF
            elif pos[0] + RADIUS > WIDTH:
                pos[0] = WIDTH - RADIUS
                vel[0] = -vel[0] * REST_COEFF

            # Ceiling
            if pos[1] - RADIUS < 0:
                pos[1] = RADIUS
                vel[1] = -vel[1] * REST_COEFF

            # Floor
            if pos[1] + RADIUS > HEIGHT:
                pos[1] = HEIGHT - RADIUS
                if vel[1] > 0:
                    vel[1] = -vel[1] * REST_COEFF

                # If near-rest on floor, kill tiny vertical jitter and apply friction
                if abs(vel[1]) < 50:
                    vel[1] = 0.0
                    # exponential friction toward zero when grounded
                    vel[0] *= max(0.0, 1.0 - GROUND_FRICTION * dt)
                    if abs(vel[0]) < 1.0:
                        vel[0] = 0.0

        # -----------------------------------
        # Draw
        # -----------------------------------
        screen.fill(BG_COLOR)

        # Floor line
        pygame.draw.rect(screen, FLOOR_COLOR, pygame.Rect(0, HEIGHT-6, WIDTH, 6))

        # Ball
        pygame.draw.circle(screen, BALL_COLOR, (int(pos[0]), int(pos[1])), RADIUS)

        # HUD
        draw_text(screen, f"pos=({pos[0]:7.1f}, {pos[1]:7.1f})  vel=({vel[0]:6.1f}, {vel[1]:6.1f})", 10, 10, font)
        draw_text(screen, f"g={'ON ' if gravity_on else 'OFF'}  e={REST_COEFF:.2f}  friction={GROUND_FRICTION:.1f}  FPS~{clock.get_fps():.0f}", 10, 32, font)
        draw_text(screen, "Space=pause  R=reset  G=toggle gravity  ←/→ impulse  ↑ jump(if grounded)  LMB=move ball", 10, 54, font)

        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
# --- launcher & safe fallback ---
if __name__ == "__main__":
    try:
        # run your main() if it exists
        main()
    except NameError:
        # Fallback: tiny window so something always appears even if main() is missing
        import sys, pygame
        pygame.init()
        screen = pygame.display.set_mode((900, 600))
        pygame.display.set_caption("Bouncing Ball – Fallback (add def main()!)")
        clock = pygame.time.Clock()
        running = True
        while running:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
            screen.fill((18, 18, 24))
            pygame.display.flip()
            clock.tick(60)
        pygame.quit()
        sys.exit(0)
