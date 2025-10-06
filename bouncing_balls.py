import sys
import math
import random
import pygame
import numpy as np

# ---------------------------
# Config
# ---------------------------
WIDTH, HEIGHT = 900, 600
FPS = 120
BG_COLOR = (18, 18, 24)
FLOOR_COLOR = (35, 40, 55)
TEXT_COLOR = (220, 220, 220)

RADIUS = 22
REST_COEFF = 0.80          # bounciness vs walls/floor
AIR_DRAG = 0.000
GROUND_FRICTION = 5.0

GRAVITY_ON = True
GRAVITY = 1400.0

N_BALLS = 8                 # try 8 to start

# ---------------------------
# Helpers
# ---------------------------
def draw_text(surface, text, x, y, font):
    img = font.render(text, True, TEXT_COLOR)
    surface.blit(img, (x, y))

def make_balls(n):
    """
    Returns positions (n,2) and velocities (n,2) arrays and a list of colors.
    Spawns balls not too close to walls/floor.
    """
    rng = np.random.default_rng()
    margin = RADIUS + 5
    pos = np.zeros((n, 2), dtype=float)
    vel = np.zeros((n, 2), dtype=float)
    colors = []
    for i in range(n):
        pos[i, 0] = rng.uniform(margin, WIDTH - margin)
        pos[i, 1] = rng.uniform(margin, HEIGHT * 0.35)   # spawn upper-ish
        vel[i, 0] = rng.uniform(-300, 300)
        vel[i, 1] = rng.uniform(-50, 50)
        # pastel-ish random colors
        colors.append(tuple(int(c) for c in rng.uniform(140, 255, size=3)))
    return pos, vel, colors

def resolve_ball_ball_collisions(pos, vel, radius=RADIUS, e=REST_COEFF):
    """
    Pairwise elastic collisions between equal-mass circles.
    - Separates interpenetrations with positional correction.
    - Applies impulse along the collision normal when approaching.
    """
    n = pos.shape[0]
    min_dist = 2.0 * radius
    min_dist2 = min_dist * min_dist
    for i in range(n):
        for j in range(i + 1, n):
            delta = pos[j] - pos[i]
            d2 = float(delta[0] * delta[0] + delta[1] * delta[1])
            if d2 < min_dist2:
                # avoid div by zero
                d = math.sqrt(d2) if d2 > 1e-12 else 1e-6
                n_hat = delta / d

                # positional correction: split the overlap
                overlap = min_dist - d
                corr = 0.5 * overlap * n_hat
                pos[i] -= corr
                pos[j] += corr

                # relative velocity along normal
                rel = vel[j] - vel[i]
                vn = float(rel[0] * n_hat[0] + rel[1] * n_hat[1])
                if vn < 0.0:
                    # equal masses -> impulse split equally
                    j_imp = -(1.0 + e) * vn * 0.5
                    impulse = j_imp * n_hat
                    vel[i] -= impulse
                    vel[j] += impulse

# ---------------------------
# Main
# ---------------------------
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Day 2 Task 1: Multi-ball (with inter-collisions)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 18)

    # State
    pos, vel, colors = make_balls(N_BALLS)
    paused = False
    gravity_on = GRAVITY_ON

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0

        # ---- events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    pos, vel, colors = make_balls(N_BALLS)
                    gravity_on = True
                elif event.key == pygame.K_g:
                    gravity_on = not gravity_on
                elif event.key == pygame.K_LEFT:
                    vel[:, 0] -= 200.0
                elif event.key == pygame.K_RIGHT:
                    vel[:, 0] += 200.0
                elif event.key == pygame.K_UP:
                    # jump those that are near ground
                    grounded = np.isclose(pos[:, 1] + RADIUS, HEIGHT, atol=3.0)
                    vel[grounded, 1] = -700.0

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # Move the nearest ball to the mouse; zero its velocity
                mx, my = pygame.mouse.get_pos()
                d2 = (pos[:, 0] - mx) ** 2 + (pos[:, 1] - my) ** 2
                i = int(np.argmin(d2))
                pos[i] = np.array([mx, my], dtype=float)
                vel[i] = np.array([0.0, 0.0], dtype=float)

        # ---- physics
        if not paused:
            if gravity_on:
                vel[:, 1] += GRAVITY * dt

            if AIR_DRAG > 0.0:
                vel -= vel * AIR_DRAG * dt

            # integrate
            pos += vel * dt

            # walls (left/right)
            hit_left = pos[:, 0] - RADIUS < 0
            pos[hit_left, 0] = RADIUS
            vel[hit_left, 0] = -vel[hit_left, 0] * REST_COEFF

            hit_right = pos[:, 0] + RADIUS > WIDTH
            pos[hit_right, 0] = WIDTH - RADIUS
            vel[hit_right, 0] = -vel[hit_right, 0] * REST_COEFF

            # ceiling
            hit_top = pos[:, 1] - RADIUS < 0
            pos[hit_top, 1] = RADIUS
            vel[hit_top, 1] = -vel[hit_top, 1] * REST_COEFF

            # floor
            hit_floor = pos[:, 1] + RADIUS > HEIGHT
            pos[hit_floor, 1] = HEIGHT - RADIUS
            falling = hit_floor & (vel[:, 1] > 0)
            vel[falling, 1] = -vel[falling, 1] * REST_COEFF

            # kill tiny vertical jitter & apply ground friction where grounded
            grounded = np.isclose(pos[:, 1] + RADIUS, HEIGHT, atol=2.5)
            small_vy = np.abs(vel[:, 1]) < 50
            settle = grounded & small_vy
            vel[settle, 1] = 0.0
            # exponential horizontal friction
            vel[grounded, 0] *= np.maximum(0.0, 1.0 - GROUND_FRICTION * dt)
            vel[np.abs(vel[:, 0]) < 1.0, 0] = 0.0

            # ---- NEW: ball–ball collisions
            resolve_ball_ball_collisions(pos, vel, RADIUS, REST_COEFF)

        # ---- draw
        screen.fill(BG_COLOR)
        pygame.draw.rect(screen, FLOOR_COLOR, pygame.Rect(0, HEIGHT - 6, WIDTH, 6))
        for i in range(len(colors)):
            pygame.draw.circle(screen, colors[i], (int(pos[i, 0]), int(pos[i, 1])), RADIUS)

        draw_text(screen, f"balls={len(colors)}  g={'ON' if gravity_on else 'OFF'}  e={REST_COEFF:.2f}  friction={GROUND_FRICTION:.1f}  FPS~{clock.get_fps():.0f}", 10, 10, font)
        draw_text(screen, "Space=pause  R=reset  G=toggle gravity  ←/→ impulses  ↑ jump(grounded)  LMB=move nearest", 10, 32, font)

        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
