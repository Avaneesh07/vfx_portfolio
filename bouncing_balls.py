import sys
import math
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

REST_COEFF = 0.80
AIR_DRAG = 0.000
GROUND_FRICTION = 5.0

GRAVITY_ON = True
GRAVITY = 1400.0

N_BALLS = 8  # initial count

# ---------------------------
# Helpers
# ---------------------------
def draw_text(surface, text, x, y, font):
    surface.blit(font.render(text, True, TEXT_COLOR), (x, y))

def make_balls(n):
    """
    Returns (pos, vel, colors, radii)
    pos, vel: (n,2) float arrays
    colors: list[(r,g,b)]
    radii: (n,) int array
    """
    rng = np.random.default_rng()
    radii = rng.integers(14, 28, size=n).astype(int)  # 14..27 px
    pos = np.zeros((n, 2), dtype=float)
    vel = np.zeros((n, 2), dtype=float)
    colors = []
    for i in range(n):
        margin = radii[i] + 5
        pos[i, 0] = rng.uniform(margin, WIDTH - margin)
        pos[i, 1] = rng.uniform(margin, HEIGHT * 0.35)  # spawn upper-ish
        vel[i, 0] = rng.uniform(-300, 300)
        vel[i, 1] = rng.uniform(-50, 50)
        colors.append(tuple(int(c) for c in rng.uniform(140, 255, size=3)))
    return pos, vel, colors, radii

def lighten(c, amt):
    # amt added to each channel, clamped
    r = min(255, c[0] + amt)
    g = min(255, c[1] + amt)
    b = min(255, c[2] + amt)
    return (int(r), int(g), int(b))

def resolve_ball_ball_collisions_var(pos, vel, radii, e, flash=None):
    """
    Elastic collisions for possibly different radii (equal mass assumed).
    Includes positional correction to separate overlaps.
    If 'flash' array is provided, sets flash timers on impact.
    """
    n = pos.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            delta = pos[j] - pos[i]
            d2 = float(delta[0]*delta[0] + delta[1]*delta[1])
            min_dist = float(radii[i] + radii[j])
            min_dist2 = min_dist * min_dist
            if d2 < min_dist2:
                d = math.sqrt(d2) if d2 > 1e-12 else 1e-6
                n_hat = delta / d
                overlap = min_dist - d

                # positional correction: split overlap
                corr = 0.5 * overlap * n_hat
                pos[i] -= corr
                pos[j] += corr

                # normal component of relative velocity
                rel = vel[j] - vel[i]
                vn = float(rel[0]*n_hat[0] + rel[1]*n_hat[1])
                if vn < 0.0:
                    # equal-mass impulse split
                    j_imp = -(1.0 + e) * vn * 0.5
                    impulse = j_imp * n_hat
                    vel[i] -= impulse
                    vel[j] += impulse

                    if flash is not None:
                        flash[i] = 0.12  # seconds
                        flash[j] = 0.12

def add_ball_at_mouse(pos, vel, colors, radii, flash):
    mx, my = pygame.mouse.get_pos()
    pos = np.vstack([pos, np.array([mx, my], dtype=float)])
    vel = np.vstack([vel, np.array([0.0, 0.0], dtype=float)])
    rng = np.random.default_rng()
    new_color = tuple(int(c) for c in rng.uniform(140, 255, size=3))
    colors.append(new_color)
    new_r = int(rng.integers(14, 28))
    radii = np.append(radii, new_r).astype(int)
    flash = np.append(flash, 0.0)
    return pos, vel, colors, radii, flash

# ---------------------------
# Main
# ---------------------------
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Day 3: Radii, Impact Flash, Right-Click Spawn")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 18)

    pos, vel, colors, radii = make_balls(N_BALLS)
    flash = np.zeros_like(radii, dtype=float)  # flash timers per ball
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
                    pos, vel, colors, radii = make_balls(N_BALLS)
                    flash = np.zeros_like(radii, dtype=float)
                    gravity_on = True
                elif event.key == pygame.K_g:
                    gravity_on = not gravity_on
                elif event.key == pygame.K_LEFT:
                    vel[:, 0] -= 200.0
                elif event.key == pygame.K_RIGHT:
                    vel[:, 0] += 200.0
                elif event.key == pygame.K_UP:
                    grounded = np.isclose(pos[:, 1] + radii, HEIGHT, atol=3.0)
                    vel[grounded, 1] = -700.0

            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # left: move nearest
                    mx, my = pygame.mouse.get_pos()
                    d2 = (pos[:, 0] - mx) ** 2 + (pos[:, 1] - my) ** 2
                    i = int(np.argmin(d2))
                    pos[i] = np.array([mx, my], dtype=float)
                    vel[i] = np.array([0.0, 0.0], dtype=float)
                elif event.button == 3:  # right: spawn new ball
                    pos, vel, colors, radii, flash = add_ball_at_mouse(pos, vel, colors, radii, flash)

        # ---- physics
        if not paused:
            if gravity_on:
                vel[:, 1] += GRAVITY * dt

            if AIR_DRAG > 0.0:
                vel -= vel * AIR_DRAG * dt

            # integrate
            pos += vel * dt

            # walls (left/right)
            hit_left = pos[:, 0] - radii < 0
            pos[hit_left, 0] = radii[hit_left]
            vel[hit_left, 0] = -vel[hit_left, 0] * REST_COEFF

            hit_right = pos[:, 0] + radii > WIDTH
            pos[hit_right, 0] = WIDTH - radii[hit_right]
            vel[hit_right, 0] = -vel[hit_right, 0] * REST_COEFF

            # ceiling
            hit_top = pos[:, 1] - radii < 0
            pos[hit_top, 1] = radii[hit_top]
            vel[hit_top, 1] = -vel[hit_top, 1] * REST_COEFF

            # floor
            hit_floor = pos[:, 1] + radii > HEIGHT
            pos[hit_floor, 1] = HEIGHT - radii[hit_floor]
            falling = hit_floor & (vel[:, 1] > 0)
            vel[falling, 1] = -vel[falling, 1] * REST_COEFF

            # grounded friction & settle
            grounded = np.isclose(pos[:, 1] + radii, HEIGHT, atol=2.5)
            small_vy = np.abs(vel[:, 1]) < 50
            settle = grounded & small_vy
            vel[settle, 1] = 0.0
            vel[grounded, 0] *= np.maximum(0.0, 1.0 - GROUND_FRICTION * dt)
            vel[np.abs(vel[:, 0]) < 1.0, 0] = 0.0

            # ball-ball collisions + flash
            resolve_ball_ball_collisions_var(pos, vel, radii, REST_COEFF, flash)

            # decay flash timers
            flash = np.maximum(0.0, flash - dt)

        # ---- draw
        screen.fill(BG_COLOR)
        pygame.draw.rect(screen, FLOOR_COLOR, pygame.Rect(0, HEIGHT - 6, WIDTH, 6))

        for i, c in enumerate(colors):
            draw_c = c
            if flash[i] > 0:
                amt = int(120 * (flash[i] / 0.12))  # 0..120 brighten
                draw_c = lighten(c, amt)
            pygame.draw.circle(screen, draw_c, (int(pos[i, 0]), int(pos[i, 1])), int(radii[i]))

        draw_text(
            screen,
            f"balls={len(colors)}  g={'ON' if gravity_on else 'OFF'}  e={REST_COEFF:.2f}  friction={GROUND_FRICTION:.1f}  FPS~{clock.get_fps():.0f}",
            10, 10, font
        )
        draw_text(
            screen,
            "Space=pause  R=reset  G=toggle g  ←/→ impulses  ↑ jump(grounded)  LMB=move  RMB=spawn",
            10, 32, font
        )

        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
