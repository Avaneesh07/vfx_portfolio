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
GEOM_COLOR = (0, 255, 180)    # brighter teal for visibility

REST_COEFF = 0.80
AIR_DRAG = 0.15               # gentle linear drag
GROUND_FRICTION = 5.0

GRAVITY_ON = True
GRAVITY = 1400.0

N_BALLS = 8                   # initial count
TRAIL_FADE_ALPHA = 55         # stronger fade so lines stay clear
DENSITY = 1.0                 # mass ~ r^2

# ---------------------------
# Helpers
# ---------------------------
def draw_text(surface, text, x, y, font):
    surface.blit(font.render(text, True, TEXT_COLOR), (x, y))

def lighten(c, amt):
    r = min(255, c[0] + amt)
    g = min(255, c[1] + amt)
    b = min(255, c[2] + amt)
    return (int(r), int(g), int(b))

def mass_from_radius(r):
    return DENSITY * float(r) * float(r)

def make_balls(n):
    """
    Returns (pos, vel, colors, radii, masses, inv_masses)
    """
    rng = np.random.default_rng()
    radii = rng.integers(14, 28, size=n).astype(int)
    pos = np.zeros((n, 2), dtype=float)
    vel = np.zeros((n, 2), dtype=float)
    colors = []
    for i in range(n):
        margin = radii[i] + 5
        pos[i, 0] = rng.uniform(margin, WIDTH - margin)
        pos[i, 1] = rng.uniform(margin, HEIGHT * 0.35)
        vel[i, 0] = rng.uniform(-300, 300)
        vel[i, 1] = rng.uniform(-50, 50)
        colors.append(tuple(int(c) for c in rng.uniform(140, 255, size=3)))
    masses = np.array([mass_from_radius(r) for r in radii], dtype=float)
    inv_masses = 1.0 / masses
    return pos, vel, colors, radii, masses, inv_masses

# ---------------------------
# Geometry: static line segments
# ---------------------------
class Segment:
    __slots__ = ("a", "b", "e", "fric")
    def __init__(self, ax, ay, bx, by, restitution=0.80, friction=0.05):
        self.a = np.array([ax, ay], dtype=float)
        self.b = np.array([bx, by], dtype=float)
        self.e = float(restitution)
        self.fric = float(friction)

def closest_point_on_segment(a, b, p):
    ab = b - a
    ab2 = float(ab[0]*ab[0] + ab[1]*ab[1])
    if ab2 <= 1e-12:
        return a.copy(), 0.0
    t = float((p[0]-a[0])*ab[0] + (p[1]-a[1])*ab[1]) / ab2
    t_clamped = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    q = a + t_clamped * ab
    return q, t_clamped

def collide_ball_with_segment(pos_i, vel_i, radius_i, seg):
    """
    Circle (pos_i, radius_i) vs static segment seg.
    Push ball out along normal; reflect/dampen normal velocity; apply tangential friction.
    Returns True if a collision occurred.
    """
    q, _ = closest_point_on_segment(seg.a, seg.b, pos_i)
    d = pos_i - q
    d2 = float(d[0]*d[0] + d[1]*d[1])
    r = float(radius_i)
    if d2 >= r*r:
        return False

    dist = math.sqrt(d2) if d2 > 1e-12 else 1e-6
    n_hat = d / dist  # outward normal from surface toward ball center
    overlap = r - dist

    # Positional correction: move ball only (segment is static "infinite mass")
    pos_i += n_hat * overlap

    # Velocity split into normal/tangent
    vn = float(vel_i[0]*n_hat[0] + vel_i[1]*n_hat[1])
    vt = vel_i - vn * n_hat

    # Bounce on normal with surface restitution
    new_vn = -seg.e * vn
    # Tangential friction (simple damping)
    new_vt = vt * max(0.0, 1.0 - seg.fric)

    vel_i[:] = new_vt + new_vn * n_hat
    return True

def build_level(level_id):
    """
    Returns (segments, level_name)
    Two example layouts:
      1) Ramp + platform
      2) Funnel + ledge
    """
    segs = []
    if level_id == 1:
        name = "Ramp + Platform"
        # Ramp (bottom-left up)
        segs.append(Segment(80, HEIGHT-120, 360, HEIGHT-40, restitution=0.80, friction=0.08))
        # Opposing ramp (bottom-right up)
        segs.append(Segment(WIDTH-80, HEIGHT-120, WIDTH-360, HEIGHT-40, restitution=0.80, friction=0.08))
        # Mid-air horizontal platform
        segs.append(Segment(450, 340, 820, 340, restitution=0.75, friction=0.05))
    else:
        name = "Funnel + Ledge"
        # Funnel
        segs.append(Segment(40, 120, 380, 320, restitution=0.80, friction=0.06))
        segs.append(Segment(WIDTH-40, 120, WIDTH-380, 320, restitution=0.80, friction=0.06))
        # Small ledge
        segs.append(Segment(420, 430, 560, 430, restitution=0.75, friction=0.05))
    return segs, name

def draw_segments(surface, segs):
    for s in segs:
        pygame.draw.line(surface, GEOM_COLOR, (int(s.a[0]), int(s.a[1])), (int(s.b[0]), int(s.b[1])), 4)  # thicker

# ---------------------------
# Ball–ball collisions (mass-aware)
# ---------------------------
def resolve_ball_ball_collisions_mass(pos, vel, radii, masses, inv_masses, e, flash=None):
    n = pos.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            delta = pos[j] - pos[i]
            d2 = float(delta[0]*delta[0] + delta[1]*delta[1])
            min_dist = float(radii[i] + radii[j])
            if d2 < min_dist*min_dist:
                d = math.sqrt(d2) if d2 > 1e-12 else 1e-6
                n_hat = delta / d
                overlap = min_dist - d

                inv_m1 = inv_masses[i]
                inv_m2 = inv_masses[j]
                inv_sum = inv_m1 + inv_m2
                if inv_sum <= 0.0:
                    continue

                # Mass-weighted separation
                corr_vec = overlap * n_hat
                pos[i] -= (inv_m1 / inv_sum) * corr_vec
                pos[j] += (inv_m2 / inv_sum) * corr_vec

                # Normal impulse
                rel = vel[j] - vel[i]
                vn = float(rel[0]*n_hat[0] + rel[1]*n_hat[1])
                if vn < 0.0:
                    j_imp = -(1.0 + e) * vn / inv_sum
                    impulse = j_imp * n_hat
                    vel[i] -= inv_m1 * impulse
                    vel[j] += inv_m2 * impulse

                    if flash is not None:
                        flash[i] = 0.12
                        flash[j] = 0.12

def add_ball_at_mouse(pos, vel, colors, radii, masses, inv_masses, flash):
    mx, my = pygame.mouse.get_pos()
    pos = np.vstack([pos, np.array([mx, my], dtype=float)])
    vel = np.vstack([vel, np.array([0.0, 0.0], dtype=float)])
    rng = np.random.default_rng()
    new_color = tuple(int(c) for c in rng.uniform(140, 255, size=3))
    colors.append(new_color)
    new_r = int(rng.integers(14, 28))
    radii = np.append(radii, new_r).astype(int)
    new_m = mass_from_radius(new_r)
    masses = np.append(masses, new_m)
    inv_masses = np.append(inv_masses, 1.0 / new_m)
    flash = np.append(flash, 0.0)
    return pos, vel, colors, radii, masses, inv_masses, flash

# ---------------------------
# Main
# ---------------------------
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Day 6: Level Geometry (ramps, platforms) + collisions")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 18)

    # Trail layer + fade rect
    trail = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    fade_rect = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    fade_rect.fill((0, 0, 0, TRAIL_FADE_ALPHA))

    pos, vel, colors, radii, masses, inv_masses = make_balls(N_BALLS)
    flash = np.zeros_like(radii, dtype=float)

    paused = False
    gravity_on = GRAVITY_ON

    # Smooth HUD FPS
    fps_ema = 0.0
    ema_alpha = 0.12

    # Geometry / levels
    level_id = 1
    segments, level_name = build_level(level_id)
    show_geom = True

    running = True
    while running:
        dt_ms = clock.tick(FPS)
        dt = dt_ms / 1000.0

        # ---- events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    pos, vel, colors, radii, masses, inv_masses = make_balls(N_BALLS)
                    flash = np.zeros_like(radii, dtype=float)
                    trail.fill((0, 0, 0, 0))
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
                elif event.key == pygame.K_l:
                    # cycle levels 1 <-> 2
                    level_id = 2 if level_id == 1 else 1
                    segments, level_name = build_level(level_id)
                elif event.key == pygame.K_h:
                    show_geom = not show_geom

            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # left: move nearest
                    mx, my = pygame.mouse.get_pos()
                    d2 = (pos[:, 0] - mx) ** 2 + (pos[:, 1] - my) ** 2
                    i = int(np.argmin(d2))
                    pos[i] = np.array([mx, my], dtype=float)
                    vel[i] = np.array([0.0, 0.0], dtype=float)
                elif event.button == 3:  # right: spawn
                    pos, vel, colors, radii, masses, inv_masses, flash = add_ball_at_mouse(
                        pos, vel, colors, radii, masses, inv_masses, flash
                    )

        # ---- physics
        if not paused:
            if gravity_on:
                vel[:, 1] += GRAVITY * dt

            if AIR_DRAG > 0.0:
                vel -= vel * AIR_DRAG * dt

            # integrate
            pos += vel * dt

            # boundary walls/ceiling/floor
            hit_left = pos[:, 0] - radii < 0
            pos[hit_left, 0] = radii[hit_left]
            vel[hit_left, 0] = -vel[hit_left, 0] * REST_COEFF

            hit_right = pos[:, 0] + radii > WIDTH
            pos[hit_right, 0] = WIDTH - radii[hit_right]
            vel[hit_right, 0] = -vel[hit_right, 0] * REST_COEFF

            hit_top = pos[:, 1] - radii < 0
            pos[hit_top, 1] = radii[hit_top]
            vel[hit_top, 1] = -vel[hit_top, 1] * REST_COEFF

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

            # --- collide with segments (static geometry)
            for i in range(len(radii)):
                for seg in segments:
                    if collide_ball_with_segment(pos[i], vel[i], radii[i], seg):
                        flash[i] = 0.12  # flash on surface impact

            # mass-aware ball-ball collisions
            resolve_ball_ball_collisions_mass(pos, vel, radii, masses, inv_masses, REST_COEFF, flash)

            # decay flash timers
            flash = np.maximum(0.0, flash - dt)

        # ---- draw
        # 1) background + floor
        screen.fill(BG_COLOR)
        pygame.draw.rect(screen, FLOOR_COLOR, pygame.Rect(0, HEIGHT - 6, WIDTH, 6))

        # 2) trails layer (under geometry)
        trail.blit(fade_rect, (0, 0))
        for i, c in enumerate(colors):
            draw_c = c
            if flash[i] > 0:
                amt = int(120 * (flash[i] / 0.12))
                draw_c = lighten(c, amt)
            pygame.draw.circle(trail, draw_c, (int(pos[i, 0]), int(pos[i, 1])), int(radii[i]))
        screen.blit(trail, (0, 0))

        # 3) draw geometry LAST (on top so it's always visible)
        if show_geom:
            draw_segments(screen, segments)

        # 4) HUD
        dt_ms_safe = max(1, dt_ms)
        inst_fps = 1000.0 / dt_ms_safe
        fps_ema = (1 - ema_alpha) * fps_ema + ema_alpha * inst_fps

        momentum = (masses[:, None] * vel).sum(axis=0)
        px, py = float(momentum[0]), float(momentum[1])

        draw_text(
            screen,
            f"lvl={level_name}  geom={'ON' if show_geom else 'OFF'}  segs={len(segments)}  balls={len(colors)}  g={'ON' if gravity_on else 'OFF'}  e={REST_COEFF:.2f}  drag={AIR_DRAG:.2f}  FPS~{fps_ema:5.1f}",
            10, 10, font
        )
        draw_text(
            screen,
            f"Σp=({px:8.1f}, {py:8.1f})",
            10, 32, font
        )
        draw_text(
            screen,
            "Space=pause  R=reset  G=toggle g  ←/→ impulses  ↑ jump  LMB=move  RMB=spawn  L=cycle level  H=toggle geom",
            10, 54, font
        )

        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
