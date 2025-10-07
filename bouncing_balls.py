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
GEOM_COLOR = (0, 255, 180)    # bright teal

REST_COEFF = 0.80
AIR_DRAG = 0.15               # gentle linear drag
GROUND_FRICTION = 5.0

GRAVITY_ON = True
GRAVITY = 1400.0

N_BALLS = 8
TRAIL_FADE_ALPHA = 55
DENSITY = 1.0                 # mass ~ r^2

# Particles (Day 7)
PARTICLE_GRAVITY = 900.0
PARTICLE_FADE = 0.92          # per-frame alpha decay multiplier (0.9..0.98)
PARTICLE_SPEED = (120, 460)   # min/max initial speed
PARTICLE_LIFE = (0.25, 0.60)  # seconds
SHAKE_DECAY = 7.0             # camera shake damping (bigger = faster settle)
SHAKE_SCALE = 0.003           # how much impulse feeds shake

# ---------------------------
# Small helpers
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

def clamp(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)

rng = np.random.default_rng()

# ---------------------------
# Particles (Day 7)
# ---------------------------
class Particle:
    __slots__ = ("x", "y", "vx", "vy", "life", "max_life", "col", "alpha")
    def __init__(self, x, y, vx, vy, life, col):
        self.x = float(x); self.y = float(y)
        self.vx = float(vx); self.vy = float(vy)
        self.life = float(life); self.max_life = float(life)
        self.col = col
        self.alpha = 255

    def update(self, dt):
        # simple gravity + integrate
        self.vy += PARTICLE_GRAVITY * dt
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.life -= dt
        # fade alpha multiplicatively (soft tail)
        self.alpha = int(self.alpha * (PARTICLE_FADE ** (dt * FPS)))
        return self.life > 0 and self.alpha > 4

    def draw(self, surf):
        # draw as small filled circle; fade via per-particle alpha
        a = clamp(self.alpha, 0, 255)
        color = (min(255, self.col[0] + 60), min(255, self.col[1] + 60), min(255, self.col[2] + 60), a)
        pygame.draw.circle(surf, color, (int(self.x), int(self.y)), 2)

particles = []
shake_mag = 0.0
shake_angle = 0.0

def add_particles_at(pos, color, strength):
    """Spawn a burst of sparks at pos. strength ~ impulse magnitude."""
    global particles, shake_mag, shake_angle
    # map strength to particle count
    count = int(clamp(2 + strength * 0.025, 6, 40))
    x, y = float(pos[0]), float(pos[1])
    # camera shake proportional to strength
    shake_mag += strength * SHAKE_SCALE
    shake_angle = rng.uniform(0, 2*math.pi)

    for _ in range(count):
        ang = rng.uniform(0, 2*math.pi)
        speed = rng.uniform(*PARTICLE_SPEED)
        vx = math.cos(ang) * speed
        vy = math.sin(ang) * speed
        life = rng.uniform(*PARTICLE_LIFE)
        particles.append(Particle(x, y, vx, vy, life, color))

def update_particles(dt, trail_surface):
    # update and draw particles to the trail layer (so they get nice motion blur)
    alive = []
    for p in particles:
        if p.update(dt):
            p.draw(trail_surface)
            alive.append(p)
    particles[:] = alive

def camera_shake_offset(dt):
    """Return a small (ox, oy) to jitter the whole frame on big impacts."""
    global shake_mag, shake_angle
    if shake_mag <= 1e-4:
        shake_mag = 0.0
        return 0, 0
    # decay
    shake_mag = max(0.0, shake_mag - SHAKE_DECAY * dt * shake_mag)
    # move angle slightly so it swirls a little
    shake_angle += 22.0 * dt
    ox = int(math.cos(shake_angle) * 6.0 * shake_mag)
    oy = int(math.sin(shake_angle * 1.3) * 6.0 * shake_mag)
    return ox, oy

# ---------------------------
# Balls & geometry (from Day 6)
# ---------------------------
def make_balls(n):
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
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    q = a + t * ab
    return q, t

def collide_ball_with_segment(pos_i, vel_i, radius_i, seg, color=None):
    """
    Circle vs static segment:
    - Push out along normal
    - Reflect/dampen normal velocity, damp tangent
    - Spawn particles & shake using impact strength
    """
    q, _ = closest_point_on_segment(seg.a, seg.b, pos_i)
    d = pos_i - q
    d2 = float(d[0]*d[0] + d[1]*d[1])
    r = float(radius_i)
    if d2 >= r*r:
        return False

    dist = math.sqrt(d2) if d2 > 1e-12 else 1e-6
    n_hat = d / dist
    overlap = r - dist

    # Positional correction
    pos_i += n_hat * overlap

    # Velocity split
    vn = float(vel_i[0]*n_hat[0] + vel_i[1]*n_hat[1])
    vt = vel_i - vn * n_hat

    # Normal bounce + tangential friction
    new_vn = -seg.e * vn
    new_vt = vt * max(0.0, 1.0 - seg.fric)

    # impact "strength" proxy: change in normal speed
    impact_strength = abs(new_vn - vn) * 0.5 + np.linalg.norm(vt - new_vt) * 0.25

    vel_i[:] = new_vt + new_vn * n_hat

    # Day 7: particles + shake
    if color is None: color = (200, 220, 255)
    add_particles_at(pos_i, color, impact_strength)
    return True

def build_level(level_id):
    segs = []
    if level_id == 1:
        name = "Ramp + Platform"
        segs.append(Segment(80, HEIGHT-120, 360, HEIGHT-40, restitution=0.80, friction=0.08))
        segs.append(Segment(WIDTH-80, HEIGHT-120, WIDTH-360, HEIGHT-40, restitution=0.80, friction=0.08))
        segs.append(Segment(450, 340, 820, 340, restitution=0.75, friction=0.05))
    else:
        name = "Funnel + Ledge"
        segs.append(Segment(40, 120, 380, 320, restitution=0.80, friction=0.06))
        segs.append(Segment(WIDTH-40, 120, WIDTH-380, 320, restitution=0.80, friction=0.06))
        segs.append(Segment(420, 430, 560, 430, restitution=0.75, friction=0.05))
    return segs, name

def draw_segments(surface, segs):
    for s in segs:
        pygame.draw.line(surface, GEOM_COLOR, (int(s.a[0]), int(s.a[1])), (int(s.b[0]), int(s.b[1])), 4)

def resolve_ball_ball_collisions_mass(pos, vel, radii, masses, inv_masses, e, flash=None, colors=None):
    """Mass-aware collisions + Day 7: impact particles/shake on strong hits."""
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

                    # Day 7: particles at contact point (midpoint)
                    contact = (pos[i] + pos[j]) * 0.5
                    # use impulse magnitude as strength
                    impact_strength = abs(j_imp)
                    col_i = colors[i] if colors else (220, 220, 255)
                    col_j = colors[j] if colors else (220, 220, 255)
                    add_particles_at(contact, col_i, impact_strength * 0.7)
                    add_particles_at(contact, col_j, impact_strength * 0.7)

def add_ball_at_mouse(pos, vel, colors, radii, masses, inv_masses, flash):
    mx, my = pygame.mouse.get_pos()
    pos = np.vstack([pos, np.array([mx, my], dtype=float)])
    vel = np.vstack([vel, np.array([0.0, 0.0], dtype=float)])
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
    pygame.display.set_caption("Day 7: Impact particles + camera shake")
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

    fps_ema = 0.0
    ema_alpha = 0.12

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

            # collide with segments (static geometry) + particles on impact
            for i in range(len(radii)):
                for seg in segments:
                    if collide_ball_with_segment(pos[i], vel[i], radii[i], seg, color=colors[i]):
                        flash[i] = 0.12

            # mass-aware ball-ball collisions + particles
            resolve_ball_ball_collisions_mass(pos, vel, radii, masses, inv_masses, REST_COEFF, flash, colors)

            # decay flash timers
            flash = np.maximum(0.0, flash - dt)

        # ---- draw
        # camera shake offset (Day 7)
        ox, oy = camera_shake_offset(dt)

        # 1) background + floor
        screen.fill(BG_COLOR)
        pygame.draw.rect(screen, FLOOR_COLOR, pygame.Rect(0 + ox, HEIGHT - 6 + oy, WIDTH, 6))

        # 2) trails layer
        # fade old trails
        trail.blit(fade_rect, (0, 0))
        # draw balls (with impact flash)
        for i, c in enumerate(colors):
            draw_c = c
            if flash[i] > 0:
                amt = int(120 * (flash[i] / 0.12))
                draw_c = lighten(c, amt)
            pygame.draw.circle(trail, draw_c, (int(pos[i, 0] + ox), int(pos[i, 1] + oy)), int(radii[i]))
        # draw + update particles on the trail layer for nice blur
        update_particles(dt, trail)
        # composite trail
        screen.blit(trail, (0, 0))

        # 3) draw geometry on top
        if show_geom:
            # redraw segments with same offset so they "shake" with the frame
            for s in segments:
                pygame.draw.line(
                    screen, GEOM_COLOR,
                    (int(s.a[0] + ox), int(s.a[1] + oy)),
                    (int(s.b[0] + ox), int(s.b[1] + oy)),
                    4
                )

        # 4) HUD
        dt_ms_safe = max(1, dt_ms)
        inst_fps = 1000.0 / dt_ms_safe
        fps_ema = (1 - ema_alpha) * fps_ema + ema_alpha * inst_fps

        momentum = (masses[:, None] * vel).sum(axis=0)
        px, py = float(momentum[0]), float(momentum[1])

        draw_text(
            screen,
            f"lvl={level_name}  geom={'ON' if show_geom else 'OFF'}  balls={len(colors)}  g={'ON' if gravity_on else 'OFF'}  e={REST_COEFF:.2f}  drag={AIR_DRAG:.2f}  FPS~{fps_ema:5.1f}",
            10, 10, font
        )
        draw_text(screen, f"Σp=({px:8.1f}, {py:8.1f})", 10, 32, font)
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
