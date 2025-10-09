import sys
import math
import pygame
import numpy as np

# =========================================================
# Cloth Simulation (mass–spring grid with pins)
# - Robust mouse handling + multiple pin toggles
# - Safe drawing (prevents invalid end_pos errors)
# - Particles, camera shake, level geometry, balls
# =========================================================

# ---------------------------
# Config
# ---------------------------
WIDTH, HEIGHT = 900, 600
FPS = 120

BG_COLOR = (18, 18, 24)
FLOOR_COLOR = (35, 40, 55)
TEXT_COLOR = (220, 220, 220)
GEOM_COLOR = (0, 255, 180)     # level geometry
LINK_COLOR = (180, 210, 255)   # cloth springs
NODE_COLOR = (240, 240, 255)

REST_COEFF = 0.80
AIR_DRAG = 0.25                # mild global damping
GROUND_FRICTION = 5.0
GRAVITY_ON = True
GRAVITY = 900.0

# Trails (off by default for cloth)
TRAIL_FADE_ALPHA = 0           # set ~80 for motion blur
DENSITY = 1.0

# Particles
PARTICLE_GRAVITY = 900.0
PARTICLE_FADE = 0.92
PARTICLE_SPEED = (120, 460)
PARTICLE_LIFE = (0.25, 0.60)
SHAKE_DECAY = 7.0
SHAKE_SCALE = 0.003

# Cloth grid
ROWS = 18
COLS = 26
SPACING = 22.0                 # rest distance between neighbors
NODE_R = 3                     # for collisions & drawing
STRUCT_K = 420.0               # structural spring stiffness
STRUCT_D = 20.0                # damping along spring
SHEAR_K  = 320.0               # diagonal springs
SHEAR_D  = 18.0
BEND_K   = 180.0               # bend (skip-1) springs
BEND_D   = 14.0
SUBSTEPS = 2                   # spring solver substeps

# Pinned (top corners)
PIN_TOP_LEFT  = True
PIN_TOP_RIGHT = True

# Balls to interact with cloth
BALL_R = 22
BALL_COLOR = (200, 230, 255)

# Heatmap for cloth speed
HEATMAP_MIN = 0.0
HEATMAP_EPS = 1e-6

# Mouse drag spring on cloth nodes
PICK_RADIUS = 48.0             # a bit larger to make picking easier
DRAG_K = 750.0
DRAG_D = 18.0

# Optional safety caps
VEL_CLAMP = 3000.0             # max cloth node speed (prevents runaway)
POS_CLAMP = 1e6                # clamp positions to huge but finite range

rng = np.random.default_rng()

# ---------------------------
# Helpers
# ---------------------------
def draw_text(surface, text, x, y, font):
    surface.blit(font.render(text, True, TEXT_COLOR), (x, y))

def clamp(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)

def lerp(a, b, t):
    return a + (b - a) * t

def heat_color_from_speed(s, s_min, s_max):
    if s_max <= s_min + HEATMAP_EPS:
        t = 0.0
    else:
        t = (s - s_min) / max(HEATMAP_EPS, (s_max - s_min))
        t = clamp(t, 0.0, 1.0)
    # blue -> cyan -> yellow -> orange
    if t < 0.5:
        u = t / 0.5
        r = int(lerp(0, 255, u))
        g = int(lerp(120, 255, u))
        b = int(lerp(255, 0, u))
    else:
        u = (t - 0.5) / 0.5
        r = 255; g = int(lerp(255, 80, u)); b = 0
    return (r, g, b)

def finite2(v):
    """Quick finite check for a 2D vector-like."""
    return (np.isfinite(v[0]) and np.isfinite(v[1]))

# ---------------------------
# Particles + camera shake
# ---------------------------
class Particle:
    __slots__ = ("x","y","vx","vy","life","max_life","col","alpha")
    def __init__(self, x, y, vx, vy, life, col):
        self.x = float(x); self.y = float(y)
        self.vx = float(vx); self.vy = float(vy)
        self.life = float(life); self.max_life = float(life)
        self.col = col; self.alpha = 255

    def update(self, dt):
        self.vy += PARTICLE_GRAVITY * dt
        self.x += self.vx * dt; self.y += self.vy * dt
        self.life -= dt
        self.alpha = int(self.alpha * (PARTICLE_FADE ** (dt * FPS)))
        return self.life > 0 and self.alpha > 4

    def draw(self, surf):
        a = clamp(self.alpha, 0, 255)
        color = (min(255, self.col[0] + 60),
                 min(255, self.col[1] + 60),
                 min(255, self.col[2] + 60), a)
        pygame.draw.circle(surf, color, (int(self.x), int(self.y)), 2)

particles = []
shake_mag = 0.0
shake_angle = 0.0

def add_particles_at(pos, color, strength):
    global particles, shake_mag, shake_angle
    count = int(clamp(2 + strength * 0.025, 6, 40))
    x, y = float(pos[0]), float(pos[1])
    shake_mag += strength * SHAKE_SCALE
    shake_angle = rng.uniform(0, 2*math.pi)
    for _ in range(count):
        ang = rng.uniform(0, 2*math.pi)
        speed = rng.uniform(*PARTICLE_SPEED)
        vx = math.cos(ang) * speed; vy = math.sin(ang) * speed
        life = rng.uniform(*PARTICLE_LIFE)
        particles.append(Particle(x, y, vx, vy, life, color))

def update_particles(dt, trail_surface):
    alive = []
    for p in particles:
        if p.update(dt):
            p.draw(trail_surface)
            alive.append(p)
    particles[:] = alive

def camera_shake_offset(dt):
    global shake_mag, shake_angle
    if shake_mag <= 1e-4:
        shake_mag = 0.0
        return 0, 0
    shake_mag = max(0.0, shake_mag - SHAKE_DECAY * dt * shake_mag)
    shake_angle += 22.0 * dt
    ox = int(math.cos(shake_angle) * 6.0 * shake_mag)
    oy = int(math.sin(shake_angle * 1.3) * 6.0 * shake_mag)
    return ox, oy

# ---------------------------
# Geometry
# ---------------------------
class Segment:
    __slots__ = ("a","b","e","fric")
    def __init__(self, ax, ay, bx, by, restitution=0.80, friction=0.05):
        self.a = np.array([ax, ay], dtype=float)
        self.b = np.array([bx, by], dtype=float)
        self.e = float(restitution)
        self.fric = float(friction)

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

def closest_point_on_segment(a, b, p):
    ab = b - a
    ab2 = float(ab[0]*ab[0] + ab[1]*ab[1])
    if ab2 <= 1e-12:
        return a.copy(), 0.0
    t = float((p[0]-a[0])*ab[0] + (p[1]-a[1])*ab[1]) / ab2
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    q = a + t * ab
    return q, t

def collide_point_with_segment(pos_i, vel_i, radius_i, seg):
    q, _ = closest_point_on_segment(seg.a, seg.b, pos_i)
    d = pos_i - q
    d2 = float(d[0]*d[0] + d[1]*d[1])
    r = float(radius_i)
    if d2 >= r*r:
        return False
    dist = math.sqrt(d2) if d2 > 1e-12 else 1e-6
    n_hat = d / dist
    overlap = r - dist
    pos_i[:] = pos_i + n_hat * overlap

    vn = float(vel_i[0]*n_hat[0] + vel_i[1]*n_hat[1])
    vt = vel_i - vn * n_hat
    new_vn = -seg.e * vn
    new_vt = vt * max(0.0, 1.0 - seg.fric)
    vel_i[:] = new_vt + new_vn * n_hat

    # particles/shake
    strength = abs(new_vn - vn) * 0.5
    add_particles_at(pos_i, (200, 220, 255), strength)
    return True

def draw_segments(surface, segs, ox=0, oy=0):
    for s in segs:
        pygame.draw.line(surface, GEOM_COLOR,
                         (int(s.a[0] + ox), int(s.a[1] + oy)),
                         (int(s.b[0] + ox), int(s.b[1] + oy)), 4)

# ---------------------------
# Balls (for interaction)
# ---------------------------
def make_ball(x, y, r=BALL_R, vx=0.0, vy=0.0):
    return np.array([x, y], dtype=float), np.array([vx, vy], dtype=float), int(r)

def integrate_ball(pos, vel, r, dt):
    if GRAVITY_ON:
        vel[1] += GRAVITY * dt
    if AIR_DRAG > 0.0:
        vel[:] = vel - vel * AIR_DRAG * dt
    pos[:] = pos + vel * dt
    # walls
    if pos[0] - r < 0:      pos[0] = r;            vel[0] = -vel[0] * REST_COEFF
    if pos[0] + r > WIDTH:  pos[0] = WIDTH - r;    vel[0] = -vel[0] * REST_COEFF
    if pos[1] - r < 0:      pos[1] = r;            vel[1] = -vel[1] * REST_COEFF
    if pos[1] + r > HEIGHT:
        pos[1] = HEIGHT - r
        if vel[1] > 0: vel[1] = -vel[1] * REST_COEFF

def collide_ball_with_segments(pos, vel, r, segments):
    for seg in segments:
        collide_point_with_segment(pos, vel, r, seg)

def cloth_point_vs_ball(p, vp, node_r, ball_pos, ball_r, ball_vel=None):
    # push cloth node out of ball if inside
    delta = p - ball_pos
    d2 = float(delta[0]*delta[0] + delta[1]*delta[1])
    min_dist = float(node_r + ball_r)
    if d2 < min_dist * min_dist:
        d = math.sqrt(d2) if d2 > 1e-12 else 1e-6
        n_hat = delta / d
        overlap = min_dist - d
        p[:] = p + n_hat * overlap
        # reflect node velocity along normal (soft)
        vn = float(vp[0]*n_hat[0] + vp[1]*n_hat[1])
        vp[:] = vp - (1.0 + 0.6) * vn * n_hat  # 0.6 ~ restitution vs ball
        # sparks
        add_particles_at(p, (255, 230, 120), abs(vn) * 0.6)
        return True
    return False

# ---------------------------
# Cloth build & solver
# ---------------------------
def make_cloth(rows, cols, spacing, origin=(120, 80)):
    x0, y0 = origin
    count = rows * cols
    pos = np.zeros((count, 2), dtype=float)
    vel = np.zeros((count, 2), dtype=float)
    inv_mass = np.ones(count, dtype=float) / (NODE_R * NODE_R)  # ~ mass ~ r^2
    pinned = np.zeros(count, dtype=bool)

    # grid positions
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            pos[idx, 0] = x0 + c * spacing
            pos[idx, 1] = y0 + r * spacing
            vel[idx] = 0.0

    # pin top corners
    if PIN_TOP_LEFT:
        pinned[0] = True
    if PIN_TOP_RIGHT:
        pinned[cols - 1] = True
    inv_mass[pinned] = 0.0

    # springs: structural (right & down)
    links = []
    for r in range(rows):
        for c in range(cols):
            i = r * cols + c
            if c + 1 < cols:  # right
                j = r * cols + (c + 1)
                rest = spacing
                links.append((i, j, rest, STRUCT_K, STRUCT_D))
            if r + 1 < rows:  # down
                j = (r + 1) * cols + c
                rest = spacing
                links.append((i, j, rest, STRUCT_K, STRUCT_D))
    # shear (diagonals)
    for r in range(rows - 1):
        for c in range(cols - 1):
            i = r * cols + c
            j = (r + 1) * cols + (c + 1)
            links.append((i, j, spacing * math.sqrt(2.0), SHEAR_K, SHEAR_D))
            i2 = r * cols + (c + 1)
            j2 = (r + 1) * cols + c
            links.append((i2, j2, spacing * math.sqrt(2.0), SHEAR_K, SHEAR_D))
    # bend (skip one)
    for r in range(rows):
        for c in range(cols):
            i = r * cols + c
            if c + 2 < cols:
                j = r * cols + (c + 2)
                links.append((i, j, spacing * 2.0, BEND_K, BEND_D))
            if r + 2 < rows:
                j = (r + 2) * cols + c
                links.append((i, j, spacing * 2.0, BEND_K, BEND_D))

    return pos, vel, inv_mass, pinned, links

def apply_links(pos, vel, inv_mass, links, dt):
    for (i, j, L0, k, d) in links:
        delta = pos[j] - pos[i]
        d2 = float(delta[0]*delta[0] + delta[1]*delta[1])
        if d2 <= 1e-12:
            continue
        dist = math.sqrt(d2)
        n_hat = delta / dist

        # Hooke force along link
        stretch = dist - L0
        F_spring = k * stretch

        # damping along the link direction
        relv = vel[j] - vel[i]
        v_rel_n = float(relv[0]*n_hat[0] + relv[1]*n_hat[1])
        F_damp = d * v_rel_n

        F = F_spring + F_damp

        inv_mi = inv_mass[i]
        inv_mj = inv_mass[j]
        if inv_mi > 0.0:
            vel[i] += (-F * inv_mi) * n_hat * dt
        if inv_mj > 0.0:
            vel[j] += ( F * inv_mj) * n_hat * dt

def draw_cloth_lines(screen, pos, rows, cols, color, ox=0, oy=0, w=2):
    """Ultra-safe renderer: validates and clamps each endpoint; skips draw on any oddity."""
    def safe_pt(idx):
        p = pos[idx]
        if not (hasattr(p, "__len__") and len(p) == 2):
            return None
        try:
            x, y = float(p[0]), float(p[1])
        except Exception:
            return None
        if not (np.isfinite(x) and np.isfinite(y)):
            return None
        x = clamp(x, -POS_CLAMP, POS_CLAMP)
        y = clamp(y, -POS_CLAMP, POS_CLAMP)
        return (int(x + ox), int(y + oy))

    # horizontal
    for r in range(rows):
        base = r * cols
        for c in range(cols - 1):
            i = base + c
            j = i + 1
            a = safe_pt(i); b = safe_pt(j)
            if a is None or b is None:
                continue
            try:
                pygame.draw.line(screen, color, a, b, w)
            except Exception:
                continue

    # vertical
    for c in range(cols):
        for r in range(rows - 1):
            i = r * cols + c
            j = (r + 1) * cols + c
            a = safe_pt(i); b = safe_pt(j)
            if a is None or b is None:
                continue
            try:
                pygame.draw.line(screen, color, a, b, w)
            except Exception:
                continue

# ---------------------------
# Main
# ---------------------------
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Cloth Simulation – robust input + pins")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 18)

    # Trail layer
    trail = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    fade_rect = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    fade_rect.fill((0, 0, 0, TRAIL_FADE_ALPHA))
    use_trails = TRAIL_FADE_ALPHA > 0

    # Level
    level_id = 1
    segments, level_name = build_level(level_id)
    show_geom = True

    # Cloth
    posC, velC, invC, pinned, links = make_cloth(ROWS, COLS, SPACING, origin=(140, 60))

    # Balls to interact (start with one)
    ball_pos, ball_vel, ball_r = make_ball(WIDTH*0.72, 140, r=BALL_R, vx=-120, vy=0)
    balls = [(ball_pos, ball_vel, ball_r)]

    paused = False
    heatmap_on = False
    show_debug = False
    show_nodes = False

    dragging = False
    drag_idx = -1

    fps_ema = 0.0
    ema_alpha = 0.12

    running = True
    while running:
        dt_ms = clock.tick(FPS)
        dt = dt_ms / 1000.0

        # Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    posC, velC, invC, pinned, links = make_cloth(ROWS, COLS, SPACING, origin=(140, 60))
                    # reset one ball
                    balls = []
                    bpos, bvel, br = make_ball(WIDTH*0.72, 140, r=BALL_R, vx=-120, vy=0)
                    balls.append((bpos, bvel, br))
                elif event.key == pygame.K_g:
                    global GRAVITY_ON
                    GRAVITY_ON = not GRAVITY_ON
                elif event.key == pygame.K_l:
                    level_id = 2 if level_id == 1 else 1
                    segments, level_name = build_level(level_id)
                elif event.key == pygame.K_h:
                    show_geom = not show_geom
                elif event.key == pygame.K_v:
                    heatmap_on = not heatmap_on
                elif event.key == pygame.K_d:
                    show_debug = not show_debug
                elif event.key == pygame.K_n:
                    show_nodes = not show_nodes
                elif event.key == pygame.K_t:
                    use_trails = not use_trails
                elif event.key == pygame.K_p:
                    # KEYBOARD PIN TOGGLE nearest to current mouse
                    mx, my = pygame.mouse.get_pos()
                    dx = posC[:,0] - mx
                    dy = posC[:,1] - my
                    d2 = dx*dx + dy*dy
                    i = int(np.argmin(d2))
                    pinned[i] = not pinned[i]
                    invC[i] = 0.0 if pinned[i] else (1.0 / (NODE_R * NODE_R))

            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()
                mods = pygame.key.get_mods()
                # find nearest node
                dx = posC[:,0] - mx
                dy = posC[:,1] - my
                d2 = dx*dx + dy*dy
                i = int(np.argmin(d2))
                dist = math.sqrt(float(d2[i]))

                if event.button == 1:  # LMB
                    if (mods & pygame.KMOD_SHIFT) or (mods & pygame.KMOD_CTRL):
                        # Shift+LMB or Ctrl+LMB -> PIN TOGGLE
                        if dist <= PICK_RADIUS:
                            pinned[i] = not pinned[i]
                            invC[i] = 0.0 if pinned[i] else (1.0 / (NODE_R * NODE_R))
                    else:
                        # Regular LMB -> start dragging if close enough
                        if dist <= PICK_RADIUS:
                            dragging = True
                            drag_idx = i

                elif event.button == 2:
                    # MMB (wheel) -> pin toggle
                    if dist <= PICK_RADIUS:
                        pinned[i] = not pinned[i]
                        invC[i] = 0.0 if pinned[i] else (1.0 / (NODE_R * NODE_R))

                elif event.button == 3:
                    # RMB -> spawn ball
                    bpos, bvel, br = make_ball(mx, my, r=BALL_R, vx=rng.uniform(-160,160), vy=-60)
                    balls.append((bpos, bvel, br))

            if event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1 and dragging:
                    dragging = False
                    drag_idx = -1

        # Physics
        if not paused:
            # external forces
            if GRAVITY_ON:
                velC[:,1] += GRAVITY * dt
            if AIR_DRAG > 0.0:
                velC -= velC * AIR_DRAG * dt

            # mouse drag (spring)
            if dragging and 0 <= drag_idx < posC.shape[0]:
                mx, my = pygame.mouse.get_pos()
                dx = mx - posC[drag_idx,0]
                dy = my - posC[drag_idx,1]
                # F = kx - c v
                Fx = DRAG_K * dx - DRAG_D * velC[drag_idx,0]
                Fy = DRAG_K * dy - DRAG_D * velC[drag_idx,1]
                if invC[drag_idx] > 0.0:
                    velC[drag_idx,0] += (Fx * invC[drag_idx]) * dt
                    velC[drag_idx,1] += (Fy * invC[drag_idx]) * dt

            # spring solver substeps
            sub_dt = dt / SUBSTEPS
            for _ in range(SUBSTEPS):
                apply_links(posC, velC, invC, links, sub_dt)

            # integrate cloth nodes
            posC += velC * dt

            # --- SANITIZE positions & velocities to avoid invalid draw coords ---
            posC = np.nan_to_num(posC, nan=0.0, posinf=0.0, neginf=0.0)
            np.clip(posC, -POS_CLAMP, POS_CLAMP, out=posC)
            # optional vel clamp to prevent explosions:
            if VEL_CLAMP is not None:
                spd = np.linalg.norm(velC, axis=1)
                mask = spd > VEL_CLAMP
                if np.any(mask):
                    velC[mask] *= (VEL_CLAMP / spd[mask])[:, None]

            # collisions: nodes vs level geometry
            for i in range(posC.shape[0]):
                for seg in segments:
                    collide_point_with_segment(posC[i], velC[i], NODE_R, seg)

            # collisions: nodes vs balls
            for (bpos, bvel, br) in balls:
                for i in range(posC.shape[0]):
                    cloth_point_vs_ball(posC[i], velC[i], NODE_R, bpos, br, bvel)

            # pinned nodes stay put (zero vel)
            velC[pinned] = 0.0

            # balls integrate and collide with level
            for idx in range(len(balls)):
                bpos, bvel, br = balls[idx]
                integrate_ball(bpos, bvel, br, dt)
                collide_ball_with_segments(bpos, bvel, br, segments)

        # Draw
        ox, oy = camera_shake_offset(dt)

        screen.fill(BG_COLOR)
        pygame.draw.rect(screen, FLOOR_COLOR, pygame.Rect(0 + ox, HEIGHT - 6 + oy, WIDTH, 6))

        # Trails layer
        if use_trails and TRAIL_FADE_ALPHA > 0:
            trail.blit(fade_rect, (0, 0))
        else:
            trail.fill((0,0,0,0))

        # Cloth heatmap by node speed (affects mesh color)
        speeds = np.linalg.norm(velC, axis=1) if len(velC) else np.array([0.0])
        s_max = max(HEATMAP_MIN + HEATMAP_EPS, float(np.percentile(speeds, 95))) if speeds.size else 1.0
        mesh_color = LINK_COLOR
        if heatmap_on:
            mesh_color = heat_color_from_speed(float(np.percentile(speeds, 75)), HEATMAP_MIN, s_max)

        # draw cloth mesh safely
        draw_cloth_lines(screen, posC + np.array([ox, oy]), ROWS, COLS, mesh_color, 0, 0, w=2)

        # optional nodes
        if show_nodes:
            for i in range(posC.shape[0]):
                if finite2(posC[i]):
                    c = NODE_COLOR if not heatmap_on else heat_color_from_speed(float(speeds[i]), HEATMAP_MIN, s_max)
                    pygame.draw.circle(screen, c, (int(posC[i,0]+ox), int(posC[i,1]+oy)), NODE_R)

        # draw balls
        for (bpos, _, br) in balls:
            pygame.draw.circle(screen, (255,255,255), (int(bpos[0]+ox), int(bpos[1]+oy)), br+1, 2)
            pygame.draw.circle(screen, BALL_COLOR, (int(bpos[0]+ox), int(bpos[1]+oy)), br)

        # geom
        if show_geom:
            draw_segments(screen, segments, ox, oy)

        # particles
        update_particles(dt, trail)
        screen.blit(trail, (0, 0))

        # HUD
        dt_ms_safe = max(1, dt_ms)
        inst_fps = 1000.0 / dt_ms_safe
        fps_ema = (1 - ema_alpha) * fps_ema + ema_alpha * inst_fps

        # Mouse overlay (to verify buttons are detected)
        mx, my = pygame.mouse.get_pos()
        btns = pygame.mouse.get_pressed(num_buttons=5)
        lmb = '1' if btns[0] else '0'
        mmb = '1' if (len(btns) > 1 and btns[1]) else '0'
        rmb = '1' if (len(btns) > 2 and btns[2]) else '0'

        draw_text(screen,
                  f"lvl={level_name}  heat={'ON' if heatmap_on else 'OFF'}  nodes={'ON' if show_nodes else 'OFF'}  balls={len(balls)}  g={'ON' if GRAVITY_ON else 'OFF'}  FPS~{fps_ema:5.1f}",
                  10, 10, font)
        draw_text(screen,
                  "Space=pause  R=reset  G=toggle g  L=level  H=geom  V=heatmap  D=debug  N=nodes  T=trails",
                  10, 32, font)
        draw_text(screen,
                  "LMB=drag  Shift+LMB=pin  Ctrl+LMB=pin  MMB=pin  RMB=spawn ball  P=pin nearest (keyboard)",
                  10, 54, font)
        draw_text(screen,
                  f"mouse=({mx:4d},{my:4d}) btns=[L:{lmb} M:{mmb} R:{rmb}]",
                  10, 76, font)

        if show_debug:
            draw_text(screen, f"[DEBUG] rows={ROWS} cols={COLS} links={len(links)} s_max(p95)={s_max:6.1f}", 10, 98, font)

        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
