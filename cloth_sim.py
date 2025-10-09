import sys, math
import pygame
import numpy as np

# ---------------------------
# Window / Colors
# ---------------------------
WIDTH, HEIGHT = 900, 600
FPS = 120
BG_COLOR = (18, 18, 24)
FLOOR_COLOR = (35, 40, 55)
TEXT_COLOR = (220, 220, 220)
GEOM_COLOR = (0, 255, 180)

# ---------------------------
# Physics
# ---------------------------
GRAVITY = np.array([0.0, 1400.0], dtype=float)

# Wind (base + turbulence)
WIND_ON = True
WIND_BASE = 180.0                  # +X accel (px/s^2)
WIND_TURB_ON = True
WIND_TURB_AMP = 140.0              # px/s^2
WIND_TURB_FREQ = 1.2               # Hz
WIND_STEP = 30.0                   # per key press

# Cloth (grid)
ROWS, COLS = 16, 26
NODE_SPACING = 20.0
DAMPING = 0.995
SOLVER_ITERS = 4

# Tearing
TEAR_MODE = True
TEAR_RATIO = 1.8                   # break when length > rest * ratio
MOUSE_TEAR = True
MOUSE_TEAR_RADIUS = 28.0

# Ball (interactive)
BALL_RADIUS = 18
BALL_COLOR = (220, 200, 255)

# Toggles
SHOW_GEOM = True
DEV_OVERLAY = True
RENDER_SPRINGS_ONLY = True         # B toggles: springs view vs grid lines

rng = np.random.default_rng()

# ---------------------------
# Geometry for levels
# ---------------------------
class Segment:
    __slots__ = ("a", "b", "e", "fric")
    def __init__(self, ax, ay, bx, by, restitution=0.80, friction=0.06):
        self.a = np.array([ax, ay], dtype=float)
        self.b = np.array([bx, by], dtype=float)
        self.e = float(restitution)
        self.fric = float(friction)

def build_level(level_id: int):
    segs = []
    if level_id == 1:
        name = "Ramp + Platform"
        segs.append(Segment(80, HEIGHT-120, 360, HEIGHT-40))
        segs.append(Segment(WIDTH-80, HEIGHT-120, WIDTH-360, HEIGHT-40))
        segs.append(Segment(450, 340, 820, 340, restitution=0.75, friction=0.05))
    else:
        name = "Funnel + Ledge"
        segs.append(Segment(40, 120, 380, 320))
        segs.append(Segment(WIDTH-40, 120, WIDTH-380, 320))
        segs.append(Segment(420, 430, 560, 430, restitution=0.75, friction=0.05))
    return segs, name

def draw_segments(surface, segs):
    for s in segs:
        pygame.draw.line(surface, GEOM_COLOR,
                         (int(s.a[0]), int(s.a[1])),
                         (int(s.b[0]), int(s.b[1])), 3)

# ---------------------------
# Cloth helpers
# ---------------------------
def grid_index(r, c, cols):
    return r*cols + c

def make_grid_positions(rows, cols, start=(120, 60), dx=NODE_SPACING, dy=NODE_SPACING):
    x0, y0 = start
    P = np.zeros((rows*cols, 2), dtype=float)
    for r in range(rows):
        for c in range(cols):
            i = grid_index(r, c, cols)
            P[i, 0] = x0 + c*dx
            P[i, 1] = y0 + r*dy
    return P

def build_springs(rows, cols, spacing):
    """
    Structural springs: right and down neighbors.
    Returns list of [i, j, rest, enabled_bool]
    """
    springs = []
    for r in range(rows):
        for c in range(cols):
            i = grid_index(r, c, cols)
            if c+1 < cols:
                j = grid_index(r, c+1, cols)
                springs.append([i, j, spacing, True])
            if r+1 < rows:
                j = grid_index(r+1, c, cols)
                springs.append([i, j, spacing, True])
    return springs

def build_anchors(rows, cols):
    """
    Anchor the entire top row for a classic hanging cloth.
    """
    anchors = np.zeros(rows*cols, dtype=bool)
    anchors[:cols] = True
    return anchors

def apply_wind(acc, phases, t):
    if WIND_ON:
        acc[:, 0] += WIND_BASE
    if WIND_TURB_ON and WIND_TURB_AMP > 1e-6:
        acc[:, 0] += WIND_TURB_AMP * np.sin(2*math.pi*WIND_TURB_FREQ * t + phases)

def satisfy_spring_constraints(P, springs, anchors, ratio_break=TEAR_RATIO):
    """
    XPBD-style positional correction for structural springs.
    Also performs tearing if enabled.
    """
    for s in springs:
        if not s[3]:        # disabled/broken
            continue
        i, j, rest = s[0], s[1], s[2]
        d = P[j] - P[i]
        d2 = float(d[0]*d[0] + d[1]*d[1])
        if d2 <= 1e-12:
            continue
        dist = math.sqrt(d2)
        if TEAR_MODE and dist > rest * ratio_break:
            s[3] = False
            continue
        diff = (dist - rest) / dist
        corr = 0.5 * diff * d
        # If an endpoint is anchored, push the free one more
        if anchors[i] and not anchors[j]:
            P[j] -= 2.0 * corr
        elif anchors[j] and not anchors[i]:
            P[i] += 2.0 * corr
        else:
            if not anchors[i]:
                P[i] += corr
            if not anchors[j]:
                P[j] -= corr

def collide_ball_with_nodes(P, V, ball_pos, ball_r, e=0.8):
    """
    Simple circle (ball) vs. cloth nodes collision.
    """
    r2 = float(ball_r)*float(ball_r)
    for k in range(P.shape[0]):
        d = P[k] - ball_pos
        d2 = float(d[0]*d[0] + d[1]*d[1])
        if d2 < r2:
            dist = math.sqrt(d2) if d2 > 1e-12 else 1e-6
            n_hat = d / dist
            overlap = ball_r - dist
            P[k] += n_hat * overlap
            vn = float(V[k,0]*n_hat[0] + V[k,1]*n_hat[1])
            V[k] -= (1.0 + e) * vn * n_hat

def mouse_drag_move_node(P, V, anchors, mouse_pos, pick_radius=36.0):
    mx, my = mouse_pos
    d2 = (P[:,0]-mx)**2 + (P[:,1]-my)**2
    i = int(np.argmin(d2))
    if not anchors[i] and d2[i] < pick_radius*pick_radius:
        P[i, 0] = mx
        P[i, 1] = my
        V[i, :] = 0.0

def mouse_tear_near(P, springs, mouse_pos, radius=MOUSE_TEAR_RADIUS):
    mx, my = mouse_pos
    r2 = radius*radius
    for s in springs:
        if not s[3]:
            continue
        i, j = s[0], s[1]
        a = P[i]; b = P[j]
        ab = b - a
        ab2 = float(ab[0]*ab[0] + ab[1]*ab[1])
        if ab2 <= 1e-12:
            continue
        t = ((mx-a[0])*ab[0] + (my-a[1])*ab[1]) / ab2
        t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
        qx = a[0] + t*ab[0]
        qy = a[1] + t*ab[1]
        dx = qx - mx
        dy = qy - my
        if dx*dx + dy*dy <= r2:
            s[3] = False

# ---------------------------
# Drawing
# ---------------------------
def draw_cloth_springs(surface, P, springs, color=(180, 200, 255), w=2):
    for s in springs:
        if not s[3]:
            continue
        i, j = s[0], s[1]
        x1, y1 = int(P[i,0]), int(P[i,1])
        x2, y2 = int(P[j,0]), int(P[j,1])
        pygame.draw.line(surface, color, (x1, y1), (x2, y2), w)

def draw_cloth_lines(surface, P, rows, cols, color=(200, 220, 255), w=2):
    for r in range(rows):
        for c in range(cols):
            i = grid_index(r, c, cols)
            x1, y1 = int(P[i,0]), int(P[i,1])
            if c+1 < cols:
                j = grid_index(r, c+1, cols)
                x2, y2 = int(P[j,0]), int(P[j,1])
                pygame.draw.line(surface, color, (x1, y1), (x2, y2), w)
            if r+1 < rows:
                j = grid_index(r+1, c, cols)
                x2, y2 = int(P[j,0]), int(P[j,1])
                pygame.draw.line(surface, color, (x1, y1), (x2, y2), w)

# ---------------------------
# Main
# ---------------------------
def main():
    global WIND_ON, WIND_BASE, WIND_TURB_ON, WIND_TURB_AMP, WIND_TURB_FREQ
    global TEAR_MODE, TEAR_RATIO, SHOW_GEOM, DEV_OVERLAY, RENDER_SPRINGS_ONLY

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Day 13: Cloth + Wind + Tearing + Ball + Levels")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 18)

    # Level
    level_id = 1
    segments, level_name = build_level(level_id)

    # Cloth state
    P = make_grid_positions(ROWS, COLS, start=(120, 60))
    V = np.zeros_like(P)
    springs = build_springs(ROWS, COLS, NODE_SPACING)
    anchors = build_anchors(ROWS, COLS)
    phases = rng.uniform(0, 2*math.pi, size=P.shape[0])

    # Ball
    ball_exists = False
    ball_pos = np.array([650.0, 220.0], dtype=float)
    ball_vel = np.array([0.0, 0.0], dtype=float)

    paused = False
    dragging = False
    tearing_drag = False

    t = 0.0  # simulation time (seconds)

    running = True
    while running:
        dt_ms = clock.tick(FPS)
        dt = max(1, dt_ms) / 1000.0
        t += dt

        # ------------- Events
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False

            elif e.type == pygame.KEYDOWN:
                # System / view
                if e.key == pygame.K_SPACE:
                    paused = not paused
                elif e.key == pygame.K_r:
                    P = make_grid_positions(ROWS, COLS, start=(120, 60))
                    V = np.zeros_like(P)
                    springs = build_springs(ROWS, COLS, NODE_SPACING)
                    anchors = build_anchors(ROWS, COLS)
                elif e.key == pygame.K_l:
                    level_id = 2 if level_id == 1 else 1
                    segments, level_name = build_level(level_id)
                elif e.key == pygame.K_h:
                    SHOW_GEOM = not SHOW_GEOM
                elif e.key == pygame.K_b:
                    RENDER_SPRINGS_ONLY = not RENDER_SPRINGS_ONLY
                elif e.key == pygame.K_d:
                    DEV_OVERLAY = not DEV_OVERLAY

                # Wind controls
                elif e.key == pygame.K_w:
                    WIND_ON = not WIND_ON
                elif e.key == pygame.K_a or e.key == pygame.K_LEFT:
                    WIND_BASE -= WIND_STEP
                elif e.key == pygame.K_RIGHT:
                    WIND_BASE += WIND_STEP
                elif e.key == pygame.K_s:
                    WIND_BASE = 0.0
                elif e.key == pygame.K_z:
                    WIND_TURB_AMP = max(0.0, WIND_TURB_AMP - 20.0)
                elif e.key == pygame.K_x:
                    WIND_TURB_AMP += 20.0
                elif e.key == pygame.K_c:
                    WIND_TURB_ON = not WIND_TURB_ON
                elif e.key == pygame.K_COMMA:
                    WIND_TURB_FREQ = max(0.05, WIND_TURB_FREQ - 0.1)
                elif e.key == pygame.K_PERIOD:
                    WIND_TURB_FREQ += 0.1

                # Tearing controls
                elif e.key == pygame.K_t:
                    TEAR_MODE = not TEAR_MODE
                elif e.key == pygame.K_LEFTBRACKET:   # [
                    TEAR_RATIO = max(1.05, TEAR_RATIO - 0.05)
                elif e.key == pygame.K_RIGHTBRACKET:  # ]
                    TEAR_RATIO = min(3.0, TEAR_RATIO + 0.05)
                elif e.key == pygame.K_y:
                    for s in springs:
                        s[3] = True

            elif e.type == pygame.MOUSEBUTTONDOWN:
                if e.button == 1:  # LMB
                    dragging = True
                    tearing_drag = (pygame.key.get_mods() & pygame.KMOD_SHIFT) and MOUSE_TEAR
                elif e.button == 3:  # RMB - toggle ball
                    if not ball_exists:
                        ball_exists = True
                        ball_pos[:] = pygame.mouse.get_pos()
                        ball_vel[:] = 0.0
                    else:
                        ball_exists = False

            elif e.type == pygame.MOUSEBUTTONUP:
                if e.button == 1:
                    dragging = False
                    tearing_drag = False

        # ------------- Physics
        if not paused:
            # Cloth integrate
            acc = np.tile(GRAVITY, (P.shape[0], 1))
            apply_wind(acc, phases, t)

            V += acc * dt
            V *= DAMPING
            P += V * dt

            # Keep inside window
            P[:, 0] = np.clip(P[:, 0], 2, WIDTH-2)
            P[:, 1] = np.clip(P[:, 1], 2, HEIGHT-2)

            # Satisfy constraints multiple passes
            for _ in range(SOLVER_ITERS):
                satisfy_spring_constraints(P, springs, anchors, TEAR_RATIO)

            # Ball physics + cloth collision
            if ball_exists:
                ball_vel += GRAVITY * dt
                ball_pos += ball_vel * dt

                # simple wall bounce
                if ball_pos[0] - BALL_RADIUS < 0:
                    ball_pos[0] = BALL_RADIUS
                    ball_vel[0] = -ball_vel[0] * 0.85
                if ball_pos[0] + BALL_RADIUS > WIDTH:
                    ball_pos[0] = WIDTH - BALL_RADIUS
                    ball_vel[0] = -ball_vel[0] * 0.85
                if ball_pos[1] - BALL_RADIUS < 0:
                    ball_pos[1] = BALL_RADIUS
                    ball_vel[1] = -ball_vel[1] * 0.85
                if ball_pos[1] + BALL_RADIUS > HEIGHT:
                    ball_pos[1] = HEIGHT - BALL_RADIUS
                    ball_vel[1] = -ball_vel[1] * 0.85

                collide_ball_with_nodes(P, V, ball_pos, BALL_RADIUS, e=0.8)

            # Mouse interactions (continuous while pressed)
            if dragging:
                mx, my = pygame.mouse.get_pos()
                if tearing_drag and MOUSE_TEAR:
                    mouse_tear_near(P, springs, (mx, my), radius=MOUSE_TEAR_RADIUS)
                else:
                    mouse_drag_move_node(P, V, anchors, (mx, my), pick_radius=36.0)

        # ------------- Draw
        screen.fill(BG_COLOR)
        pygame.draw.rect(screen, FLOOR_COLOR, pygame.Rect(0, HEIGHT-6, WIDTH, 6))

        if RENDER_SPRINGS_ONLY:
            draw_cloth_springs(screen, P, springs, color=(180, 200, 255), w=2)
        else:
            draw_cloth_lines(screen, P, ROWS, COLS, color=(200, 220, 255), w=2)

        if ball_exists:
            pygame.draw.circle(screen, BALL_COLOR, (int(ball_pos[0]), int(ball_pos[1])), BALL_RADIUS)

        if SHOW_GEOM:
            draw_segments(screen, segments)

        if DEV_OVERLAY:
            l1 = f"lvl={level_name}  springs_on={sum(1 for s in springs if s[3])}/{len(springs)}"
            l2 = f"wind={'ON' if WIND_ON else 'OFF'} base={WIND_BASE:.1f}  turb={'ON' if WIND_TURB_ON else 'OFF'} amp={WIND_TURB_AMP:.1f}  freq={WIND_TURB_FREQ:.2f}Hz"
            l3 = f"tear={'ON' if TEAR_MODE else 'OFF'}  ratio={TEAR_RATIO:.2f}  Shift+LMB to tear  RMB ball"
            l4 = "Space pause | R reset | L level | H geom | B view | D HUD | W/A/Left/S/Right wind | Z/X amp | C turb | ,/. freq | T tear | [/] ratio | Y repair"
            for i, msg in enumerate((l1, l2, l3, l4)):
                screen.blit(font.render(msg, True, TEXT_COLOR), (10, 8 + 20*i))

        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
