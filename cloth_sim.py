# cloth_sim.py
import os, sys, json, math
import pygame
import numpy as np
from spatial_hash import SpatialHash

# =========================================
# Window / Colors
# =========================================
WIDTH, HEIGHT = 900, 600
FPS = 120

BG_COLOR     = (18, 18, 24)
FLOOR_COLOR  = (35, 40, 55)
TEXT_COLOR   = (220, 220, 220)
GEOM_COLOR   = (0, 255, 180)
MESH_COLOR   = (190, 210, 255)
PIN_COLOR    = (255, 240, 40)
BALL_COLOR   = (220, 200, 255)

# =========================================
# Cloth Grid & Physics (Day 14 baseline)
# =========================================
ROWS, COLS = 26, 36
SPACING    = 18.0                 # rest distance
ORIGIN_X   = 120.0
ORIGIN_Y   = 90.0

# Verlet-like integrator parameters
GRAVITY        = 1800.0           # px/s^2 downward
DAMPING        = 0.0005           # global damping factor (tiny) for Verlet
ITERATIONS     = 12               # constraint iterations per frame
BOUNCE         = 0.30             # restitution with walls/floor

# Wind (base + turbulence)
WIND_ON        = True
WIND_BASE      = 140.0            # px/s^2 (horizontal)
WIND_STEP      = 20.0
TURB_ON        = True
TURB_AMP       = 140.0            # px/s^2
TURB_FREQ      = 1.2              # Hz

# Tearing
TEAR_MODE      = True
TEAR_RATIO     = 1.8              # break when length > rest * ratio
MOUSE_TEAR     = True
MOUSE_TEAR_RADIUS = 28.0

# Ball
BALL_RADIUS    = 18

# Toggles
SHOW_GEOM           = True
DEV_OVERLAY         = True
RENDER_SPRINGS_ONLY = True  # 'B' toggles springs view vs grid lines

# =========================================
# Day 15: fast picking via Spatial Hash
# =========================================
PICK_CELL_SIZE = 48.0
PICK_RADIUS    = 28.0          # how close the click must be to grab a node

# =========================================
# Presets (Day 14)
# =========================================
PRESET_PATH = os.path.join("assets", "presets", "cloth_pins.json")


# =========================================
# Level Geometry
# =========================================
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


# =========================================
# Helpers
# =========================================
def grid_index(r, c, cols):
    return r * cols + c

def draw_text(surface, text, x, y, font):
    surface.blit(font.render(text, True, TEXT_COLOR), (x, y))

def build_grid(rows, cols, origin_x, origin_y, spacing):
    pos = np.zeros((rows, cols, 2), dtype=float)
    for r in range(rows):
        for c in range(cols):
            pos[r, c, 0] = origin_x + c * spacing
            pos[r, c, 1] = origin_y + r * spacing
    prev = pos.copy()
    return pos, prev

def build_springs(rows, cols, spacing):
    """
    Structural springs (right and down) stored as [i, j, rest, enabled_bool],
    with i/j being flat indices over rows*cols.
    """
    springs = []
    for r in range(rows):
        for c in range(cols):
            i = grid_index(r, c, cols)
            if c + 1 < cols:
                j = grid_index(r, c+1, cols)
                springs.append([i, j, spacing, True])
            if r + 1 < rows:
                j = grid_index(r+1, c, cols)
                springs.append([i, j, spacing, True])
    return springs

def apply_wind(acc_flat, phases, t):
    # base + optional turbulence in +X
    if WIND_ON:
        acc_flat[:, 0] += WIND_BASE
    if TURB_ON and TURB_AMP > 1e-6:
        acc_flat[:, 0] += TURB_AMP * np.sin(2*math.pi*TURB_FREQ * t + phases)

def satisfy_spring_constraints(P_flat, springs, ratio_break=TEAR_RATIO):
    """
    Positional spring correction with optional tearing.
    P_flat: shape (N,2)
    """
    for s in springs:
        if not s[3]:
            continue
        i, j, rest = s[0], s[1], s[2]
        d = P_flat[j] - P_flat[i]
        d2 = float(d[0]*d[0] + d[1]*d[1])
        if d2 <= 1e-12:
            continue
        dist = math.sqrt(d2)
        if TEAR_MODE and dist > rest * ratio_break:
            s[3] = False
            continue
        diff = (dist - rest) / dist
        corr = 0.5 * diff * d
        P_flat[i] += corr
        P_flat[j] -= corr

def collide_ball_with_nodes(P_flat, V_flat, ball_pos, ball_r, e=0.8):
    r2 = float(ball_r) * float(ball_r)
    for k in range(P_flat.shape[0]):
        d = P_flat[k] - ball_pos
        d2 = float(d[0]*d[0] + d[1]*d[1])
        if d2 < r2:
            dist = math.sqrt(d2) if d2 > 1e-12 else 1e-6
            n_hat = d / dist
            overlap = ball_r - dist
            P_flat[k] += n_hat * overlap
            vn = float(V_flat[k,0]*n_hat[0] + V_flat[k,1]*n_hat[1])
            V_flat[k] -= (1.0 + e) * vn * n_hat

def collide_bounds(pos, prev, bounce=0.30):
    rows, cols = pos.shape[:2]
    for r in range(rows):
        for c in range(cols):
            x, y = pos[r, c]
            px, py = prev[r, c]
            vx, vy = x - px, y - py
            hit = False
            if x < 0:
                x = 0; vx = -vx * bounce; hit = True
            elif x > WIDTH - 1:
                x = WIDTH - 1; vx = -vx * bounce; hit = True
            if y < 0:
                y = 0; vy = -vy * bounce; hit = True
            elif y > HEIGHT - 6:
                y = HEIGHT - 6; vy = -vy * bounce; hit = True
            if hit:
                pos[r, c, 0]  = x
                pos[r, c, 1]  = y
                prev[r, c, 0] = x - vx
                prev[r, c, 1] = y - vy

def mouse_drag_move_node(P_flat, V_flat, idx, mx, my, k=1600.0, d=22.0):
    # Overwrite directly (simple and stable for dragging)
    P_flat[idx, 0] = mx
    P_flat[idx, 1] = my
    V_flat[idx, :] = 0.0

def mouse_tear_near(P_flat, springs, mx, my, radius=MOUSE_TEAR_RADIUS):
    r2 = radius * radius
    for s in springs:
        if not s[3]:
            continue
        i, j = s[0], s[1]
        a = P_flat[i]; b = P_flat[j]
        ab = b - a
        ab2 = float(ab[0]*ab[0] + ab[1]*ab[1])
        if ab2 <= 1e-12:
            continue
        t = ((mx-a[0])*ab[0] + (my-a[1])*ab[1]) / ab2
        t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
        qx = a[0] + t*ab[0]
        qy = a[1] + t*ab[1]
        dx = qx - mx
        dy = qy - my
        if dx*dx + dy*dy <= r2:
            s[3] = False

def draw_cloth_springs(surface, P_flat, springs, color=(180, 200, 255), w=2):
    for s in springs:
        if not s[3]:
            continue
        i, j = s[0], s[1]
        x1, y1 = int(P_flat[i,0]), int(P_flat[i,1])
        x2, y2 = int(P_flat[j,0]), int(P_flat[j,1])
        pygame.draw.line(surface, color, (x1, y1), (x2, y2), w)

def draw_cloth_grid(surface, pos, color=(200, 220, 255), w=2):
    rows, cols = pos.shape[:2]
    # horizontal
    for r in range(rows):
        for c in range(cols - 1):
            x1, y1 = pos[r, c]
            x2, y2 = pos[r, c+1]
            pygame.draw.line(surface, color, (int(x1), int(y1)), (int(x2), int(y2)), w)
    # vertical
    for r in range(rows - 1):
        for c in range(cols):
            x1, y1 = pos[r, c]
            x2, y2 = pos[r+1, c]
            pygame.draw.line(surface, color, (int(x1), int(y1)), (int(x2), int(y2)), w)


# =========================================
# Day 15 helpers: picking hash
# =========================================
def rebuild_pick_hash(pos, pick_hash, radius):
    rows, cols = pos.shape[:2]
    items = []
    r = float(radius)
    for rr in range(rows):
        for cc in range(cols):
            x, y = float(pos[rr, cc, 0]), float(pos[rr, cc, 1])
            items.append((rr * cols + cc, (x - r, y - r, x + r, y + r)))
    pick_hash.rebuild(items)

def nearest_node_from_hash(mx, my, pos, pick_hash, radius):
    rows, cols = pos.shape[:2]
    ids = pick_hash.neighbors_of_point(mx, my, radius * 2.0)
    best = None
    best_d2 = 1e30
    for id_ in ids:
        rr = id_ // cols
        cc = id_ %  cols
        dx = pos[rr, cc, 0] - mx
        dy = pos[rr, cc, 1] - my
        d2 = dx*dx + dy*dy
        if d2 < best_d2:
            best_d2 = d2
            best = (rr, cc)
    if best is None:
        return None, None, 1e30**0.5
    return best[0], best[1], best_d2**0.5


# =========================================
# Save / Load pin presets (Day 14)
# =========================================
def save_preset(path, pos, pinned):
    data = {"pos": pos.tolist(), "pinned": pinned.tolist()}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)

def load_preset(path, pos, prev, pinned, pin_pos):
    if not os.path.isfile(path):
        return False
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    pos_arr = np.array(data.get("pos", []), dtype=float)
    pin_arr = np.array(data.get("pinned", []), dtype=bool)
    if pos_arr.shape != pos.shape or pin_arr.shape != pinned.shape:
        return False
    pos[:] = pos_arr
    prev[:] = pos_arr
    pinned[:] = pin_arr
    pin_pos[pinned] = pos[pinned]
    return True


# =========================================
# Main
# =========================================
def main():
    global SHOW_GEOM, DEV_OVERLAY, RENDER_SPRINGS_ONLY
    global WIND_ON, WIND_BASE, TURB_ON, TURB_AMP, TURB_FREQ
    global TEAR_MODE, TEAR_RATIO

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Day 15: Cloth fast picking + Day 14 pins/presets")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 18)

    # Level
    level_id = 1
    segments, level_name = build_level(level_id)

    # Cloth state
    pos, prev = build_grid(ROWS, COLS, ORIGIN_X, ORIGIN_Y, SPACING)
    springs = build_springs(ROWS, COLS, SPACING)
    pinned  = np.zeros((ROWS, COLS), dtype=bool)
    pin_pos = np.zeros((ROWS, COLS, 2), dtype=float)

    # Flattened views
    P_flat = pos.reshape(-1, 2)
    V_flat = np.zeros_like(P_flat)
    N = P_flat.shape[0]

    rng = np.random.default_rng()
    phases = rng.uniform(0, 2*math.pi, size=N)

    # Ball
    ball_exists = False
    ball_pos = np.array([650.0, 220.0], dtype=float)
    ball_vel = np.array([0.0, 0.0], dtype=float)

    # Picking hash (Day 15)
    pick_hash = SpatialHash(cell_size=PICK_CELL_SIZE)
    rebuild_pick_hash(pos, pick_hash, PICK_RADIUS)

    paused = False
    dragging = False
    drag_idx = -1
    tearing_drag = False

    running = True
    t = 0.0

    while running:
        dt = max(1, clock.tick(FPS)) / 1000.0
        t += dt

        # ---------------- Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                mods = pygame.key.get_mods()

                # System / view
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    pos, prev = build_grid(ROWS, COLS, ORIGIN_X, ORIGIN_Y, SPACING)
                    springs = build_springs(ROWS, COLS, SPACING)
                    pinned[:] = False
                    pin_pos[:] = 0
                    rebuild_pick_hash(pos, pick_hash, PICK_RADIUS)
                elif event.key == pygame.K_l:
                    level_id = 2 if level_id == 1 else 1
                    segments, level_name = build_level(level_id)
                elif event.key == pygame.K_h:
                    SHOW_GEOM = not SHOW_GEOM
                elif event.key == pygame.K_b:
                    RENDER_SPRINGS_ONLY = not RENDER_SPRINGS_ONLY
                elif event.key == pygame.K_d:
                    DEV_OVERLAY = not DEV_OVERLAY

                # Wind
                elif event.key == pygame.K_w:
                    WIND_ON = not WIND_ON
                elif event.key == pygame.K_a or event.key == pygame.K_LEFT:
                    WIND_BASE = max(0.0, WIND_BASE - WIND_STEP)
                elif event.key == pygame.K_RIGHT:
                    WIND_BASE += WIND_STEP
                elif event.key == pygame.K_s and not (mods & pygame.KMOD_CTRL):
                    WIND_BASE = 0.0
                elif event.key == pygame.K_z:
                    TURB_AMP = max(0.0, TURB_AMP - 20.0)
                elif event.key == pygame.K_x:
                    TURB_AMP += 20.0
                elif event.key == pygame.K_c:
                    TURB_ON = not TURB_ON
                elif event.key == pygame.K_COMMA:
                    TURB_FREQ = max(0.05, TURB_FREQ - 0.1)
                elif event.key == pygame.K_PERIOD:
                    TURB_FREQ += 0.1

                # Tearing
                elif event.key == pygame.K_t:
                    TEAR_MODE = not TEAR_MODE
                elif event.key == pygame.K_LEFTBRACKET:
                    TEAR_RATIO = max(1.05, TEAR_RATIO - 0.05)
                elif event.key == pygame.K_RIGHTBRACKET:
                    TEAR_RATIO = min(3.0, TEAR_RATIO + 0.05)
                elif event.key == pygame.K_y:
                    for s in springs:
                        s[3] = True

                # Pins (Day 14)
                elif event.key == pygame.K_p:
                    mx, my = pygame.mouse.get_pos()
                    rr, cc, dist = nearest_node_from_hash(mx, my, pos, pick_hash, PICK_RADIUS)
                    if rr is not None and dist <= PICK_RADIUS:
                        pinned[rr, cc] = ~pinned[rr, cc]
                        if pinned[rr, cc]:
                            pin_pos[rr, cc] = pos[rr, cc]
                elif event.key == pygame.K_u:
                    pinned[:] = False

                # Save / Load presets
                elif event.key == pygame.K_F5 or (event.key == pygame.K_s and (mods & pygame.KMOD_CTRL)) or (event.key == pygame.K_s and (mods & pygame.KMOD_SHIFT)):
                    save_preset(PRESET_PATH, pos, pinned)
                elif event.key == pygame.K_F9 or (event.key == pygame.K_o and (mods & pygame.KMOD_CTRL)) or (event.key == pygame.K_o and (mods & pygame.KMOD_SHIFT)):
                    load_preset(PRESET_PATH, pos, prev, pinned, pin_pos)
                    rebuild_pick_hash(pos, pick_hash, PICK_RADIUS)

            # Mouse buttons
            if event.type == pygame.MOUSEBUTTONDOWN:
                # Shift+LMB = toggle nearest pin
                if event.button == 1 and (pygame.key.get_mods() & pygame.KMOD_SHIFT):
                    mx, my = pygame.mouse.get_pos()
                    rr, cc, dist = nearest_node_from_hash(mx, my, pos, pick_hash, PICK_RADIUS)
                    if rr is not None and dist <= PICK_RADIUS:
                        pinned[rr, cc] = ~pinned[rr, cc]
                        if pinned[rr, cc]:
                            pin_pos[rr, cc] = pos[rr, cc]
                # LMB = drag nearest
                elif event.button == 1:
                    mx, my = pygame.mouse.get_pos()
                    rr, cc, dist = nearest_node_from_hash(mx, my, pos, pick_hash, PICK_RADIUS)
                    if rr is not None and dist <= PICK_RADIUS:
                        dragging = True
                        drag_idx = rr * COLS + cc
                        tearing_drag = (pygame.key.get_mods() & pygame.KMOD_SHIFT) and MOUSE_TEAR
                # RMB = toggle ball
                elif event.button == 3:
                    ball_exists = not ball_exists
                    if ball_exists:
                        ball_pos[:] = pygame.mouse.get_pos()
                        ball_vel[:] = 0.0

            if event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1 and dragging:
                    dragging = False
                    drag_idx = -1
                    tearing_drag = False

        # ---------------- Physics
        if not paused:
            # 1) integrate (simple Verlet flavor)
            old_pos = pos.copy()
            acc_flat = np.zeros_like(P_flat)
            acc_flat[:, 1] += GRAVITY
            apply_wind(acc_flat, phases, t)

            pos[:] = pos + (pos - prev) * (1.0 - DAMPING) + acc_flat.reshape(ROWS, COLS, 2) * (dt * dt)
            prev[:] = old_pos

            # 2) mouse interaction
            if dragging and 0 <= drag_idx < N:
                mx, my = pygame.mouse.get_pos()
                if tearing_drag and MOUSE_TEAR:
                    mouse_tear_near(P_flat, springs, mx, my, radius=MOUSE_TEAR_RADIUS)
                mouse_drag_move_node(P_flat, V_flat, drag_idx, mx, my)

            # 3) satisfy constraints
            for _ in range(ITERATIONS):
                satisfy_spring_constraints(P_flat, springs, TEAR_RATIO)

            # 4) enforce pins
            pos[pinned] = pin_pos[pinned]
            prev[pinned] = pin_pos[pinned]

            # 5) bounds
            collide_bounds(pos, prev, BOUNCE)

            # 6) ball motion + collide with cloth nodes
            if ball_exists:
                ball_vel[1] += GRAVITY * dt
                ball_pos += ball_vel * dt
                if ball_pos[0] - BALL_RADIUS < 0:
                    ball_pos[0] = BALL_RADIUS;  ball_vel[0] = -ball_vel[0] * 0.85
                if ball_pos[0] + BALL_RADIUS > WIDTH:
                    ball_pos[0] = WIDTH - BALL_RADIUS; ball_vel[0] = -ball_vel[0] * 0.85
                if ball_pos[1] - BALL_RADIUS < 0:
                    ball_pos[1] = BALL_RADIUS;  ball_vel[1] = -ball_vel[1] * 0.85
                if ball_pos[1] + BALL_RADIUS > HEIGHT:
                    ball_pos[1] = HEIGHT - BALL_RADIUS; ball_vel[1] = -ball_vel[1] * 0.85
                collide_ball_with_nodes(P_flat, V_flat, ball_pos, BALL_RADIUS, e=0.8)

            # Day 15: rebuild picking hash AFTER positions are final this frame
            rebuild_pick_hash(pos, pick_hash, PICK_RADIUS)

        # ---------------- Draw
        screen.fill(BG_COLOR)
        pygame.draw.rect(screen, FLOOR_COLOR, pygame.Rect(0, HEIGHT - 6, WIDTH, 6))

        if RENDER_SPRINGS_ONLY:
            draw_cloth_springs(screen, P_flat, springs, color=(180, 200, 255), w=2)
        else:
            draw_cloth_grid(screen, pos, color=MESH_COLOR, w=2)

        if SHOW_GEOM:
            draw_segments(screen, segments)

        if ball_exists:
            pygame.draw.circle(screen, BALL_COLOR, (int(ball_pos[0]), int(ball_pos[1])), BALL_RADIUS)

        for (r, c), is_pin in np.ndenumerate(pinned):
            if is_pin:
                x, y = pos[r, c]
                pygame.draw.circle(screen, PIN_COLOR, (int(x), int(y)), 4)

        draw_text(screen, f"Cloth {ROWS}x{COLS}  wind={'ON' if WIND_ON else 'OFF'} ({WIND_BASE:.0f})  pins={int(np.count_nonzero(pinned))}", 10, 10, font)
        draw_text(screen, "LMB drag  Shift+LMB toggle pin  P pin-nearest  U unpin-all", 10, 32, font)
        draw_text(screen, "Save: F5/Ctrl+S  Load: F9/Ctrl+O  (assets/presets/cloth_pins.json)", 10, 54, font)
        draw_text(screen, f"Pick cell={int(PICK_CELL_SIZE)}  PickR={int(PICK_RADIUS)}", 10, 76, font)

        if DEV_OVERLAY:
            l1 = f"lvl={level_name}  springs_on={sum(1 for s in springs if s[3])}/{len(springs)}"
            l2 = f"wind={'ON' if WIND_ON else 'OFF'} base={WIND_BASE:.1f}  turb={'ON' if TURB_ON else 'OFF'} amp={TURB_AMP:.1f}  freq={TURB_FREQ:.2f}Hz"
            l3 = f"tear={'ON' if TEAR_MODE else 'OFF'}  ratio={TEAR_RATIO:.2f}  RMB ball"
            l4 = "Space pause | R reset | L level | H geom | B view | D HUD | W/A/Left/S/Right wind | Z/X amp | C turb | ,/. freq | T tear | [/] ratio | Y repair"
            for i, msg in enumerate((l1, l2, l3, l4)):
                screen.blit(font.render(msg, True, TEXT_COLOR), (10, 100 + 20*i))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
