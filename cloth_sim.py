import sys, os, json
import math
import pygame
import numpy as np

# =========================
# Config
# =========================
WIDTH, HEIGHT = 900, 600
FPS = 120

BG_COLOR    = (18, 18, 24)
FLOOR_COLOR = (35, 40, 55)
TEXT_COLOR  = (220, 220, 220)
MESH_COLOR  = (0, 255, 180)   # cloth lines

# Cloth grid
ROWS, COLS   = 26, 36
SPACING      = 18.0           # rest distance between nodes
ORIGIN_X     = 120.0
ORIGIN_Y     = 90.0

# Physics
GRAVITY      = 1800.0         # px/s^2 downward
DAMPING      = 0.0005         # small global damping (Verlet)
ITERATIONS   = 12             # constraint iterations per frame
BOUNCE       = 0.30           # collision restitution with walls/floor

# Wind (simple)
WIND_ON_START  = True
WIND_STRENGTH  = 140.0        # px/s^2 (horizontal)
WIND_STEP      = 20.0         # per keypress

# Dragging
DRAG_PICK_RADIUS = 28.0       # select nearest node within this radius
DRAG_STIFFNESS   = 1600.0     # spring K when dragging
DRAG_DAMP        = 22.0       # damper

# Save/Load
PRESET_PATH = os.path.join("assets", "presets", "cloth_pins.json")

# =========================
# Helpers
# =========================
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

def node_from_mouse(mx, my, posC):
    d2 = (posC[..., 0] - mx) ** 2 + (posC[..., 1] - my) ** 2
    idx = int(np.argmin(d2))
    r = idx // posC.shape[1]
    c = idx %  posC.shape[1]
    return r, c, float(d2.ravel()[idx]) ** 0.5

def toggle_pin_nearest(mx, my, posC, pinned, pin_pos):
    r, c, _ = node_from_mouse(mx, my, posC)
    pinned[r, c] = ~pinned[r, c]
    if pinned[r, c]:
        pin_pos[r, c] = posC[r, c]
    return r, c

def unpin_all(pinned):
    pinned[:] = False

def save_preset(path, posC, pinned):
    data = {"pos": posC.tolist(), "pinned": pinned.tolist()}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)

def load_preset(path, posC, prevC, pinned, pin_pos):
    if not os.path.isfile(path):
        return False
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    pos_arr = np.array(data.get("pos", []), dtype=float)
    pin_arr = np.array(data.get("pinned", []), dtype=bool)
    if pos_arr.shape != posC.shape or pin_arr.shape != pinned.shape:
        return False
    posC[:] = pos_arr
    prevC[:] = pos_arr
    pinned[:] = pin_arr
    pin_pos[pinned] = posC[pinned]
    return True

def draw_cloth_lines(surface, posC, color, w=2):
    rows, cols = posC.shape[:2]
    # horizontal
    for r in range(rows):
        for c in range(cols - 1):
            x1, y1 = posC[r, c]
            x2, y2 = posC[r, c+1]
            pygame.draw.line(surface, color, (int(x1), int(y1)), (int(x2), int(y2)), w)
    # vertical
    for r in range(rows - 1):
        for c in range(cols):
            x1, y1 = posC[r, c]
            x2, y2 = posC[r+1, c]
            pygame.draw.line(surface, color, (int(x1), int(y1)), (int(x2), int(y2)), w)

# =========================
# Constraints
# =========================
def satisfy_distance(p0, p1, w0, w1, rest, stiffness=1.0):
    delta = p1 - p0
    d2 = float(delta[0]*delta[0] + delta[1]*delta[1])
    if d2 < 1e-12:
        return
    d = math.sqrt(d2)
    diff = (d - rest) / d
    inv_sum = w0 + w1
    if inv_sum <= 0.0:
        return
    corr = stiffness * diff * delta
    p0 += (w0 / inv_sum) * corr
    p1 -= (w1 / inv_sum) * corr

def project_constraints(posC, inv_mass, spacing, iters=12):
    rows, cols = posC.shape[:2]
    rest = spacing
    rest_diag = spacing * (2 ** 0.5)
    for _ in range(iters):
        # Structural
        for r in range(rows):
            for c in range(cols - 1):
                satisfy_distance(posC[r, c], posC[r, c+1],
                                 inv_mass[r, c], inv_mass[r, c+1], rest)
        for r in range(rows - 1):
            for c in range(cols):
                satisfy_distance(posC[r, c], posC[r+1, c],
                                 inv_mass[r, c], inv_mass[r+1, c], rest)
        # Shear
        for r in range(rows - 1):
            for c in range(cols - 1):
                satisfy_distance(posC[r, c],   posC[r+1, c+1],
                                 inv_mass[r, c], inv_mass[r+1, c+1], rest_diag, 0.9)
                satisfy_distance(posC[r, c+1], posC[r+1, c],
                                 inv_mass[r, c+1], inv_mass[r+1, c], rest_diag, 0.9)

# =========================
# Collisions with bounds
# =========================
def collide_bounds(posC, prevC, bounce=0.30):
    rows, cols = posC.shape[:2]
    for r in range(rows):
        for c in range(cols):
            x, y = posC[r, c]
            px, py = prevC[r, c]
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
                posC[r, c, 0] = x
                posC[r, c, 1] = y
                prevC[r, c, 0] = x - vx
                prevC[r, c, 1] = y - vy

# =========================
# Main
# =========================
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Day 14: Cloth pinning + save/load (S/O, Ctrl+S/Ctrl+O, F5/F9)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 18)

    posC, prevC = build_grid(ROWS, COLS, ORIGIN_X, ORIGIN_Y, SPACING)
    acc = np.zeros_like(posC)

    pinned  = np.zeros((ROWS, COLS), dtype=bool)
    pin_pos = np.zeros((ROWS, COLS, 2), dtype=float)
    inv_mass = np.ones((ROWS, COLS), dtype=float)

    dragging = False
    drag_idx = (0, 0)

    wind_on = WIND_ON_START
    wind_strength = WIND_STRENGTH
    paused = False

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                mods = pygame.key.get_mods()

                # Sim control
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    posC, prevC = build_grid(ROWS, COLS, ORIGIN_X, ORIGIN_Y, SPACING)
                    acc[:] = 0; pinned[:] = False; inv_mass[:] = 1.0; pin_pos[:] = 0
                elif event.key == pygame.K_w:
                    wind_on = not wind_on
                elif event.key == pygame.K_LEFT:
                    wind_strength = max(0.0, wind_strength - WIND_STEP)
                elif event.key == pygame.K_RIGHT:
                    wind_strength += WIND_STEP

                # Pin controls
                elif event.key == pygame.K_p:
                    mx, my = pygame.mouse.get_pos()
                    toggle_pin_nearest(mx, my, posC, pinned, pin_pos)
                elif event.key == pygame.K_u:
                    unpin_all(pinned)

                # Save (F5 or Ctrl+S or S)
                elif event.key == pygame.K_F5 or (event.key == pygame.K_s and (mods & pygame.KMOD_CTRL)) or (event.key == pygame.K_s):
                    save_preset(PRESET_PATH, posC, pinned)

                # Load (F9 or Ctrl+O or O)
                elif event.key == pygame.K_F9 or (event.key == pygame.K_o and (mods & pygame.KMOD_CTRL)) or (event.key == pygame.K_o):
                    load_preset(PRESET_PATH, posC, prevC, pinned, pin_pos)

            # Mouse
            if event.type == pygame.MOUSEBUTTONDOWN:
                # Shift+LMB = toggle pin nearest
                if event.button == 1 and (pygame.key.get_mods() & pygame.KMOD_SHIFT):
                    mx, my = pygame.mouse.get_pos()
                    toggle_pin_nearest(mx, my, posC, pinned, pin_pos)
                # LMB = start drag
                elif event.button == 1:
                    mx, my = pygame.mouse.get_pos()
                    r, c, dist = node_from_mouse(mx, my, posC)
                    if dist <= DRAG_PICK_RADIUS:
                        dragging = True
                        drag_idx = (r, c)
                # RMB = unpin all quick
                elif event.button == 3:
                    unpin_all(pinned)

            if event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1 and dragging:
                    dragging = False

        # Update inv_mass from pins
        inv_mass[:] = 1.0
        inv_mass[pinned] = 0.0

        if not paused:
            # 1) forces
            acc[:] = 0.0
            acc[..., 1] += GRAVITY
            if wind_on:
                acc[..., 0] += wind_strength

            # drag spring
            if dragging:
                mx, my = pygame.mouse.get_pos()
                r, c = drag_idx
                if inv_mass[r, c] > 0.0:
                    dx = mx - posC[r, c, 0]
                    dy = my - posC[r, c, 1]
                    vx = posC[r, c, 0] - prevC[r, c, 0]
                    vy = posC[r, c, 1] - prevC[r, c, 1]
                    Fx = DRAG_STIFFNESS * dx - DRAG_DAMP * vx
                    Fy = DRAG_STIFFNESS * dy - DRAG_DAMP * vy
                    acc[r, c, 0] += Fx
                    acc[r, c, 1] += Fy

            # 2) verlet
            old_pos = posC.copy()
            free = inv_mass > 0.0
            posC[free] = posC[free] + (posC[free] - prevC[free]) * (1.0 - DAMPING) + acc[free] * (dt * dt)
            prevC[free] = old_pos[free]

            # 3) constraints
            project_constraints(posC, inv_mass, SPACING, ITERATIONS)

            # 4) enforce pins
            posC[pinned] = pin_pos[pinned]
            prevC[pinned] = pin_pos[pinned]

            # 5) bounds
            collide_bounds(posC, prevC, BOUNCE)

        # ---- draw
        screen.fill(BG_COLOR)
        pygame.draw.rect(screen, FLOOR_COLOR, pygame.Rect(0, HEIGHT - 6, WIDTH, 6))

        draw_cloth_lines(screen, posC, MESH_COLOR, w=2)

        # pins
        for (r, c), is_pin in np.ndenumerate(pinned):
            if is_pin:
                x, y = posC[r, c]
                pygame.draw.circle(screen, (255, 240, 40), (int(x), int(y)), 4)

        # HUD
        pinned_count = int(np.count_nonzero(pinned))
        draw_text(screen, f"Cloth {ROWS}x{COLS}  wind={'ON' if wind_on else 'OFF'} ({wind_strength:.0f})  pins={pinned_count}", 10, 10, font)
        draw_text(screen, "Space=pause  R=reset  ←/→ wind-+  W toggle wind  LMB drag  Shift+LMB toggle pin  P pin-nearest  U unpin-all", 10, 32, font)
        draw_text(screen, "Save: S / Ctrl+S / F5   Load: O / Ctrl+O / F9   (assets/presets/cloth_pins.json)", 10, 54, font)

        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()

