# Physics Sim: Bouncing Ball

Day 1 — first physics-based simulation using **Pygame + NumPy**.

# Physics Sim: Bouncing Ball (Pygame + NumPy)

Interactive physics sandbox built step-by-step as a VFX + Computer Science portfolio project.

---

## How to Run
```bat
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install pygame numpy
python bouncing_balls.py   # full version with levels, particles & shake


Day 1 – Single Bouncing Ball

Basic gravity, wall/floor collisions, HUD.

Controls: Space, R, G, ←/→, ↑, LMB.

Demo: assets/day1_bouncing_ball_demo.mp4 or .zip

Log: docs/DAY1_LOG.md

Day 2 – Multi-Ball Simulation

Multiple balls moving and colliding elastically.

Pairwise ball-ball collision resolution.

Log: docs/DAY2_LOG.md

Day 3 – Visual & Interactivity Upgrades

Per-ball radii (sizes vary).

Impact flash on collisions.

Right-click to spawn new balls.

Log: docs/DAY3_LOG.md

Day 4 – Mass & Momentum

Mass proportional to radius².

Momentum (Σp) HUD added.

Mass-aware impulse and friction system.

Log: docs/DAY4_LOG.md

Day 5 – Recording & Runbook

Steps Recorder used for demo documentation.

Created a consistent daily project log format.

Log: docs/DAY5_LOG.md

Day 6 – Level Geometry

Added ramps, platforms, and funnels.

Keys: H toggle geometry, L switch levels.

Log: docs/DAY6_LOG.md

Day 7 – Impact Particles + Camera Shake

Sparks on ball–ball and ball–geometry impacts.

Subtle screen shake for strong collisions.

Demo: assets/day7_particles_demo.zip

Log: docs/DAY7_LOG.md

Controls (current build)
Key	Action
Space	Pause / Resume
R	Reset
G	Toggle Gravity
← / →	Apply horizontal impulse
↑	Jump (if grounded)
LMB	Move nearest ball
RMB	Spawn a new ball
L	Cycle level (Ramp / Funnel)
H	Toggle geometry visibility
Key Files

bouncing_ball.py — Day 1 single-ball baseline

bouncing_balls.py — Multi-ball build with geometry, particles, and shake

assets/ — Video and Steps Recorder demos

docs/ — Daily logs (DAY1–DAY7)

progress.md — Overall tracking