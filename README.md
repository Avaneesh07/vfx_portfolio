# Physics Sim: Bouncing Ball

Day 1 — first physics-based simulation using **Pygame + NumPy**.

**Controls**
- `Space`: pause/resume
- `R`: reset
- `G`: toggle gravity
- `← / →`: horizontal impulse
- `↑`: jump (only when grounded)
- **Left-click**: reposition ball

**Demo**
- Video/Report: `assets/day1_bouncing_ball_demo.mp4` *(or)* `assets/day1_bouncing_ball_demo.zip`
- Run log: `docs/DAY1_LOG.md`

## How to Run
```bash
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install pygame numpy
python bouncing_ball.py
### Day 3 — Visual & Interactivity Upgrades
- Per-ball radii (varied sizes)
- Impact flash on collisions
- Right-click to spawn new balls

See: `bouncing_balls.py` and `docs/DAY3_LOG.md`