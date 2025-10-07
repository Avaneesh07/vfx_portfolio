# 🎮 Physics Sim: Bouncing Ball (Pygame + NumPy)

Interactive physics sandbox built step-by-step as a **VFX + Computer Science portfolio project**.  
Each day adds new simulation or visual effects concepts — perfect for combining CS fundamentals with real-time animation skills.

---

## 🚀 How to Run
```bat
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install pygame numpy
python bouncing_balls.py   # full version with levels, particles & shake


🧱 Day 1 — Single Bouncing Ball

Implemented gravity, wall & floor collisions, and pause/reset controls.

Added on-screen HUD for velocity and position.

Controls: Space, R, G, ←/→, ↑, LMB.

🎥 Demo: assets/day1_bouncing_ball_demo.mp4
 or assets/day1_bouncing_ball_demo.zip

🗒️ Log: docs/DAY1_LOG.md

⚙️ Day 2 — Multi-Ball Simulation

Created multiple balls with independent motion and gravity.

Added ball–ball elastic collision resolution and overlap correction.

Vectorized physics using NumPy arrays.

🎥 Demo: (recorded during testing phase)

🗒️ Log: docs/DAY2_LOG.md

🎨 Day 3 — Visual & Interactivity Upgrades

Introduced per-ball radii (different sizes & colors).

Added impact flashes and mouse-based repositioning.

Right-click spawns new balls dynamically.

🎥 Demo: (interactive visual test with variable sizes)

🗒️ Log: docs/DAY3_LOG.md

⚖️ Day 4 — Mass & Momentum

Mass scales with radius² for realistic energy transfer.

Added total momentum (Σp) display to HUD.

Implemented mass-aware friction & impulse handling.

🎥 Demo: (momentum conservation test)

🗒️ Log: docs/DAY4_LOG.md

🧩 Day 5 — Recording & Runbook

Used Windows Steps Recorder for live simulation capture.

Created documentation structure under /docs for consistent daily logs.

🎥 Demo: assets/day5_mass_collisions_demo.zip

🗒️ Log: docs/DAY5_LOG.md

🧗 Day 6 — Level Geometry

Added ramps, platforms, and funnels (Level 1 & 2 scenes).

Toggle geometry visibility (H) or switch levels (L).

Balls interact and slide smoothly on sloped surfaces.

🎥 Demo: assets/day6_geometry_demo.zip

🗒️ Log: docs/DAY6_LOG.md

💥 Day 7 — Impact Particles + Camera Shake

Added spark particles for ball–ball and ball–geometry collisions.

Implemented subtle camera shake proportional to impact force.

Polished visuals: smoother trails and decaying shake.

🎥 Demo: assets/day7_particles_demo.zip

🗒️ Log: docs/DAY7_LOG.md

⚡ **Day 8 — Energy Visualization**

Introduced **real-time kinetic + potential energy bars** for each ball.  
Shows ΣKE, ΣPE, ΣE totals in the HUD.  
Spark particles + camera shake from Day 7 retained.  

🎥 **Demo:** `assets/day8_energy_visualization_demo.zip`  
🗒️ **Log:** `docs/DAY8_LOG.md`

🌈 **Day 9 — Velocity Heatmap + Debug HUD**

Velocity-based colors (blue→yellow→red) show per-ball speed; toggle **V**.  
Developer overlay shows s_max (95th percentile), trail alpha, and key hints; toggle **D**.  
All Day 7–8 effects retained (particles, shake, energy bars).

🎥 **Demo:** `assets/day9_velocity_heatmap_demo.zip`  
🗒️ **Log:** `docs/DAY9_LOG.md`

🪢 **Day 10 — Mouse Drag Spring**

Grab any ball with **LMB** and move it with a springy “rubber band”.  
Adjust spring stiffness with **+ / -**.  
Works with heatmap, particles, camera shake, energy bars, and geometry.

🎥 **Demo:** `assets/day10_mouse_spring_demo.zip`  
🗒️ **Log:** `docs/DAY10_LOG.md`



🎮 Controls (current build)
Key	Action
Space	Pause / Resume
R	Reset
G	Toggle Gravity
← / →	Apply Horizontal Impulse
↑	Jump (if grounded)
LMB	Move nearest ball
RMB	Spawn a new ball
L	Cycle Levels (Ramp / Funnel)
H	Toggle Geometry Visibility
D       Toggle Debug HUD 
LMB     Drag (spring)
+ / -   Spring stiffness


📁 Key Files
File	Description
bouncing_ball.py	Day 1 – single-ball baseline
bouncing_balls.py	Multi-ball + levels + VFX + camera shake
assets/	All demos, screen recordings, and steps recorder ZIPs
docs/	Daily logs (DAY1–DAY7)
progress.md	Master progress tracker
🧠 About

This project merges physics simulation, animation design, and computational thinking — showcasing how real-world physics can be visualized interactively through Python (NumPy + Pygame).
Perfect for portfolios targeting VFX, Game Dev, or Computational Arts research.