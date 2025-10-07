\# Visual \& Interactivity Upgrades



\## What I added

\- \*\*Per-ball radii\*\*: each ball has its own size; boundaries and collisions respect radius.

\- \*\*Impact flash\*\*: brief color brighten on collision (visual feedback).

\- \*\*Right-click spawn\*\*: add a new ball at the mouse with random size/color.



\## Technical notes

\- Collision uses positional correction + elastic impulse with restitution `e=0.80`.

\- Ground detection \& friction use per-ball radii: `pos\_y + radius ≈ floor`.

\- Flash timers decay each frame; brightness scales with remaining time.

\- Spawn safely appends arrays (pos, vel, radii, flash) and color list.



\## Validation

\- Verified collisions separate smoothly and conserve direction realistically.

\- Flash occurs only on approaching collisions (`v\_rel · n < 0`).

\- Stable at ~120 FPS with ~15–25 balls on current settings.



\## Next

: subtle trails, gentle air drag, nicer HUD (averaged FPS / counts).



