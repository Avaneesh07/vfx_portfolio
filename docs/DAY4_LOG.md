\# Day 4 – Trails, Drag, and Smooth HUD



\## Features

\- \*\*Motion trails\*\* via an alpha-accumulated layer with per-frame fade.

\- \*\*Gentle air drag\*\* (`AIR\_DRAG = 0.15`) to smooth motion without killing energy too fast.

\- \*\*Smooth HUD FPS\*\* using exponential moving average.



\## Implementation Notes

\- Trails use a `pygame.SRCALPHA` surface and a semi-transparent fade rect each frame.

\- Balls are drawn onto the trail surface; trail is then composited onto the main screen.

\- `R` clears trails by filling the trail surface with full transparency.

\- EMA FPS: `fps\_ema = (1-a)\*fps\_ema + a\*inst\_fps` with `a = 0.12`.



\## Tunables

\- `TRAIL\_FADE\_ALPHA` (25..45 good range)

\- `AIR\_DRAG` (0.05..0.25 reasonable)



\## Validation

\- Trails visually stable at 120 FPS; collisions remain correct.

\- Drag + friction produce natural settling on ground.



