\# 🎬 Day 7 – Impact Particles + Camera Shake



\## 🎯 Goal

To make the simulation \*feel alive\* by adding:

\- Spark particles that emit when balls collide.

\- Subtle camera shake proportional to collision intensity.



These effects simulate energy transfer during impacts, creating a cinematic VFX touch suitable for both \*\*CS simulation portfolios\*\* and \*\*real-time animation showcases\*\*.



---



\## 🧩 Key Additions



\### 1. \*\*Particle System\*\*

\- Each collision spawns multiple short-lived particles (tiny glowing sparks).

\- Particles have:

&nbsp; - Random direction and speed.

&nbsp; - Gravity influence.

&nbsp; - Gradual fade and motion blur.

\- Implemented via a new `Particle` class with update and draw functions.



\### 2. \*\*Camera Shake\*\*

\- Large impacts (determined by collision impulse) slightly move the camera’s view.

\- Added global variables `shake\_mag` and `shake\_angle` for dynamic offset.

\- Every frame applies a small random offset `(ox, oy)` to all drawing calls.



\### 3. \*\*Impact Integration\*\*

\- Ball–ball and ball–segment collisions now call:

&nbsp; ```python

&nbsp; add\_particles\_at(contact, color, impact\_strength)



