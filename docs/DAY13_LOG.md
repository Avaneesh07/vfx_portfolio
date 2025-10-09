\# Day 13 – Cloth Tearing + Full Controls



\*\*What’s new\*\*

\- Restored all controls (wind base/turbulence, levels, HUD, geometry)

\- Cloth tearing via length threshold and Shift+Drag mouse tear

\- RMB toggles a bouncing ball that collides with the cloth

\- Fixed `TypeError` by keeping `t` as a float; clarified keybindings



\*\*Controls\*\*

\- Run: `python cloth\_sim.py`

\- Wind: `W` toggle, `A/←` down, `→` up, `S` zero

\- Turbulence: `Z/X` amp −/+, `,/.` freq −/+, `C` toggle turb

\- Tearing: `T` toggle, `\[` / `]` ratio −/+ (break sooner/later), `Y` repair all

\- Mouse: `LMB` drag nearest node, `Shift + LMB` tear nearby springs, `RMB` toggle ball

\- View: `B` springs vs grid, `H` geometry on/off, `D` dev HUD, `L` change level

\- System: `Space` pause, `R` reset cloth/grid



\*\*Demo\*\*

\- Steps Recorder ZIP: `assets/day13\_cloth\_tearing\_demo.zip`



\*\*Notes\*\*

\- Anchored top row keeps the cloth hanging

\- Tearing breaks individual springs; repair with `Y`

\- Ball and cloth collision is node–sphere, elastic along normal



