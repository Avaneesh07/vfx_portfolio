\# Day 15 — Fast Picking (Spatial Hash) + Pins/Presets Integration



\*\*Goal:\*\* Make selecting/dragging cloth nodes fast even on large grids.



\## What I built

\- `spatial\_hash.py` uniform-grid for quick neighbor lookups.

\- Cloth picking \& pin toggling now query the hash (O(k) nearby) instead of scanning all nodes.

\- Integrated with Day 14 pin edit + JSON save/load.



\## Why this matters

\- 26×36 (936 nodes) remains snappy.

\- Dragging is smooth; saving/loading pin presets is instant.



\## Controls

LMB drag · Shift+LMB toggle pin · \*\*P\*\* pin nearest · \*\*U\*\* unpin all ·  

\*\*F5 / Ctrl+S\*\* save pins → `assets/presets/cloth\_pins.json` · \*\*F9 / Ctrl+O\*\* load ·  

\*\*Space\*\* pause · \*\*R\*\* reset · \*\*W/A/S/←/→\*\* wind · \*\*B/H/D\*\* view/HUD



\## Implementation Notes

\- `SpatialHash(cell\_size=48)` stores node AABBs; rebuilt each frame.

\- `rebuild\_pick\_hash(pos, pick\_hash, PICK\_RADIUS)` after physics.

\- `nearest\_node\_from\_hash(mx, my, ...)` picks the true nearest of local candidates.



\## Performance

\- Picking is effectively constant-time per frame and scales with local density.



\## Next Steps

\- Day 16: add \*\*shear\*\* and \*\*bending\*\* springs with live stiffness controls.



