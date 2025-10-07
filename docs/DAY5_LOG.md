\# Day 5 – Mass-aware Collisions



\## What changed

\- Added per-ball \*\*mass\*\* (`m ∝ r^2`) and \*\*inv\_masses\*\* for efficiency.

\- \*\*Mass-weighted positional correction\*\* (heavier moves less).

\- \*\*General collision impulse\*\* for unequal masses:

&nbsp; \\\[

&nbsp;   j = -\\frac{(1+e)(v\_{rel}\\cdot \\hat{n})}{\\frac{1}{m\_1} + \\frac{1}{m\_2}}

&nbsp; \\]

&nbsp; with velocity updates:

&nbsp; \\\[

&nbsp;   v\_1' = v\_1 - \\frac{j}{m\_1}\\hat{n}, \\quad v\_2' = v\_2 + \\frac{j}{m\_2}\\hat{n}

&nbsp; \\]

\- HUD shows \*\*Σp\*\* (sum of momentum) as a sanity check.



\## Notes

\- Vertical momentum changes under gravity; wall/floor collisions add impulses.

\- During ball–ball impacts (in-flight), total momentum is nearly conserved (up to drag \& numerical error).

\- Trails, flashes, RMB spawn, and smooth HUD from Day 4 remain intact.



\## Tunables

\- `DENSITY` (mass scaling), `REST\_COEFF` (bounciness), `AIR\_DRAG`, `TRAIL\_FADE\_ALPHA`.



