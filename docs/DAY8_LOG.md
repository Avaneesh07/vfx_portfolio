\# Day 8 – Energy Visualization (Kinetic + Potential Energy)



\## Summary

Today we introduced \*\*real-time physics energy analysis\*\* into the bouncing-balls simulation.



Each ball now tracks:

\- \*\*Kinetic Energy (KE)\*\* = ½ m v²  

\- \*\*Potential Energy (PE)\*\* = m g h  

and the total energy = KE + PE.



At the bottom of the screen, stacked \*\*energy bars\*\* visualize how energy moves between motion and height.



\## What Happens

\- As a ball \*\*falls\*\*, PE ↓ and KE ↑ (shown by color-coded bars).  

\- When it \*\*bounces\*\*, KE drops and PE spikes up.  

\- The HUD displays live totals (ΣKE, ΣPE, ΣE) and updates every frame.



\## Controls

Same as before:  

`Space` pause   `R` reset   `G` gravity toggle   `←/→` impulse   `↑` jump   `RMB` spawn ball   `LMB` move nearest.



\## Files Added/Updated

\- `bouncing\_balls.py` → added energy computation \& bar visualization  

\- `docs/DAY8\_LOG.md` → this log entry  



\## Outcome

You can now see physics in action — a clear visual link between height, speed, and total energy.  

It’s a great base for the later \*\*profiling \& optimization\*\* tasks.



