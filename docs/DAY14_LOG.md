\# Day 14 — Cloth Pinning + Save/Load Presets



\## What we built

\- \*\*Pin Edit Mode (`P`)\*\* — left-click any cloth node to pin/unpin.

\- \*\*Save pins (`Ctrl+S`)\*\* — writes a JSON preset to `assets/presets/cloth\_pins.json`.

\- \*\*Load pins (`Ctrl+L`)\*\* — restores pin layout instantly.

\- Works alongside wind (`A/←` and `D/→`), turbulence (`C`), drag-to-move, and ball collisions.



\## How to try

1\. Run: `python cloth\_sim.py`

2\. Press `P` to show pins. Click some nodes to toggle.

3\. Press `Ctrl+S` to save. Check the console for a success path.

4\. Press `Ctrl+L` to reload and verify.

5\. Add a ball with \*\*Right-Click\*\*, watch cloth react.

6\. Optional: Wind up/down (`A/←`, `D/→`), turbulence toggle (`C`).



\## Demo \& Files

\- \*\*Demo\*\*: `assets/day14\_cloth\_pins\_demo.zip` (Steps Recorder)

\- \*\*Preset\*\*: `assets/presets/cloth\_pins.json`

\- \*\*Code\*\*: `cloth\_sim.py`



\## Notes

\- Preset keeps only pin indices (not positions), so it’s resolution/mesh-layout aware.

\- If you change ROWS/COLS drastically, re-create pins and save again.



