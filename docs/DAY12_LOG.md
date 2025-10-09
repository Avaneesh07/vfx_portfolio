\# Day 12 — Wind Field on Cloth



\### 🌬 Features Added

\- Constant +X wind field acting on every node  

\- Adjustable base wind (`A/D`) and on/off toggle (`W`)  

\- Optional sinusoidal turbulence per node with random phase offsets  

\- Frequency control for fluttering/billowing motion (` , / . `)  

\- Works alongside gravity, collisions, and cloth interaction  



\### 🎮 Controls

| Key | Action |

|-----|---------|

| \*\*W\*\* | Toggle wind on/off |

| \*\*A / D\*\* | Decrease / Increase base wind |

| \*\*S\*\* | Zero base wind |

| \*\*Z / X\*\* | Decrease / Increase turbulence amplitude |

| \*\*C\*\* | Toggle turbulence |

| \*\*, / .\*\* | Decrease / Increase turbulence frequency |

| \*\*LMB\*\* | Drag cloth |

| \*\*RMB\*\* | Spawn new ball |

| \*\*L\*\* | Switch level |



\### 🎥 Demo

Recorded with Windows Steps Recorder  

\*\*File:\*\* `assets/day12\_wind\_demo.zip`



\### 🧪 Testing Notes

1\. Enabled wind and adjusted speed using A/D  

2\. Toggled turbulence (C) and observed per-node variation  

3\. Adjusted frequency (, and .) to test flutter rate  

4\. Spawned a ball under wind to verify lift interaction  

5\. Switched levels (L) and ensured consistent response  



\### 💡 Observations

\- Turbulence phases differ per node for natural motion  

\- Combining gravity + wind gives realistic billow  

\- Smoothly integrates with existing collision system



