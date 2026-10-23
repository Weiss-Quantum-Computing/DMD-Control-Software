# DMD-Control-Software
Control software 

Python 3 scripts for an ALP-4.3 DMD (via ALP4lib) and a Basler camera (via pypylon).

| Script | What it does |
| --- | --- |
| `beamtracking.py` | Finds the beam spots in a camera image (trackpy) and returns one intensity per beam (`getIntenList(img)`) |
| `DMDcalib.py` | Builds the beam footprints and equalises the beams by switching pixels off; saves `calibSave.npy`, `intenSave.npy`, `gramSave.npy` |
| `imageGen.py` | Builds an image sequence from `calibSave.npy` and saves it as `img.npy` (uint8, 0/255) |
| `DMDdisplay.py` | Shows `img.npy` on the DMD |
| `DMDautoScan.py` | Shows the images of `img.npy` one at a time and reports the one with the brightest camera image |
| `cameraaccess.py` | Grabs one camera image and saves it as a PNG |

Usual order: `DMDcalib.py`, then `imageGen.py`, then `DMDdisplay.py` (or `DMDautoScan.py` for a bar scan).
All files are read and written in the current directory.

Requirements: numpy, scipy, pandas, matplotlib, pillow, trackpy, pypylon, ALP4lib (no requirements file).

`imageGen.oneperrow()` and `imageGen.twobeams()` build far more images than memory or the DMD can hold for a 7x7 array; only use them for small arrays.
