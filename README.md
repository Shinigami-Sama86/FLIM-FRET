# FLIM–FRET code packets (no-special deps)

Standalone scripts (NumPy/Pandas/Matplotlib/SciPy only) used in the thesis/paper analyses:

- Phasor plot + Δr + efficiency + distance conversion
- Amplitude-weighted mean arrival time (AWLT) maps
- FRET-positive fraction (lifetime-index thresholding)
- EMCCD spectra preprocessing + PCA (SVD implementation)
- IRF folding/alignment + reconvolution E stability vs fit-start (fixed and Δt-free)

## Install

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Confirmed project binary formats

### FLIM counts cubes (e.g. `green test flim 450nm flim bins 1`)
- 12-byte header: 3× big-endian int32: (nx, ny, nbins)
- payload: nx*ny*nbins samples stored as big-endian float64 (>f8)
- reshape to (ny, nx, nbins)

Examples found in this project:
- Donor-only: nx=16, ny=16, nbins=500
- Green+Orange: nx=11, ny=11, nbins=500

### Time-axis files (e.g. `mito green orange FLIM 450nm time 1`)
- 4-byte leading big-endian int32 n
- followed by n × big-endian int32 values in ps, typically edges (length nbins+1)
- for nbins=500: n=501 edges from 0 to 50000 ps in 100 ps steps

## Note on phasors for gated/windowed data
These scripts compute *windowed-relative phasors* (t=0 at first gate bin). Those points are
not expected to lie on the universal semicircle; the plots overlay a *relative mono-exponential locus*.
