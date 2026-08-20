A Fourier Neural Operator (FNO) that learns the mapping
`(crosslinker ratio, ω) → (G'(ω), G''(ω))` from small-amplitude
oscillatory shear (SAOS) measurements on PDMS/Sylgard networks, and
serves as a continuous, differentiable digital twin of the network
across the composition-frequency plane.

### What it does

- **Learns from sparse sweeps.** Trains on a handful of frequency
  sweeps at discrete crosslinker ratios and interpolates continuously
  in both ratio and ω.
- **Respects the physics.** A soft loss enforces the terminal-region
  scalings G' ~ ω² and G'' ~ ω, plus Kramers-Kronig-motivated
  monotonicity of G'(ω).
- **Focuses spectral capacity.** Gated spectral convolutions apply a
  learned complex sigmoid to each retained Fourier mode, damping noisy
  modes and concentrating capacity on physically meaningful ones.
- **Extracts derived quantities.** Crossover frequency ω_c and modulus
  G_c, plateau modulus G_N⁰, and entanglement molecular weight
  M_e = ρRT/G_N⁰, in a single dataframe per run.
- **Produces a digital twin.** An ipywidgets slider sweeps ratio in
  real time, overlays the nearest experimental curve, and exports the
  predicted curve to Excel.
- **Generates a report.** Fan plot, tan δ sensitivity map, and physics
  table are compiled into a single PDF.

### Architecture

Two spectral convolution blocks with residual pointwise 1D convolutions,
lifted from a 2-dim input (ratio, log₁₀ω) through a width-64 channel
space and projected back to (log₁₀ G', log₁₀ G''). All quantities are
learned in log space to span the multi-decade dynamic range of the
moduli. Modes are truncated to match the frequency-grid Nyquist bound.

### Data

`Rheology_Data_PDMS_Sylgard.xlsx` holds one sheet per Sylgard 184
base:crosslinker ratio, with columns `Omega` (rad/s), `Gp` (Pa),
`Gpp` (Pa). Measurements are SAOS frequency sweeps in the linear
viscoelastic regime.

### Requirements

`torch`, `numpy`, `pandas`, `scipy`, `matplotlib`, `openpyxl`, `fpdf`,
`ipywidgets`. Runs on CPU or CUDA. The notebook is written for Google
Colab (Drive mount, `google.colab.files`); to run locally, replace the
`FILE_PATH` and remove the `files.download` call in the export cell.

### Usage

Open `FNO_rheology_clean.ipynb`, set the paths and toggles in the
config cell (`MODES`, `WIDTH`, `EPOCHS`, `USE_GATE`,
`USE_PHYSICS_LOSS`), and run top to bottom. Trained weights are
written to `pdms_fno_weights.pth` and physical properties to
`pdms_physics_results.csv`.
