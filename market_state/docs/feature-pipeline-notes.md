# Feature-pipeline tuning notes (2026-09-10)

Investigation of what the Mamba-3 actually consumes, done by plotting every
stage of the pipeline on one window / all 30 crypto assets
(`scripts/demo_pipeline.py` → `docs/pipeline_stages*.png`).

## What was wrong

The `[T, d_spec]` spectral feature matrix was **near-constant in time** — one
slowly-varying scalar (λ₁ / spectral concentration) and ~69 effectively dead
dimensions. A model cannot learn from that. Three causes, in order of impact:

1. **`nfreqs` was dead config.** `build_features.py` built the CWT bank from
   the 7 named period *labels* only (`financial_periods` ignored `nfreqs`), so
   the pipeline ran a 7-scale CWT → 7-bin coherence. Fixed: `financial_periods`
   now expands to a dense log-spaced grid of `nfreqs` periods.
2. **Slow bands carried no coherence dynamics.** Above ~2h period every pair
   sits at |C|≈1, so those scales only flattened the market graph and pinned
   λ₁. Fixed: band cut to `["15min", "2h"]` (3–24 bar), `nfreqs: 24`
   (~8 scales/oct; the Morlet's ~8% fractional bandwidth makes more redundant).
3. **`diagonal: "one"` + no market-factor removal.** H_ii=1 made the identity
   term dominate every eigendecomposition; and with no beta removal the graph
   was always "everything tracks BTC" → λ₁ huge and constant. Fixed:
   `coherence.diagonal: zero` and `data.cross_sectional_demean: true`
   (subtract per-bar equal-weight market return before the CWT).

After 1–3 the feature matrix has real time structure across ~all 70 dims and
λ₁…λ₄ are distinct and moving (`docs/pipeline_stages_fixed.png`).

## Locked config (`configs/crypto.yaml`)

| key | value | why |
|---|---|---|
| `window.bars` | 1152 (4d) | multi-day memory now comes only from the Mamba window, not the wavelet bank |
| `wavelet.periods` | `["15min","2h"]` | scale-span endpoints (3–24 bar) |
| `wavelet.nfreqs` | 24 | dense grid resolution |
| `coherence.smooth_time_steps` | 32 | at 5 the coherence *magnitude* saturated at ~1; 32 makes \|C\| a live channel |
| `coherence.diagonal` | `zero` | unpin the eigenspectrum |
| `data.cross_sectional_demean` | `true` | graph = residual co-movement, not beta |
| `spectral.n_components` | 12 | d_spec = 3·12 + 4 + 30 = 70 |
| `backtest.batch_size` | 8 | complex parallel scan over W=1152 is the memory bottleneck |

## Build memory

`nfreqs: 24` (was effectively 7) makes the per-chunk coherence tensor
`[435, 24, chunk+lead]` ~3x bigger. On a 16 GB Mac the default
`--chunk-rows 16000` thrashes swap (process pins at ~0% CPU). Use
**`--chunk-rows 4000`** locally — ~8 min, CPU-bound, RSS ~6 GB. A GPU box
with more RAM can use the default.

## OPEN — fix #3: eigenvector phase is discarded

`SpectralDecomposition.features()` keeps only **gauge-invariant `|u_m|²`**
node loadings. The lead-lag / rotation information in the market graph lives in
the **relative phases of the eigenvector components** (`canonicalize_phase`
already fixes the global gauge, so relative phases are well-defined and usable).

Right now that channel is thrown away entirely. Adding per-mode eigenvector
phase features (e.g. `angle(u_m[j] · conj(anchor))` for the top few modes, or a
few complex-bilinear invariants) is expected to be a real improvement.

**Deferred** because it changes `d_spec` and therefore the model input dim —
do it as a second experiment once there is a baseline `return_5_ic` number
from the current config. Mode-crossing in the low eigenvalues (λ₅+ spiky
zero-crossings in `docs/pipeline_stages_fixed.png` panel 6) should be handled
at the same time — either eigenvalue continuity-tracking or drop
`n_components` to ~8 to cut the noisy tail.
