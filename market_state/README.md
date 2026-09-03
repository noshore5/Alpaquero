# market_state

A causal, multivariate **financial-market state model**: aligned multi-asset
prices → Morlet CWT → complex wavelet coherence → complex **Hermitian market
graph** → eigendecomposition → stable spectral features → **Mamba-3** sequence →
per-horizon prediction heads → walk-forward backtesting → inference benchmark.

This is a **research benchmark**. The model is currently benchmarked with a
**randomly initialised** (untrained) model; training is out of scope. Causality
is a hard requirement: features at `t` use signal bars `≤ t` only, and targets
use `t+1 .. t+H`.

---

## Pipeline

```
aligned log-returns [A, T]                (data.alignment, data.preprocessing)
        │  causal Morlet CWT (transform_causal)
        ▼
coeffs [A, F, T_valid]                    (transforms.cwt)
        │  complex wavelet coherence, causal time-smoothing
        ▼
complex coherence [P, F, T_valid]         (transforms.wavelet_coherence)
        │  Hermitian market graph H(t)
        ▼
H(t) [T, N, N]                            (transforms.hermitian)
        │  eigendecomposition → stable spectral quantities
        ▼
feature rows [T_valid, D]                 (transforms.spectral / pipeline)
        │  Mamba-3 over causal windows
        ▼
latent market state z_t → prediction heads (models.market_state)

heads: return_H, realized_vol_H, max_drawdown, correlation, regime
        │
        ▼
walk-forward backtest + per-fold metrics   (backtest.*)
```

- **`data/`** — Alpaca historical bar download (equities + crypto), explicit
  UTC alignment onto a common grid with coverage-gating (gaps stay NaN, never
  forward-filled), leakage-safe log-return representation.
- **`transforms/`** — the deterministic feature stack. All downstream steps run
  on causal coefficients, so coherence, the Hermitian graph and the spectral
  features at `tau` use only bars `≤ tau`.
- **`models/`** — complex-diagonal Mamba-3 sequence (`Mamba3Sequence`) plus the
  per-horizon heads. The model never touches raw prices or the Hermitian
  matrix; causality is enforced entirely upstream.
- **`datasets/`** — target engine (realized vol, forward return, max drawdown,
  correlation matrix, regime classes), causal windowing, walk-forward splits.
- **`backtest/`** — cost-aware (turnover × bps) portfolio simulation, per-fold
  metrics, cross-fold pooling.
- **`inference/`** — streaming causal realtime state over a rolling buffer.
- **`benchmarks/`** — CPU + MPS inference latency / throughput.

## Causality (critical)

The standard FFT/Morlet CWT is **acausal**: `W(t, P)` samples `[t - L_f, t + L_f]`,
"seeing" up to the longest period ahead. For a 1-day period that would leak
enormously into `t+1..t+H`. Handling is done at the **input level**:

- The input is right-zero-padded by `l_max` bars so no future support lands on
  real signal.
- The causal coefficient assigned to real time `tau` is `W(tau - L_f, f)` — the
  wavelet whose trailing edge touches `tau`.
- `L_f = coi_factor · fb · P_f` (single knob, `config.wavelet.coi_factor`).
- The first `l_max` output rows are the causal warm-up and are dropped, and all
  targets / one-bar returns are sliced by `[dropped : dropped + T_out]` to stay
  row-aligned with the features.

> **Note on the Morlet tail.** The Morlet envelope is Gaussian with nominally
> unbounded support, so *no* finite `coi_factor` gives exactly zero leakage —
> it only makes it negligible. Default `coi_factor=3.0` (three e-folding
> half-widths) matches the project spec but leaves a small (~1e-3) residual
> sensitivity in the far tail. Raise it (e.g. `6.0`) for strictly-negligible
> leakage at the cost of a longer warm-up (`l_max`).

Additional leakage guards:
- Coherence **time-smoothing is causal** (one-sided, past-only kernel), so a
  perturbation of future samples never changes the coherence at `tau`.
- Eigenvectors are phase-canonicalised and the model consumes *stable spectral
  quantities* (sorted/normalised eigenvalues, spectral entropy, concentration,
  and gauge-invariant projector loadings when enabled). Raw eigenvector columns
  are off by default (`spectral.use_eigenvectors: false`) due to phase/sign
  instability near eigenvalue crossings.
- **IC metric** is the cross-sectional rank IC (per-timestep Spearman across
  assets, averaged); a pooled ravel would produce spurious high IC with a
  random model.

## Setup

```bash
python -m venv .venv_market
../.venv_market/bin/pip install -r requirements.txt        # from market_state/
```

Alpaca credentials come from `APCA_API_KEY_ID` / `APCA_API_SECRET_KEY` (fallback
`ALPACA_API_KEY` / `ALPACA_SECRET_KEY`), pulled from the environment or a leading
`.env`. Never hardcode keys.

## Usage

Run from the `market_state/` directory.

```bash
# 1. Download historical bars -> data/raw
../.venv_market/bin/python scripts/download.py --config configs/default.yaml

# 2. Build aligned features + targets -> data/features
../.venv_market/bin/python scripts/build_features.py --config configs/default.yaml

# 3. Walk-forward backtest (random-model benchmark)
../.venv_market/bin/python scripts/run_backtest.py --config configs/default.yaml

# 4. Inference latency benchmark (cpu + mps)
../.venv_market/bin/python scripts/benchmark.py --config configs/default.yaml
```

### Crypto universe (Binance public data, no API key)

The free Alpaca IEX feed cannot serve the cross-asset equity universe. Crypto
has no such restriction. `configs/crypto.yaml` selects 30 liquid Binance USDT
pairs and routes `download.py` to `src/data/binance.py` (public `/api/v3/klines`,
24/7, no session gaps). The model is symbol-agnostic — only the universe and a
few numeric knobs differ from `default.yaml`.

```bash
../.venv_market/bin/python scripts/download.py       --config configs/crypto.yaml
../.venv_market/bin/python scripts/build_features.py  --config configs/crypto.yaml

# random-model leakage/plumbing benchmark (IC ~ 0)
../.venv_market/bin/python scripts/run_backtest.py    --config configs/crypto.yaml

# actual walk-forward training: per-fold z-score normalization fit on train
# rows only, early stopping on validation cross-sectional IC of the traded
# return_H head, per-fold checkpoints + report.json under --out
../.venv_market/bin/python scripts/train.py --config configs/crypto.yaml \
    --out runs/crypto_01 --max-epochs 40 --patience 6 --device cpu   # --device cuda on a GPU box
```

Runs are deterministic given (config, data, seed): `reproducibility.seed` is
applied via `src/repro.py::set_seed` in every entry-point script.

Run tests:

```bash
../.venv_market/bin/python -m pytest
```

## Configuration

All knobs live in `configs/default.yaml`: universe (`data.symbols`), timeframe,
lookback (`window.bars`), CWT periods / `nfreqs` / `coi_factor` / `causal`,
coherence smoothing + frequency reduction, spectral `n_components` /
`use_eigenvectors`, model size, target horizons / regime thresholds, backtest
cost / fold sizing, and inference device.

## Known limitations

- **Live Alpaca data access is limited** by the account (stock & crypto bar
  requests returned empty without an exception). Use `configs/crypto.yaml`
  (Binance public data) for the current working data path; broad equity
  coverage would need a paid Alpaca plan or another bar provider.
- `scripts/run_backtest.py` (default `--epochs 0`) is a latent-dynamics
  benchmark on a **randomly initialised** Mamba-3. Actual training is
  `scripts/train.py`; it has per-fold normalization + early stopping but no LR
  schedule or multi-seed ensembling yet.
- The spectral eigendecomposition falls back to CPU on the MPS device (PyTorch
  does not implement `torch.linalg.eigh` on MPS), which raises MPS feature-
  pipeline latency relative to CPU for the current per-step full-recompute mode.
