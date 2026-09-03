# PROJECT CONTEXT — Alpaquero

Comprehensive project context. The repo now has two generations of code; the
**market_state** subproject is the current, active research work. The root
legacy "Alpaquero" trading-bot scaffolding predates it and has been largely
shelved.

Last updated: 2026-09-03

---

## 1. Executive summary

- **Repo**: `Alpaquero` (git, branch `main`). Remote: `origin` =
  `https://github.com/noshore5/Alpaquero`.
- **Active project**: `market_state/` — a **causal, multivariate
  financial-market state model**:
  aligned multi-asset prices → Morlet CWT → complex wavelet coherence → complex
  **Hermitian market graph** → eigendecomposition → stable spectral features →
  **Mamba-3** sequence → per-horizon prediction heads → walk-forward
  backtesting → inference benchmark.
- **Status**: feature/transform/model/backtest/**training**/inference/benchmark
  stacks implemented and verified; 14 unit tests pass; CLI scripts work
  end-to-end on synthetic data. Live-data blocker addressed via a **Binance
  public-API** crypto downloader (`configs/crypto.yaml`); the training loop
  works after a complex-pscan backward fix. **Not yet run on real data.**
- **Hard requirement**: strict causality — features at `t` use signal bars
  `≤ t`; targets use `t+1..t+H`. Never leak future information.

---

## 2. Repository layout

```
Alpaquero/
├── CONTEXT.md              # THIS FILE — project context
├── README.md / QUICKSTART.md   # legacy Alpaca trading-bot docs (mostly stale)
├── main.py                 # legacy bot entry point (stale)
├── setup.py / setup_easy.py / gitpush.bat   # legacy setup scripts (stale)
├── config/                 # legacy config.yaml + .env.template (stale)
├── src/                    # legacy bot: config/settings, trading/trader,
│                           #   trading/risk_manager, utils/logger (shelved)
├── tests/                  # legacy bot tests (shelved)
├── alpaca_facts.txt        # notes on Alpaca data/limits
├── .env / .env.template    # Alpaca credentials (template in git; .env ignored)
│
└── market_state/           # ACTIVE research project (see §3)
    ├── README.md           # full pipeline/usage docs
    ├── pyproject.toml / requirements.txt
    ├── configs/default.yaml
    ├── scripts/
    │   ├── download.py     # fetch bars -> data/raw (provider: alpaca | binance)
    │   ├── build_features.py  # raw -> aligned features+targets -> data/features
    │   ├── run_backtest.py    # walk-forward RANDOM-model benchmark
    │   ├── train.py           # walk-forward training (norm + early stop + ckpts)
    │   └── benchmark.py       # CPU + MPS inference latency
    ├── src/
    │   ├── config.py           # YAML load + validation + hashing
    │   ├── data/               # alpaca, binance, alignment, preprocessing, symbols
│   ├── repro.py            # set_seed (wired into all entry-point scripts)
    │   ├── transforms/         # cwt, wavelet_coherence, hermitian, spectral, pipeline
    │   ├── models/             # mamba (Mamba3), heads, market_state
    │   ├── datasets/           # targets, windows, splits
    │   ├── backtest/           # metrics, portfolio, walk_forward, run
    │   ├── inference/          # realtime streaming
    │   └── benchmarks/         # benchmark_inference
    └── tests/                  # causality, transforms, metrics
        └── conftest.py         # puts src/ on sys.path
```

The old strategy scaffolding (`backtest.py`, `src/strategies/*`,
`tests/test_strategies.py`) was removed when the project pivoted to
`market_state/`.

---

## 3. Environment & tooling

- **Working dir**: `/Users/noahshore/Documents/Projects/Alpaquero`
- **Python**: venv at `market_state/.venv_market` (Python 3.11). Invoke from
  inside `market_state/` as `../.venv_market/bin/python`.
- **Installed**: torch 2.13 (CUDA off, **MPS on**), numpy, scipy, pandas,
  pyarrow, PyYAML, scikit-learn, matplotlib, python-dotenv, mambapy
  (`mambapy.pscan` works), alpaca-py, pytest.
- **Alpaca creds**: env vars `APCA_API_KEY_ID`/`APCA_API_SECRET_KEY` (fallback
  `ALPACA_API_KEY`/`ALPACA_SECRET_KEY`). `.env` at root uses the `ALPACA_*`
  names. **Never hardcode secrets.** `.env` is gitignored.
- **Package layout**: `src/` is on `sys.path`; the packages `data`,
  `transforms`, `models`, `datasets`, `backtest`, `inference`, `benchmarks` are
  **top-level** in `src` (not a namespace package). Intra-src imports between
  sibling packages use **absolute** imports (`from datasets.splits import ...`);
  only same-package imports use relative `.`.

---

## 4. Git state

- HEAD: `bf5dfc8` "market_state: causal spectral Mamba-3 market-state research
  project". The prior HEAD `c874642` (2025-08-13, legacy) was **amended away**
  and `origin/main` **force-pushed** to the new head.
- `.gitignore` additions: `.venv_market/` and `market_state/data/` (runtime
  download/feature outputs) are ignored.
- **Uncommitted (as of last update)**:
  - `market_state/src/backtest/metrics.py` — IC-metric fix (see §7)
  - `market_state/tests/test_metrics.py` — new regression tests
  - `market_state/README.md` — new
  - `CONTEXT.md` — this file
  These should be committed (amend `bf5dfc8` or a new commit).

---

## 5. Project pipeline & key files

```
aligned log-returns [A, T]
   → causal Morlet CWT                     (transforms/cwt.py: transform_causal)
   → complex wavelet coherence (causal)    (transforms/wavelet_coherence.py)
   → Hermitian market graph H(t)           (transforms/hermitian.py)
   → eigendecomposition → spectral feats   (transforms/spectral.py)
   → feature rows [T, D]                   (transforms/pipeline.py)
   → Mamba-3 over causal windows           (models/mamba.py, market_state.py)
   → latent state z_t → heads              (models/heads.py)
   → walk-forward backtest + metrics       (backtest/*)
```

Heads (keyed by name + horizon, e.g. `return_5`, `realized_vol_78`):
`return_H`, `realized_vol_H`, `max_drawdown`, `correlation`, `regime`.

### Causality contract (critical)
- The standard FFT/Morlet CWT is acausal (`W(t,P)` samples `[t−L_f, t+L_f]`).
- Fix is at the **input level** in `transform_causal`: right-zero-pad input by
  `l_max`; the causal coefficient at real time `tau` is `W(tau − L_f, f)` (the
  wavelet whose trailing edge touches `tau`). `L_f = ceil(coi_factor · fb ·
  P_f)`.
- First `l_max` output rows are causal **warm-up** and are dropped. **All
  targets and one-bar returns are sliced by `[dropped : dropped + T_out]`** to
  stay row-aligned with the feature rows.
- Coherence **time-smoothing is causal** (one-sided past-only kernel) — a
  future perturbation must not change coherence at `tau`.
- `coi_factor` (default 3.0) is the single knob shared by padding, trailing
  support, and COI trimming.

> **Morlet-tail caveat**: the Morlet envelope is Gaussian with unbounded
> support, so no finite `coi_factor` gives exactly zero leakage — only
> negligible. Default `3.0` leaves ~1e-3 residual; raise to `~6.0` for
> negligible leakage (longer warm-up). Documented in README.

### End-to-end causal path verified
- Causal CWT index math: `out[u] = acausal[u + (n_drop − lf)]` (explicit slice,
  no wrapper/roll) — after replacing a `torch.roll` that wrapped columns.
- Causal coherence smoothing (replaces symmetric Gaussian which leaked future
  neighbors into `tau`).
- Spectral MPS fallback: `torch.linalg.eigh` not implemented on MPS; moved to
  CPU for decomposition, results moved back; `_sanitize_complex` handles
  NaN/Inf without MPS-unsupported `nan_to_num` on complex.

---

## 6. Default config (`market_state/configs/default.yaml`)

- timeframe `5Min`; lookback `window.bars: 78` (= 1 trading day).
- wavelet periods `["15min","30min","1h","2h","4h","8h","1d"]`, `nfreqs: 32`,
  `causal: true`, `coi_factor: 3.0`.
- coherence `smooth_time_steps: 5`, frequency reduction `magnitude_weighted`.
- spectral `n_components: 8`, `use_eigenvectors: false` (stable spectral
  quantities only; raw eigenvector columns phase-unstable near crossings).
- model `state_dim: 64`, Mamba-3 (1 layer default).
- targets horizons `[5, 12, 78]`; regime thresholds `[0.005, 0.02]`.
- backtest cost 1.0 bps; train 20000 / validate 5000 / test 2000 / step 5000,
  expanding window.
- seed 42 (defined in config but **not yet wired** into model init / pipeline).

---

## 7. History of bugs found & fixed (important)

1. **IC≈0.95 spurious artifact (HIGH)** — `backtest/metrics.py::ic()` raveled
   the pooled `[B, A]` pred/target into one vector and computed a single
   Spearman. Between-asset level offsets (random model biases; always-positive
   realized_vol/max_drawdown targets) produced huge false IC. Raw per-asset
   Pearson was −0.009 but pooled Spearman read 0.45–0.95; pure random data gave
   0.59. **Fix**: cross-sectional rank IC — per-timestep Spearman across assets,
   averaged over time; matches the cross-sectional long/short portfolio. Now a
   random model gives IC ≈ 0 (±0.02). Regression tests in
   `tests/test_metrics.py`. **This is the reason benchmark numbers were
   previously untrustworthy.**
2. **Causality leak in `transform_causal`** — `torch.roll(shifts=−lf)` wrapped
   columns around, allowing future samples to leak into `tau`. Replaced with
   explicit slicing (`out[u] = acausal[u + (n_drop − lf)]`). Coefficient-level
   future-perturbation test now passes.
3. **Symmetric coherence smoothing leaked future** — `_smooth_time` convolved
   over `tau±k`, pulling future-based coherence into `tau`. Made causal
   (one-sided past-only kernel). Verified by future-perturbation tests.
4. **MPS eigendecomposition unsupported** — `torch.linalg.eigh`/`nan_to_num`
   complex not on MPS. Added CPU fallback for the decomposition + real/imag
   `_sanitize_complex`.
5. **`alignment.build_timeline` tz bug** — compared tz-naive vs tz-aware
   timestamps (raw parquet can strip UTC). Normalise all symbol timestamps to
   UTC-aware before min/max.
6. **Sibling-package imports** — `backtest/`, `inference/`, `benchmarks/` use
   absolute imports (`from transforms...`, `from datasets...`), not relative
   `..`, because they are top-level packages under `src`.
7. **Complex pscan backward was wrong (blocked ALL training)** —
   `models/mamba.py::_ComplexPScan.backward` (mambapy/Blelloch branch) sliced
   the `d_inner` axis where it meant the time axis; `loss.backward()` raised
   `_ComplexPScanBackward returned an invalid gradient ... [.,.,64,16] vs
   [.,.,128,16]` for any `expand>1`. Fixed to index time on axis 1; the fast
   path now matches the sequential reference for fwd/gradA/gradX to machine
   precision. `--epochs > 0` runs end-to-end.
8. **`.gitignore` `data/` was unanchored** — matched `market_state/src/data/`
   too, so new files in the source package (e.g. `data/binance.py`) were
   silently untracked. Anchored to `/data/` + `market_state/data/` with a
   `!market_state/src/data/**` negation.

---

## 8. Tests & verification

Run with `../.venv_market/bin/python -m pytest` from `market_state/`.
- `tests/test_causality.py` — coefficient & feature-level future/past
  perturbation separation (validates no-leak causality construction).
- `tests/test_transforms.py` — determinism, finiteness, Hermiticity error,
  model forward shapes.
- `tests/test_metrics.py` — IC near-zero on random/noised model, near-perfect
  on monotone match, 1-D support, NaN handling.
- `tests/test_mamba.py` — complex pscan fast-path vs sequential reference
  (fwd + both grads, several seq lengths); one end-to-end backward step
  through a MarketStateModel.
- **14 tests pass.** Verified end-to-end synthetic runs for both
  `configs/crypto.yaml` (binance provider path) and training via
  `scripts/train.py` (checkpoints + report.json written, IC ≈ 0 on
  no-signal synthetic data, as expected).

---

## 9. CLI usage (from `market_state/`)

```bash
# 1. Download historical bars -> data/raw
#    (Alpaca equities/crypto by default; crypto.yaml routes to Binance public API)
../.venv_market/bin/python scripts/download.py --config configs/crypto.yaml
# 2. Build aligned features + targets -> data/features
../.venv_market/bin/python scripts/build_features.py --config configs/crypto.yaml
# 3a. Walk-forward RANDOM-model benchmark (leakage/plumbing check, IC ~ 0)
../.venv_market/bin/python scripts/run_backtest.py --config configs/crypto.yaml
# 3b. Walk-forward TRAINING (per-fold norm + early stop + checkpoints)
../.venv_market/bin/python scripts/train.py --config configs/crypto.yaml \
    --out runs/crypto_01 --max-epochs 40 --patience 6 --device cpu
# 4. Inference latency benchmark (cpu + mps)
../.venv_market/bin/python scripts/benchmark.py --config configs/default.yaml
```

`build_features.py` performs alignment + causal-warmup trimming + target
slicing (`[dropped : dropped + T_out]`) so feature/target rows stay aligned.

---

## 10. Blockers / next steps

**Live data access (blocker, unresolved).** Live Alpaca calls returned **empty
bars** with no exception for both stocks and crypto — a data-access limit, not
a bug. Research shows the cause is likely the Alpaca **free Basic (IEX)** feed:
IEX historical data is restricted to updates every 15 minutes, and the `end`
parameter must be **at least 15 minutes in the past** to query without a paid
subscription. Our downloader already defaults to `stock_feed="iex"`, so for
**historical research windows comfortably in the past** (e.g. 2020–2023) the
free IEX stock feed may work without any paid plan. **Action**: re-test
`download.py` with a historical window well in the past (and/or relax/validate
that `end` is ≥ 15 min before now).

**Alpaca data facts (research recaps):**
- Free (Basic) plan: equities real-time **IEX only** (~2.5% of US volume);
  historical since 2016; **15-min delayed / 15-min-old query restriction**;
  200 req/min. Crypto is a separate feed and not subject to these restrictions
  (hence `BTC/USD`, `ETH/USD` were the only usable live sources earlier).
- Paid Algo Trader Plus ($99/mo): full **SIP** consolidated tape (100% US
  volume), removes the 15-min restriction, 10,000 req/min.

**Crypto data path (new, unblocks live data).** `src/data/binance.py`
(`BinanceDownloader`) pulls OHLCV from Binance's public `/api/v3/klines`
(no API key, 24/7, no 15-min restriction) into `data/raw/crypto/<PAIR>.parquet`,
matching the Alpaca output schema. `scripts/download.py` now routes on
`data.provider` (`alpaca` | `binance`) or `--provider`. `configs/crypto.yaml`
= 30 liquid Binance USDT spot pairs, 2021–2024, `window.bars: 288` (24h),
lifted regime thresholds, bigger folds. The model/transforms/targets are
unchanged — the pipeline only ever consumed aligned multi-asset log-returns.
Verified end-to-end on synthetic data (build_features → backtest → `--epochs`).

**Training entry point (new).** `scripts/train.py` runs the walk-forward with
real training hygiene:
- per-fold z-score normalization fit on **train rows only**
  (`WalkForwardRunner._fit_normalization`), applied to train/val/test windows;
- early stopping on **validation cross-sectional IC of the traded `return_H`
  head** (`TrainingLoop.fit_early_stop`), patience-based, best weights restored;
- deterministic via `src/repro.py::set_seed` (fold uses `seed + fold_idx`),
  wired into build_features / train / run_backtest;
- per-fold checkpoints (`fold_XXX.pt` = weights + norm stats + model cfg) and
  `report.json` under `--out`; prints a per-fold IC t-stat.
`run_backtest.py --epochs 0` stays the random-init leakage benchmark.
Still missing: LR schedule, multi-seed ensembling, richer eval (Sharpe of the
equity curve is computed as `pf_*` pooled metrics but not annualised).

`configs/crypto.yaml` also widens the deterministic feature vector
(`spectral.n_components: 12`, `use_eigenvectors: true`) → `d_spec = 70`
(was 28), which only widens the Mamba `in_proj` (~1% of params).

**Next steps (prioritised):**
1. Commit uncommitted changes (see Git state §4 — now also: mamba backward fix,
   binance provider, crypto.yaml, train.py, repro.py, walk_forward
   normalization/early-stop, .gitignore anchor, test_mamba).
2. Run the real crypto pipeline on a GPU box:
   `download.py --config configs/crypto.yaml` → `build_features.py` →
   `train.py --config configs/crypto.yaml --out runs/crypto_01 --device cuda`.
3. Iterate on features / universe size (30 → ~50 pairs is the highest-leverage
   change) rather than lengthening history.
4. Re-test Alpaca on a 2021-era historical window (free IEX may serve old
   equity data) to optionally add an equity universe back.
5. (Optional) Wire `inference/realtime.py` to a live crypto feed.
