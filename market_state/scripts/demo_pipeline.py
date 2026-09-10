"""End-to-end feature pipeline on one window, every stage, all 30 crypto assets.

    aligned prices  ->  log-returns  ->  causal Morlet CWT  ->  complex wavelet
    coherence  ->  Hermitian market graph H(t)  ->  eigendecomposition  ->
    [T, d_spec] spectral feature matrix  (what the Mamba actually consumes)

Config-faithful: reads configs/crypto.yaml (nfreqs, smooth_time_steps,
n_components, coi_factor, ...). Illustration only -- the production build runs
this continuously over the whole series via FeaturePipeline.compute_chunked.

    cd market_state
    ../.venv_market/bin/python scripts/demo_pipeline.py docs/pipeline_stages.png
"""
import sys
import numpy as np, pandas as pd, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, "src")
from config import load_config
from transforms.cwt import period_to_bars, financial_periods, MorletCWTBank
from transforms.wavelet_coherence import WaveletCoherence
from transforms.hermitian import HermitianGraph
from transforms.spectral import SpectralDecomposition, SpectralConfig

CFG = load_config("configs/crypto.yaml")
TF = CFG["data"]["timeframe"]
w, coh_cfg, sp_cfg = CFG["wavelet"], CFG["coherence"], CFG["spectral"]
N_SHOW = min(CFG["window"]["bars"], 1152)
WIN_END = "2023-06-15 00:00:00+00:00"
SYMS = [s.replace("/", "_") for s in CFG["data"]["symbols"]]

periods = financial_periods(w["periods"], TF, w.get("nfreqs"))
KEEP_FRAC = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0   # keep the FASTEST fraction of the grid
if KEEP_FRAC < 1.0:
    periods = periods[: max(2, int(round(len(periods) * KEEP_FRAC)))]
    print(f"cut slowest {(1-KEEP_FRAC)*100:.0f}% of scales -> {len(periods)} kept, "
          f"{periods[0]:.0f}..{periods[-1]:.0f} bar")
F = len(periods)
l_max = int(np.ceil(w["coi_factor"] * 2.0 * periods.max()))
# enough real signal behind the shown window that its low-freq bands are clean;
# a bit less than the pipeline's full pad+l_max lead (illustration, not bit-exact)
n_time = l_max + N_SHOW + 640
print(f"{len(SYMS)} assets | {F} scales {periods[0]:.0f}-{periods[-1]:.0f} bar | "
      f"l_max {l_max} | window {N_SHOW} | n_time {n_time}")


def load_close(sym):
    d = pd.read_parquet(f"data/raw/crypto/{sym}.parquet")[["timestamp", "close"]]
    return d.set_index("timestamp")["close"].sort_index()


series = {s: load_close(s) for s in SYMS}
idx = None
for s in series.values():
    idx = s.index if idx is None else idx.intersection(s.index)
end_pos = idx.get_indexer([pd.Timestamp(WIN_END)], method="nearest")[0]
sl = slice(end_pos - n_time, end_pos)
ts = idx[sl]
P = np.stack([series[s].reindex(idx).values[sl] for s in SYMS])          # [A, n_time]
A = P.shape[0]

logret = np.zeros_like(P, dtype=np.float32)
logret[:, 1:] = np.diff(np.log(P), axis=1)
logret = np.nan_to_num(logret)
if bool(CFG["data"].get("cross_sectional_demean", False)):
    logret = logret - logret.mean(axis=0, keepdims=True)
    print("cross-sectional demean: ON")
# standardise per asset (scale-only; coherence + trace-normalised eigenspectrum
# are invariant to it -- keeps long-period power off the 1e-12 coherence floor)
lr_std = logret / (logret.std(axis=1, keepdims=True) + 1e-12)

sig = torch.tensor(lr_std)
print("CWT...", flush=True)
bank = MorletCWTBank(periods, n_time, coi_factor=w["coi_factor"])
coeffs, ndrop = bank.transform_causal(sig)                              # [A, F, n_time-l_max]
print(f"  coeffs {tuple(coeffs.shape)}; coherence ({A*(A-1)//2} pairs)...", flush=True)
wc = WaveletCoherence(A, smooth_time_steps=coh_cfg["smooth_time_steps"], causal=True)(coeffs)
print("  hermitian graph...", flush=True)
herm = HermitianGraph(A, frequency_reduction=coh_cfg["frequency_reduction"],
                      diagonal=coh_cfg.get("diagonal", "one"))
H = herm.build(wc["complex_coherence"], coeffs.shape[-1])              # [T', A, A]
print("  eigendecomposition...", flush=True)
dec = SpectralDecomposition(SpectralConfig(
    n_components=sp_cfg["n_components"],
    eigenvalue_normalize=sp_cfg["eigenvalue_normalize"],
    canonicalize_phase=sp_cfg["canonicalize_phase"],
)).decompose(H)
feats = SpectralDecomposition(SpectralConfig(n_components=sp_cfg["n_components"])).features(
    dec["eigenvalues"], dec["eigenvectors"] if sp_cfg["use_eigenvectors"] else None)  # [T', d_spec]

# --- slice everything to the last N_SHOW output bars ---
sw = slice(-N_SHOW, None)
show_ts = ts[ndrop:][sw]
btc_power = (coeffs[0].abs() ** 2).numpy()[:, sw]                       # [F, N_SHOW]
coh_mag_f = wc["coherence"].mean(0).numpy()[:, sw]                      # mean |C| over pairs [F, N_SHOW]
Hs = H[sw].numpy()                                                     # [N_SHOW, A, A]
evals = dec["eigenvalues"][sw].numpy()                                 # [N_SHOW, k]
featm = feats[sw].numpy()                                              # [N_SHOW, d_spec]
k = evals.shape[1]
print("feature matrix:", featm.shape, " (== d_spec)")

# ---------------------------------------------------------------- plotting
x = np.arange(N_SHOW)
xt = np.linspace(0, N_SHOW - 1, 9).astype(int)
xtl = [show_ts[i].strftime("%m-%d\n%H:%M") for i in xt]
yti = np.linspace(0, F - 1, 7).astype(int)
ylf = [f"{periods[i]:.0f}b/{periods[i]*5/60:.1f}h" for i in yti]
snap_i = np.linspace(N_SHOW // 8, N_SHOW - 1, 4).astype(int)

fig = plt.figure(figsize=(12, 20))
gs = GridSpec(7, 4, figure=fig, height_ratios=[1, 1, 1.2, 1.2, 1.3, 1.2, 1.5], hspace=.42, wspace=.28)
fig.suptitle(f"market_state feature pipeline — one {N_SHOW}-bar ({N_SHOW*5/60/24:g}d) window, "
             f"all {A} assets, ending {show_ts[-1]:%Y-%m-%d %H:%M} UTC\n"
             f"nfreqs={w['nfreqs']}  smooth_time_steps={coh_cfg['smooth_time_steps']}  "
             f"n_components={sp_cfg['n_components']}  d_spec={featm.shape[1]}", fontsize=11)

ax = fig.add_subplot(gs[0, :])
for a in range(A):
    ax.plot(x, P[a, ndrop:][sw] / P[a, ndrop:][sw][0] - 1, lw=.5, alpha=.5)
ax.plot(x, P[0, ndrop:][sw] / P[0, ndrop:][sw][0] - 1, lw=1.4, color="k", label="BTC")
ax.set_ylabel("cum. return"); ax.legend(fontsize=8, loc="upper left")
ax.set_title(f"1 · aligned prices  [{A} assets]  — 5-min Binance bars, common timestamp index", loc="left", fontsize=9)
ax.set_xticks(xt); ax.set_xticklabels([])

ax = fig.add_subplot(gs[1, :])
im = ax.imshow(lr_std[:, ndrop:][:, sw], aspect="auto", cmap="RdBu_r", vmin=-4, vmax=4,
               extent=[0, N_SHOW, A - .5, -.5])
ax.set_ylabel("asset"); fig.colorbar(im, ax=ax, pad=.01, label="σ")
ax.set_title("2 · log-returns  r = Δ ln P  [A × T]  (per-asset unit-variance; CWT input)", loc="left", fontsize=9)
ax.set_xticks(xt); ax.set_xticklabels([])

ax = fig.add_subplot(gs[2, :])
im = ax.imshow(np.log10(btc_power + 1e-12), aspect="auto", origin="lower", cmap="magma",
               extent=[0, N_SHOW, -.5, F - .5])
ax.set_yticks(yti); ax.set_yticklabels(ylf, fontsize=7); ax.set_ylabel("period")
fig.colorbar(im, ax=ax, pad=.01, label="log₁₀|W|²")
ax.set_title("3 · causal Morlet CWT — BTC scalogram  [F × T]  (one per asset; right-zero-padded, trailing-aligned)", loc="left", fontsize=9)
ax.set_xticks(xt); ax.set_xticklabels([])

ax = fig.add_subplot(gs[3, :])
im = ax.imshow(coh_mag_f, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=1,
               extent=[0, N_SHOW, -.5, F - .5])
ax.set_yticks(yti); ax.set_yticklabels(ylf, fontsize=7); ax.set_ylabel("period")
fig.colorbar(im, ax=ax, pad=.01, label="mean |C| over pairs")
ax.set_title(f"4 · complex wavelet coherence  W_ij(f,t) = C·e^{{iφ}}  [P={A*(A-1)//2} pairs × F × T]  — mean magnitude shown; phase kept", loc="left", fontsize=9)
ax.set_xticks(xt); ax.set_xticklabels([])

for c, si in enumerate(snap_i):
    axm = fig.add_subplot(gs[4, c])
    im = axm.imshow(np.abs(Hs[si]), cmap="cividis", vmin=0, vmax=1)
    axm.set_title(show_ts[si].strftime("%m-%d %H:%M"), fontsize=8)
    axm.set_xticks([]); axm.set_yticks([])
    if c == 0:
        axm.set_ylabel("|H(t)|  [A×A]", fontsize=8)
fig.text(0.13, gs[4, 0].get_position(fig).y1 + .006,
         "5 · Hermitian market graph  H_ij = Σ_f w_f W_ij(f) / Σ_f w_f  (magnitude-weighted freq reduction) — 4 snapshots",
         fontsize=9)

ax = fig.add_subplot(gs[5, :])
for j in range(k):
    ax.plot(x, evals[:, j], lw=.9, label=f"λ{j+1}" if j < 4 else None)
ax.axhline(0, color="k", lw=.5)
ax.set_ylabel("eigenvalue λ"); ax.legend(fontsize=7, ncol=4, loc="upper left")
ax.set_title(f"6 · eigendecomposition  H(t) = Σ λ_m(t) u_m(t) u_m(t)*  — top {k} |λ| tracks  (λ₁ ≫ rest ⇒ market moving as one)", loc="left", fontsize=9)
ax.set_xticks(xt); ax.set_xticklabels([])

ax = fig.add_subplot(gs[6, :])
im = ax.imshow(featm.T, aspect="auto", cmap="RdBu_r", vmin=-.4, vmax=.4,
               extent=[0, N_SHOW, featm.shape[1] - .5, -.5])
ax.set_ylabel("feature dim"); fig.colorbar(im, ax=ax, pad=.01)
bounds = np.cumsum([0, k, k, k, 1, 1, 1, 1, A])
names = ["λ", "λ/Σ", "|λ|/Σ", "λmax", "entropy", "conc", "top3", "node loadings |u|²"]
for b0, b1, nm in zip(bounds[:-1], bounds[1:], names):
    ax.axhline(b1 - .5, color="k", lw=.6)
    ax.text(N_SHOW * 1.005, (b0 + b1) / 2, nm, fontsize=7, va="center")
ax.set_title(f"7 · spectral feature matrix  [T × d_spec={featm.shape[1]}]  — the Mamba-3 input (one 288/1152-row window slides over this)", loc="left", fontsize=9)
ax.set_xticks(xt); ax.set_xticklabels(xtl, fontsize=7); ax.set_xlabel("bar time (UTC)")

out = sys.argv[1] if len(sys.argv) > 1 else "pipeline_stages.png"
fig.savefig(out, dpi=120, bbox_inches="tight")
print("wrote", out)
