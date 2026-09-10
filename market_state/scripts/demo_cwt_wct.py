"""One causal-CWT window and one wavelet-coherence (WCT) window from real data.

Illustrative only -- shows what a single ``window.bars`` slice looks like after
``MorletCWTBank.transform_causal`` + ``WaveletCoherence``, with the full
``l_max`` causal warm-up computed and discarded ahead of it.

    cd market_state
    ../.venv_market/bin/python scripts/demo_cwt_wct.py docs/cwt_wct_example.png

Reads data/raw/crypto/{BTC,ETH}_USDT.parquet; writes the figure to argv[1]
(default cwt_wct_demo.png).
"""
import sys, glob
import numpy as np, pandas as pd, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "src")
from transforms.cwt import period_to_bars, _log_spaced, MorletCWTBank
from transforms.wavelet_coherence import WaveletCoherence

TF = "5Min"
NFREQS = 48                        # dense log-spaced scale grid (crypto.yaml)
SMOOTH = 32                        # coherence.smooth_time_steps (crypto.yaml)
P_LO, P_HI = "15min", "2d"
A, B = "BTC_USDT", "ETH_USDT"
N_SHOW = 1152                      # causal output bars to display (4 days = proposed window.bars)
WIN_END = "2023-06-15 00:00:00+00:00"

periods = np.sort(_log_spaced(period_to_bars(P_LO, TF), period_to_bars(P_HI, TF), NFREQS))
l_max = int(np.ceil(3.0 * 2.0 * periods.max()))
# need left-pad (1728) + l_max (1728) of real signal behind the retained region
# before the longest-period band is clean -> take a long signal, show only the
# LAST N_SHOW causal-output bars (matches pipeline's lead = pad + l_max + 256).
n_time = l_max + l_max + N_SHOW
print(f"periods (bars): {periods}   l_max={l_max}   window n_time={n_time}")


def load_close(sym):
    d = pd.read_parquet(f"data/raw/crypto/{sym}.parquet")[["timestamp", "close"]]
    return d.set_index("timestamp")["close"].sort_index()


ca, cb = load_close(A), load_close(B)
idx = ca.index.intersection(cb.index)
ca, cb = ca.reindex(idx), cb.reindex(idx)
end_pos = idx.get_indexer([pd.Timestamp(WIN_END)], method="nearest")[0]
sl = slice(end_pos - n_time, end_pos)
ts = idx[sl]
pa, pb = ca.values[sl], cb.values[sl]
# log-returns, first bar -> 0
la = np.concatenate([[0.0], np.diff(np.log(pa))]).astype(np.float32)
lb = np.concatenate([[0.0], np.diff(np.log(pb))]).astype(np.float32)
# coherence is scale-invariant (C_ij unchanged if you rescale either series),
# but raw 5-min log-returns are ~2e-3 so long-period wavelet power lands near
# the 1e-12 epsilon in the coherence denominator and reads as ~0. Standardising
# each series to unit variance lifts it off that floor without touching the
# true coherence structure. (The production pipeline feeds RAW log-returns.)
la /= la.std(); lb /= lb.std()

sig = torch.tensor(np.stack([la, lb]))                  # [2, n_time]
bank = MorletCWTBank(periods, n_time, coi_factor=3.0)
coeffs_full, ndrop = bank.transform_causal(sig)         # [2, F, n_time - l_max]
coh_full = WaveletCoherence(2, smooth_time_steps=SMOOTH, causal=True)(coeffs_full)
coh_heavy = WaveletCoherence(2, smooth_time_steps=5, causal=True)(coeffs_full)  # old default, contrast
coeffs = coeffs_full[..., -N_SHOW:]
print("coeffs", tuple(coeffs_full.shape), "dropped", ndrop, "-> showing last", N_SHOW)

show_ts = ts[ndrop:][-N_SHOW:]
power_btc = (coeffs[0].abs() ** 2).numpy()              # [F, N_SHOW]
wct_mag = coh_full["coherence"][0][..., -N_SHOW:].numpy()   # pair 0 = BTC-ETH
wct_phase = coh_full["phase"][0][..., -N_SHOW:].numpy()
wct_mag_heavy = coh_heavy["coherence"][0][..., -N_SHOW:].numpy()
pa_s, pb_s = pa[-N_SHOW:], pb[-N_SHOW:]
yti = np.linspace(0, len(periods) - 1, 8).astype(int)
yl = [f"{periods[i]:.0f}b / {periods[i]*5/60:.1f}h" for i in yti]
x = np.arange(N_SHOW)
xt = np.linspace(0, N_SHOW - 1, 9).astype(int)
xtl = [show_ts[i].strftime("%m-%d\n%H:%M") for i in xt]

fig, ax = plt.subplots(5, 1, figsize=(11, 15.5), sharex=True)
fig.suptitle(f"Causal CWT + wavelet coherence — window ending {show_ts[-1]:%Y-%m-%d %H:%M} UTC\n"
             f"{N_SHOW} causal bars ({N_SHOW*5/60/24:g}d), {NFREQS} scales {periods[0]:.0f}–{periods[-1]:.0f} bar, "
             f"{l_max}-bar warm-up discarded", fontsize=11)

ax[0].plot(x, pa_s / pa_s[0] - 1, label="BTC", lw=.9)
ax[0].plot(x, pb_s / pb_s[0] - 1, label="ETH", lw=.9, alpha=.8)
ax[0].set_ylabel("cum. return\n(window start=0)"); ax[0].legend(loc="upper left", fontsize=8)
ax[0].set_title("1  aligned prices", loc="left", fontsize=9)

im1 = ax[1].imshow(np.log10(power_btc + 1e-12), aspect="auto", origin="lower",
                   extent=[0, N_SHOW, -.5, len(periods) - .5], cmap="magma")
ax[1].set_yticks(yti); ax[1].set_yticklabels(yl, fontsize=7)
ax[1].set_ylabel("period"); fig.colorbar(im1, ax=ax[1], pad=.01, label="log10 |W|²")
ax[1].set_title("2  BTC causal CWT scalogram (wavelet power)", loc="left", fontsize=9)

im2 = ax[2].imshow(wct_mag, aspect="auto", origin="lower", vmin=0, vmax=1,
                   extent=[0, N_SHOW, -.5, len(periods) - .5], cmap="viridis")
ax[2].set_yticks(yti); ax[2].set_yticklabels(yl, fontsize=7)
ax[2].set_ylabel("period"); fig.colorbar(im2, ax=ax[2], pad=.01, label="coherence |C|")
ax[2].set_title("3  BTC–ETH coherence magnitude  (smooth_time_steps=32, the config)", loc="left", fontsize=9)

im3 = ax[3].imshow(wct_phase, aspect="auto", origin="lower", vmin=-np.pi, vmax=np.pi,
                   extent=[0, N_SHOW, -.5, len(periods) - .5], cmap="twilight")
ax[3].set_yticks(yti); ax[3].set_yticklabels(yl, fontsize=7)
ax[3].set_ylabel("period"); fig.colorbar(im3, ax=ax[3], pad=.01, label="rel. phase (rad)")
ax[3].set_title("4  BTC–ETH relative phase  (+ = BTC leads)",
                loc="left", fontsize=9)

im4 = ax[4].imshow(wct_mag_heavy, aspect="auto", origin="lower", vmin=0, vmax=1,
                   extent=[0, N_SHOW, -.5, len(periods) - .5], cmap="viridis")
ax[4].set_yticks(yti); ax[4].set_yticklabels(yl, fontsize=7)
ax[4].set_ylabel("period"); fig.colorbar(im4, ax=ax[4], pad=.01, label="coherence |C|")
ax[4].set_title("5  same coherence at smooth_time_steps=5 (old default) — |C| saturates ~1",
                loc="left", fontsize=9)
ax[4].set_xticks(xt); ax[4].set_xticklabels(xtl, fontsize=7); ax[4].set_xlabel("bar time (UTC)")

fig.tight_layout(rect=[0, 0, 1, .96])
out = sys.argv[1] if len(sys.argv) > 1 else "cwt_wct_demo.png"
fig.savefig(out, dpi=130)
print("wrote", out)

# also a compact numeric peek
print("\nBTC-ETH coherence, last bar, by period:")
for p, m, ph in zip(periods, wct_mag[:, -1], wct_phase[:, -1]):
    print(f"  {p:4g} bars ({p*5/60:5.2f}h):  |C|={m:.3f}   phase={ph:+.2f} rad")
