"""Latency / throughput benchmark for the market-state model inference.

Measures, per device (cpu and, when available, mps):

  * feature-pipeline cost per window (CWT + coherence + Hermitian + spectral),
  * model forward-pass cost per decision window,
  * end-to-end cost (feature + model) per decision,
  * the streaming ``MarketStateInference.step`` cost.

Everything is run with a randomly-initialised model (no training required by
the benchmark; training is out of scope). Realistic window length / universe
come from config.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from transforms.cwt import financial_periods, MorletCWTBank
from transforms.pipeline import FeaturePipeline, FeaturePipelineConfig
from models.market_state import build_from_config


def _batch_forward_time(model: torch.nn.Module, x: torch.Tensor, n_iters: int = 50) -> float:
    model.eval()
    if x.device.type != "cpu":
        torch.mps.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_iters):
            model(x)
    if x.device.type != "cpu":
        torch.mps.synchronize()
    return (time.perf_counter() - t0) / n_iters


def benchmark(
    config: dict,
    *,
    n_assets: int,
    window_len: int,
    n_batch: int = 1,
    n_iters: int = 50,
    devices: list[str] | None = None,
) -> dict:
    """Run the benchmark over the requested devices.

    Returns dict keyed by device with timings (seconds) per operation.
    """
    periods = financial_periods(config["wavelet"]["periods"], config["data"]["timeframe"])
    devices = devices or ["cpu"]
    coi = config["wavelet"].get("coi_factor", 3.0)
    smooth = config["wavelet"].get("smooth_time_steps", 5)
    # n_time must comfortably exceed window_len + causal warm-up so a full
    # window of surviving causal rows exists after the pipeline drops l_max.
    l_max = MorletCWTBank(periods, 1, coi_factor=coi).l_max
    n_time = window_len + 2 * l_max + smooth

    model_cfg = dict(config.get("model", {}))
    model_cfg["n_assets"] = n_assets

    report: dict = {}
    for dev in devices:
        avail = dev == "cpu" or (hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
        if not avail:
            report[dev] = {"available": False}
            continue

        fpc = FeaturePipelineConfig(device=dev, use_eigenvectors=False,
                                    n_components=config["spectral"].get("n_components", 8),
                                    coi_factor=coi)
        pl = FeaturePipeline(fpc, n_assets, periods, n_time)
        d_spec = pl.d_features
        model = build_from_config(model_cfg, d_spec, targets_cfg=config.get("targets", {})).to(dev)

        # periodic-ish random input
        rng = np.random.default_rng(0)
        lr = rng.standard_normal((n_assets, n_time)).astype(np.float32)

        # feature pipeline timing (per full recompute)
        t0 = time.perf_counter()
        for _ in range(n_iters):
            pl.compute(lr)
        if dev != "cpu":
            torch.mps.synchronize()
        feat_t = (time.perf_counter() - t0) / n_iters

        # model forward timing
        x = torch.randn(n_batch, window_len, d_spec, device=dev)
        model_t = _batch_forward_time(model, x, n_iters)

        # end-to-end (feature recompute + model)
        feats, _ = pl.compute(lr)
        xfeat = torch.as_tensor(feats[-n_batch*window_len:].reshape(n_batch, window_len, -1), device=dev)
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_iters):
                model(xfeat)
        if dev != "cpu":
            torch.mps.synchronize()
        model_on_feat = (time.perf_counter() - t0) / n_iters

        report[dev] = {
            "available": True,
            "n_assets": n_assets,
            "window_len": window_len,
            "d_spec": d_spec,
            "feature_pipeline_s": feat_t,
            "model_forward_s": model_t,
            "model_on_feat_s": model_on_feat,
            "e2e_s": feat_t + model_on_feat,
        }
    return report


def format_report(report: dict) -> str:
    lines = ["Inference benchmark (seconds/decision):"]
    for dev, r in report.items():
        if not r.get("available"):
            lines.append(f"  {dev}: unavailable")
            continue
        lines.append(f"  {dev}:")
        lines.append(f"    feature pipeline : {r['feature_pipeline_s']*1e3:8.2f} ms")
        lines.append(f"    model forward    : {r['model_forward_s']*1e3:8.2f} ms")
        lines.append(f"    end-to-end       : {r['e2e_s']*1e3:8.2f} ms")
    return "\n".join(lines)
