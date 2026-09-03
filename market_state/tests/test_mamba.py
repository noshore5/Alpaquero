"""Regression tests for the complex parallel scan in models/mamba.py.

The mambapy (Blelloch) backward branch previously sliced the ``d_inner`` axis
where it meant the time axis, so ``loss.backward()`` raised
``_ComplexPScanBackward returned an invalid gradient`` for any ``expand > 1``
and no training was possible. These tests pin the fast path to the sequential
reference for the forward pass and both gradients, and check an end-to-end
backward through a MarketStateModel.
"""
import numpy as np
import pytest
import torch

import models.mamba as mm
from models.market_state import build_from_config


def _reference(A, X):
    """Sequential complex scan H_t = A_t H_{t-1} + X_t (autograd reference)."""
    H = []
    prev = torch.zeros_like(X[:, 0])
    for t in range(X.shape[1]):
        prev = A[:, t] * prev + X[:, t]
        H.append(prev)
    return torch.stack(H, dim=1)


@pytest.mark.skipif(not mm._HAS_MAMBAPY, reason="mambapy not installed; fast path inactive")
@pytest.mark.parametrize("L", [4, 7, 16, 31])
def test_complex_pscan_fast_matches_sequential(L):
    torch.manual_seed(0)
    B, D, N = 2, 5, 4
    A = (0.3 * torch.randn(B, L, D, N) + 0.1j * torch.randn(B, L, D, N)).cdouble()
    X = (torch.randn(B, L, D, N) + 1j * torch.randn(B, L, D, N)).cdouble()

    a1, x1 = A.clone().requires_grad_(), X.clone().requires_grad_()
    H_fast = mm._ComplexPScan.apply(a1, x1)
    g = torch.randn_like(H_fast)
    H_fast.backward(g)

    a2, x2 = A.clone().requires_grad_(), X.clone().requires_grad_()
    H_ref = _reference(a2, x2)
    H_ref.backward(g)

    assert torch.allclose(H_fast, H_ref, atol=1e-9)
    assert torch.allclose(a1.grad, a2.grad, atol=1e-7)
    assert torch.allclose(x1.grad, x2.grad, atol=1e-7)


def test_market_state_model_trains_one_step():
    torch.manual_seed(0)
    d_spec, n_assets = 12, 6
    model = build_from_config(
        {"n_assets": n_assets, "state_dim": 16, "mamba_expand": 2, "mamba_layers": 1},
        d_spec,
        targets_cfg={"return_horizons": [5], "realized_vol_horizons": [5]},
    )
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    x = torch.randn(8, 20, d_spec)
    tgt = torch.randn(8, n_assets)
    out = model(x)
    loss = torch.nn.functional.smooth_l1_loss(out["return_5"], tgt)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)
    opt.step()
