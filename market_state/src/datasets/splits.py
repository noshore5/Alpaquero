"""Walk-forward splits.

The causality/leakage requirements forbid a random train/test split. Instead
we use strictly chronological, non-overlapping gap-free folds following the
config:

    train_bars, validate_bars, test_bars, step_bars, expanding_window

At fold ``k``:
  - expanding: train = [0, boundary + train_bars], boundary advances by
    ``step_bars`` each fold; train grows monotonically.
  - sliding:   train = [boundary, boundary + train_bars], boundary advances
    by ``step_bars`` each fold; train is fixed-length tail.
  - val = [train_e, train_e + validate_bars]
  - test = [val_e, val_e + test_bars]

Folds stop when the test region would exceed ``n_rows``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Fold:
    train: np.ndarray
    validate: np.ndarray
    test: np.ndarray


def walk_forward_folds(
    n_rows: int,
    *,
    train_bars: int = 20000,
    validate_bars: int = 5000,
    test_bars: int = 2000,
    step_bars: int = 5000,
    expanding_window: bool = True,
) -> list[Fold]:
    """Chronological walk-forward fold row-index lists.

    Test regions advance by ``step_bars``. Validation is the block of
    ``validate_bars`` rows immediately before each test. Training is the
    prefix ending at the validation start (expanding) or the fixed-length tail
    (sliding), whichever keeps the training bank in-sample and disjoint from
    the validation/test regions.

    Non-overlap guard: training always ends strictly before the start of the
    *immediately preceding* test region (tracked as ``prev_test_start``), so
    data seen in an earlier test is never re-fed as training -- the causal
    "test-before-next-train" guard. When ``step_bars <= validate_bars`` this
    reduces to expanding training up to the current validation start.
    """
    folds: list[Fold] = []
    prev_test_start: int | None = None
    start_test = train_bars + validate_bars   # first test region start
    while True:
        test_s = start_test
        test_e = min(n_rows, test_s + test_bars)
        if test_e <= test_s or test_s >= n_rows:
            break
        val_s = max(0, test_s - validate_bars)
        # training end: before val start, and cleaned of any prior test overlap
        train_e_candidate = val_s if expanding_window else test_s
        if prev_test_start is not None:
            train_e_candidate = min(train_e_candidate, prev_test_start)
        if train_e_candidate <= 0:
            break
        train_s = 0 if expanding_window else max(0, train_e_candidate - train_bars)
        folds.append(Fold(
            train=np.arange(train_s, train_e_candidate),
            validate=np.arange(val_s, test_s),
            test=np.arange(test_s, test_e),
        ))
        prev_test_start = test_s
        start_test += step_bars

    return folds


def last_valid_target_row(targets: dict[str, np.ndarray]) -> int:
    """Index of the last row at which *all* targets are non-NaN."""
    limit = min(arr.shape[0] for arr in targets.values())
    while limit > 0 and any(np.isnan(np.asarray(arr[limit - 1])).any() for arr in targets.values()):
        limit -= 1
    return limit
