"""Spot-instance training helpers: SIGTERM handling, S3 checkpoint sync, sentinels.

The walk-forward run is a sequence of independent per-fold trainings (fresh
model each fold), so the natural interruption granularity is **per fold**:

  * every completed fold's checkpoint (``fold_XXX.pt``) is pushed to
    ``<s3_prefix>/`` immediately;
  * ``progress.json`` (key ``epoch`` = last completed fold index, for the
    keepalive crash-loop guard) is updated alongside it;
  * on ``SIGTERM`` (spot 2-minute reclaim) the trainer stops at the next fold
    boundary via :class:`SpotInterrupt` and exits 0 **without** writing
    ``DONE`` -- the partial fold is discarded and redone on the next box;
  * on a clean finish of every fold the trainer writes ``<s3_prefix>/DONE``,
    which tells the keepalive workflow to stop relaunching.

All S3 calls shell out to the ``aws`` CLI and are best-effort: a transient
failure logs and continues rather than killing a multi-hour run.
"""
from __future__ import annotations

import json
import signal
import subprocess
import time
from pathlib import Path


class SpotInterrupt(Exception):
    """Raised inside the fold loop when a stop has been requested."""


_STOP = {"v": False}


def install_signal_handlers() -> None:
    def _handler(signum, _frame):
        _STOP["v"] = True
        print(f"[spot] caught signal {signum} -- stopping at next fold boundary", flush=True)

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _handler)
        except (ValueError, OSError):
            pass


def should_stop() -> bool:
    return _STOP["v"]


def _run(args: list[str], timeout: int = 900) -> bool:
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0:
            print(f"[spot] {' '.join(args[:3])}... rc={r.returncode}: {r.stderr.strip()[:200]}", flush=True)
        return r.returncode == 0
    except Exception as exc:  # noqa: BLE001 - best effort
        print(f"[spot] {' '.join(args[:3])}... raised {exc}", flush=True)
        return False


def s3_cp(src, dst) -> bool:
    return _run(["aws", "s3", "cp", str(src), str(dst)])


def pull_checkpoints(s3_prefix: str, local_dir) -> None:
    """Best-effort restore of any ``fold_*.pt`` already in S3 into ``local_dir``."""
    local = Path(local_dir)
    local.mkdir(parents=True, exist_ok=True)
    _run([
        "aws", "s3", "cp", s3_prefix.rstrip("/") + "/", str(local) + "/",
        "--recursive", "--exclude", "*", "--include", "fold_*.pt",
    ], timeout=1800)
    got = sorted(p.name for p in local.glob("fold_*.pt"))
    print(f"[spot] resumed {len(got)} fold checkpoint(s) from {s3_prefix}: {got}", flush=True)


def write_progress(s3_prefix: str | None, local_dir, fold_idx: int, total: int,
                   extra: dict | None = None) -> None:
    payload = {"epoch": int(fold_idx), "fold": int(fold_idx),
               "total_folds": int(total), "ts": time.time()}
    if extra:
        payload.update(extra)
    p = Path(local_dir) / "progress.json"
    p.write_text(json.dumps(payload, indent=2))
    if s3_prefix:
        s3_cp(p, s3_prefix.rstrip("/") + "/progress.json")


def write_done(s3_prefix: str | None, local_dir, status: str = "ok", msg: str = "") -> None:
    p = Path(local_dir) / "DONE"
    p.write_text(f"{status} {msg}".strip() + "\n")
    if s3_prefix:
        s3_cp(p, s3_prefix.rstrip("/") + "/DONE")
    print(f"[spot] wrote DONE ({status}) {msg}".strip(), flush=True)
