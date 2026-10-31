"""Distribution-shift statistics (pure functions, numpy only).

* **PSI** (population stability index): bins come from the *reference*
  sample's quantiles, so each reference bin holds about the same share;
  ``PSI = sum((cur% - ref%) * ln(cur% / ref%))``. Rule of thumb: below 0.1
  stable, 0.1-0.25 moderate shift, above 0.25 significant shift. Each bin
  count gets +0.5 (additive smoothing) so an empty bin in a small current
  sample does not blow PSI up, which a tiny floor such as 1e-4 would.
* **KS** (two-sample Kolmogorov-Smirnov statistic): the largest gap between
  the two empirical CDFs, 0 (identical) to 1 (disjoint). Reported next to PSI
  because it needs no binning.
"""

from __future__ import annotations

import numpy as np

SMOOTHING = 0.5  # added to every bin count, so ln() stays finite and small samples stay calm


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population stability index of ``current`` against ``reference``."""
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref, cur = ref[np.isfinite(ref)], cur[np.isfinite(cur)]
    if len(ref) < bins or len(cur) == 0:
        raise ValueError(f"need at least {bins} reference and 1 current values")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # (almost) constant reference: fall back to equal-width bins
        lo, hi = min(ref.min(), cur.min()), max(ref.max(), cur.max())
        edges = np.linspace(lo, hi + 1e-12, bins + 1)
    # Open-ended outer bins: values beyond the reference range still count.
    inner = edges[1:-1]
    ref_share = np.bincount(np.searchsorted(inner, ref, side="right"), minlength=len(edges) - 1)
    cur_share = np.bincount(np.searchsorted(inner, cur, side="right"), minlength=len(edges) - 1)
    ref_p = (ref_share + SMOOTHING) / (ref_share.sum() + SMOOTHING * len(ref_share))
    cur_p = (cur_share + SMOOTHING) / (cur_share.sum() + SMOOTHING * len(cur_share))
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))


def ks_statistic(reference: np.ndarray, current: np.ndarray) -> float:
    """Two-sample KS statistic (max distance between the empirical CDFs)."""
    ref = np.sort(np.asarray(reference, dtype=float))
    cur = np.sort(np.asarray(current, dtype=float))
    if len(ref) == 0 or len(cur) == 0:
        raise ValueError("both samples must be non-empty")
    grid = np.concatenate([ref, cur])
    cdf_ref = np.searchsorted(ref, grid, side="right") / len(ref)
    cdf_cur = np.searchsorted(cur, grid, side="right") / len(cur)
    return float(np.max(np.abs(cdf_ref - cdf_cur)))
