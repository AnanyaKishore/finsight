"""Champion / challenger decision rules.

A challenger replaces the champion (the version holding the ``@production``
alias) only if

1. it passes its **acceptance criterion** (LSTM: test MAPE within
   ``acceptance_max_ratio_to_naive`` of the naive forecast; sentiment: 3-class
   macro-F1 above the majority-class baseline), **and**
2. it is **better than the champion on the same evaluation data** by more than
   ``min_improvement`` (LSTM: lower MAPE on today's test windows, with the
   champion re-scored on them; sentiment: higher macro-F1 on the same test half).

With no champion yet (first run), the best candidate is promoted so the system
can serve, and the decision records whether it passed acceptance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

PRODUCTION, STAGING = "production", "staging"


@dataclass(frozen=True)
class Decision:
    """Outcome of one gate evaluation."""

    promote: bool
    reason: str
    challenger_version: str | None
    champion_version: str | None
    challenger_metric: float | None
    champion_metric: float | None
    metric: str
    challenger_accepted: bool

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly form (metrics rounded)."""
        out = asdict(self)
        for key in ("challenger_metric", "champion_metric"):
            if out[key] is not None:
                out[key] = round(float(out[key]), 6)
        return out


def decide(
    *,
    metric: str,
    challenger_version: str | None,
    challenger_metric: float | None,
    challenger_accepted: bool,
    champion_version: str | None,
    champion_metric: float | None,
    higher_is_better: bool,
    min_improvement: float = 0.0,
) -> Decision:
    """Apply the promotion rule (see module docstring)."""
    common = {
        "metric": metric,
        "challenger_version": challenger_version,
        "champion_version": champion_version,
        "challenger_metric": challenger_metric,
        "champion_metric": champion_metric,
        "challenger_accepted": challenger_accepted,
    }
    if challenger_version is None:
        return Decision(False, "no eligible challenger", **common)
    if champion_version is not None and challenger_version == champion_version:
        return Decision(False, "challenger is already production", **common)
    if champion_version is None:
        note = "passes acceptance" if challenger_accepted else "FAILS acceptance (bootstrap only)"
        return Decision(True, f"no production model yet; promoted ({note})", **common)
    if not challenger_accepted:
        return Decision(False, "challenger fails acceptance criterion", **common)
    if champion_metric is None:
        return Decision(
            True, "champion not comparable on current data; challenger accepted", **common
        )
    gain = (
        challenger_metric - champion_metric
        if higher_is_better
        else champion_metric - challenger_metric
    )
    if gain > min_improvement:
        return Decision(True, f"challenger better by {gain:.6f} {metric}", **common)
    return Decision(
        False, f"challenger not better (gain {gain:.6f} <= {min_improvement})", **common
    )


def lstm_accepted(test_mape: float, naive_mape: float, max_ratio: float) -> bool:
    """LSTM acceptance: MAPE no worse than ``max_ratio`` x the naive forecast's."""
    return float(test_mape) <= float(naive_mape) * float(max_ratio)


def pick_best(candidates: dict[str, dict[str, Any]], order: list[str]) -> str | None:
    """Best eligible sentiment candidate: accepted first, then highest metric.

    Ties keep the order of ``sentiment.candidates`` in ``params.yaml`` (the
    proposal's DistilBERT first). Candidates marked ``rolled_back`` are skipped.
    """
    eligible = [k for k in order if k in candidates and not candidates[k].get("rolled_back")]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda k: (
            bool(candidates[k]["accepted"]),
            float(candidates[k]["metric"]),
            -order.index(k),
        ),
    )
