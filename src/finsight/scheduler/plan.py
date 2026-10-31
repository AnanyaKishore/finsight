"""The steps of a pipeline run.

Two kinds of run:

* **daily** (scheduled every weekday morning, or "Run now" in the console)::

      ingest -> data -> score_sentiment -> optimize_portfolio -> forecasts
             -> monitor -> version

  ``data`` is ``dvc repro process`` (unit tests -> export -> validate -> Spark
  ETL). Sentiment and portfolio use ``dvc repro --single-item``, so the daily
  run never drags in ``train_lstm`` upstream of ``register``: the LSTM is not
  retrained every day.

* **retrain** ("Retrain now", or the ``/retrain`` webhook that Day 8's drift
  alerts call)::

      ingest -> retrain (full dvc repro: train, evaluate, register gate, ...)
             -> forecasts -> monitor -> version

``monitor`` (Day 8) recomputes drift and live forecast error right away, so
Grafana and the alerts see the new data without waiting for the next refresh.

A *critical* step stops the run when it fails (later steps would work on bad
or missing data). A non-critical failure is recorded and the run continues;
the run then ends as ``partial``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Literal

Kind = Literal["daily", "retrain"]
StepType = Literal["command", "forecasts", "monitor", "version"]

PY = sys.executable


@dataclass(frozen=True)
class Step:
    """One step of a run."""

    name: str
    label: str
    type: StepType = "command"
    argv: tuple[str, ...] = ()
    critical: bool = False
    partial_exit_codes: frozenset[int] = field(default_factory=frozenset)


INGEST = Step(
    "ingest",
    "Fetch new prices and news",
    argv=(PY, "-m", "finsight.ingestion.cli", "incremental"),
    # Exit 3 = some tickers failed but the rest were stored: carry on.
    partial_exit_codes=frozenset({3}),
)
DATA = Step(
    "data",
    "Tests, export, validation, Spark ETL",
    argv=("dvc", "repro", "process"),
    critical=True,
)
SCORE = Step(
    "score_sentiment",
    "Score new news articles",
    argv=("dvc", "repro", "--single-item", "score_sentiment"),
)
PORTFOLIO = Step(
    "optimize_portfolio",
    "Optimise the portfolio",
    argv=("dvc", "repro", "--single-item", "optimize_portfolio"),
)
RETRAIN = Step(
    "retrain",
    "Full pipeline: retrain, evaluate, promotion gate",
    argv=("dvc", "repro"),
    critical=True,
)
FORECASTS = Step("forecasts", "Store today's forecasts", type="forecasts")
MONITOR = Step("monitor", "Check drift and live forecast error", type="monitor")
VERSION = Step("version", "dvc push + local commit of dvc.lock", type="version")

PLANS: dict[str, tuple[Step, ...]] = {
    "daily": (INGEST, DATA, SCORE, PORTFOLIO, FORECASTS, MONITOR, VERSION),
    "retrain": (INGEST, RETRAIN, FORECASTS, MONITOR, VERSION),
}


def plan_for(kind: str) -> tuple[Step, ...]:
    """Steps for a run kind (``daily`` or ``retrain``)."""
    try:
        return PLANS[kind]
    except KeyError:
        raise ValueError(f"unknown run kind {kind!r}; expected one of {sorted(PLANS)}") from None


def overall_status(step_statuses: list[str], critical_failed: bool) -> str:
    """Run status from its steps: failed > partial > success."""
    if critical_failed:
        return "failed"
    if any(s in {"failed", "partial"} for s in step_statuses):
        return "partial"
    return "success"
