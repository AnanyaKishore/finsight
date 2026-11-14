"""Load test of the backend API (Day 9): latency and error rate under concurrent users.

    python scripts/load_test.py                    # 10 users for 60 s against localhost:8000
    python scripts/load_test.py --users 20 --duration 120

Each simulated user loops over the portal's read endpoints (weighted like a
person browsing: search, stock page, forecast, sentiment, portfolio). The
forecasts are warmed first, so ``/api/prediction`` measures the cached path
the portal normally hits. The live model endpoint (``/api/sentiment/analyze``)
runs DistilBERT/FinBERT on the CPU, so it is measured separately with a few
sequential calls and judged against its own 2 s target.

Writes ``reports/load_test.json`` (read by ``scripts/test_report.py``) and
prints a table. Acceptance (TEST_PLAN.md): p95 < 300 ms on cached
endpoints, error rate < 1 %, p95 < 2 s on the live model endpoint. Only the
standard library and ``requests`` are needed; no Locust/JMeter install.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import threading
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import requests

CACHED_P95_MS = 300.0
MODEL_P95_MS = 2000.0
MAX_ERROR_RATE = 0.01
QUERIES = ["app", "micro", "goo", "bank", "tes", "nv", "walm", "visa", "meta", "amaz"]


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile (q in 0..100) of a non-empty list."""
    ordered = sorted(values)
    rank = max(1, round(q / 100 * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def summarise(samples: list[float], errors: int) -> dict[str, float | int]:
    """Latency percentiles (ms) and error rate for one route."""
    n = len(samples) + errors
    if not samples:
        return {"requests": n, "errors": errors, "error_rate": 1.0 if n else 0.0}
    return {
        "requests": n,
        "errors": errors,
        "error_rate": round(errors / n, 4),
        "p50_ms": round(percentile(samples, 50), 1),
        "p95_ms": round(percentile(samples, 95), 1),
        "p99_ms": round(percentile(samples, 99), 1),
        "mean_ms": round(statistics.fmean(samples), 1),
        "max_ms": round(max(samples), 1),
    }


class Recorder:
    """Thread-safe latency and error collection, per route template."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.samples: dict[str, list[float]] = defaultdict(list)
        self.errors: dict[str, int] = defaultdict(int)
        self.error_examples: list[str] = []

    def call(self, session: requests.Session, route: str, method: str, url: str, **kw) -> None:
        started = time.perf_counter()
        try:
            response = session.request(method, url, timeout=10, **kw)
            ok = response.status_code < 500 and response.status_code != 429
            detail = f"{response.status_code} {url}"
        except requests.RequestException as exc:
            ok, detail = False, f"{exc.__class__.__name__} {url}"
        elapsed_ms = (time.perf_counter() - started) * 1000
        with self.lock:
            if ok:
                self.samples[route].append(elapsed_ms)
            else:
                self.errors[route] += 1
                if len(self.error_examples) < 10:
                    self.error_examples.append(detail)


def user_loop(
    base: str, tickers: list[str], rec: Recorder, stop: threading.Event, seed: int
) -> None:
    """One simulated user: a weighted random walk over the portal's endpoints."""
    rng = random.Random(seed)  # noqa: S311 -- workload mix, not security
    session = requests.Session()
    actions = [
        (
            3,
            "/api/stock/{ticker}",
            lambda t: f"/api/stock/{t}?period={rng.choice(['1M', '3M', '1Y'])}",
        ),
        (3, "/api/prediction/{ticker}", lambda t: f"/api/prediction/{t}"),
        (2, "/api/sentiment/{ticker}", lambda t: f"/api/sentiment/{t}?weeks=12"),
        (2, "/api/search", lambda t: f"/api/search?q={rng.choice(QUERIES)}"),
        (1, "/api/portfolio", lambda t: "/api/portfolio"),
        (1, "/api/stocks", lambda t: "/api/stocks"),
    ]
    weights = [a[0] for a in actions]
    while not stop.is_set():
        _, route, make = rng.choices(actions, weights=weights)[0]
        rec.call(session, route, "GET", base + make(rng.choice(tickers)))
        time.sleep(rng.uniform(0.05, 0.25))  # think time between clicks


def main(argv: list[str] | None = None) -> int:
    """Run the load test and write the JSON report."""
    parser = argparse.ArgumentParser(description="FinSight API load test")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--users", type=int, default=10)
    parser.add_argument("--duration", type=int, default=60, help="seconds")
    parser.add_argument("--model-calls", type=int, default=10)
    parser.add_argument("--out", default="reports/load_test.json")
    args = parser.parse_args(argv)
    base = args.base_url.rstrip("/")

    try:
        tickers = [
            s["symbol"] for s in requests.get(f"{base}/api/stocks", timeout=10).json()["stocks"]
        ]
    except (requests.RequestException, ValueError, KeyError) as exc:
        print(f"backend not reachable at {base}: {exc}", file=sys.stderr)
        return 2
    print(f"warming forecasts for {len(tickers)} stocks ...")
    for ticker in tickers:
        requests.get(f"{base}/api/prediction/{ticker}", timeout=60)

    rec = Recorder()
    stop = threading.Event()
    threads = [
        threading.Thread(target=user_loop, args=(base, tickers, rec, stop, i), daemon=True)
        for i in range(args.users)
    ]
    print(f"{args.users} users for {args.duration} s ...")
    started = time.perf_counter()
    for t in threads:
        t.start()
    time.sleep(args.duration)
    stop.set()
    for t in threads:
        t.join(15)
    wall = time.perf_counter() - started

    print(f"live model endpoint: {args.model_calls} sequential calls ...")
    model = Recorder()
    session = requests.Session()
    texts = [
        "Quarterly revenue beat expectations and the company raised guidance.",
        "The regulator opened an investigation into the accounting irregularities.",
    ]
    for _ in range(args.model_calls):
        model.call(
            session,
            "/api/sentiment/analyze",
            "POST",
            f"{base}/api/sentiment/analyze",
            json={"texts": texts},
        )

    routes = {
        r: summarise(rec.samples[r], rec.errors[r])
        for r in sorted(set(rec.samples) | set(rec.errors))
    }
    all_samples = [x for v in rec.samples.values() for x in v]
    total_errors = sum(rec.errors.values())
    overall = summarise(all_samples, total_errors)
    overall["throughput_rps"] = round(overall["requests"] / wall, 1)
    analyze = summarise(
        model.samples["/api/sentiment/analyze"], model.errors["/api/sentiment/analyze"]
    )
    acceptance = {
        "cached_p95_ms": {"value": overall.get("p95_ms"), "target": f"< {CACHED_P95_MS:.0f}",
                          "pass": overall.get("p95_ms", 1e9) < CACHED_P95_MS},
        "error_rate": {"value": overall["error_rate"], "target": f"< {MAX_ERROR_RATE:.0%}",
                       "pass": overall["error_rate"] < MAX_ERROR_RATE},
        "model_p95_ms": {"value": analyze.get("p95_ms"), "target": f"< {MODEL_P95_MS:.0f}",
                         "pass": analyze.get("p95_ms", 1e9) < MODEL_P95_MS
                         and analyze["error_rate"] == 0},
    }  # fmt: skip
    report = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "base_url": base,
        "users": args.users,
        "duration_s": args.duration,
        "overall": overall,
        "routes": routes,
        "model_endpoint": analyze,
        "acceptance": acceptance,
        "error_examples": rec.error_examples + model.error_examples,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"\n{'route':32} {'requests':>8} {'errors':>6} {'p50 ms':>8} {'p95 ms':>8} {'p99 ms':>8}")
    for name, r in [
        *routes.items(),
        ("ALL (cached endpoints)", overall),
        ("/api/sentiment/analyze (live)", analyze),
    ]:
        cells = [r.get(k, "-") for k in ("p50_ms", "p95_ms", "p99_ms")]
        print(
            f"{name:32} {r['requests']:>8} {r['errors']:>6} " + " ".join(f"{c:>8}" for c in cells)
        )
    print(f"\nthroughput {overall['throughput_rps']} requests/s")
    for name, a in acceptance.items():
        print(f"{'PASS' if a['pass'] else 'FAIL'}  {name}: {a['value']} (target {a['target']})")
    print(f"\nwrote {out}")
    return 0 if all(a["pass"] for a in acceptance.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
