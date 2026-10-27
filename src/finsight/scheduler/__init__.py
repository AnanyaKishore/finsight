"""Scheduler service (Day 7): runs the daily pipeline and on-demand retraining.

Modules:
    schedule  -- pure time logic: when is the next run, is a run due now
    plan      -- the steps of a daily / retrain run and the commands they run
    runner    -- executes a plan step by step, captures output, decides status
    store     -- MongoDB persistence of pipeline executions and the DVC DAG
    dag       -- DAG + stale stages from ``dvc dag --dot`` / ``dvc status --json``
    service   -- one run at a time, the background loop, Prometheus metrics
    app       -- HTTP API (``/run``, ``/retrain``, ``/status``, ``/metrics``)
"""
