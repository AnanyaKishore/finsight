"""Data ingestion: external providers -> MongoDB.

Flow of one run (``python -m finsight.ingestion.cli incremental``)::

    plan ranges per ticker  (gap-fill from last stored date, or backfill window)
          |
          v
    prices: MarketStack ---(quota exhausted / provider down)---> yfinance fallback
    news:   Finnhub (weekly windows, rate-limited)
          |
          v
    idempotent upserts into Mongo  ->  run record in `pipeline_runs`
                                   ->  reports/ingestion_last_run.json
"""
