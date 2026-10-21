"""FinSight: an AIOps-driven stock analysis platform.

Sub-packages:
    common     -- configuration, structured logging, retries, HTTP and MongoDB helpers.
    ingestion  -- MarketStack / Finnhub ingestion with quota guarding and run records.

Later days add: etl (Spark), models (LSTM, sentiment, portfolio), serving, backend,
scheduler and monitoring.
"""

__version__ = "0.1.0"
