"""Data provider adapters (one module per external API)."""

from finsight.ingestion.providers.base import NewsProvider, PriceProvider

__all__ = ["NewsProvider", "PriceProvider"]
