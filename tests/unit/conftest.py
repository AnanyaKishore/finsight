"""Unit-test isolation from the developer's machine.

Unit tests must behave identically on every machine, so they never read the
real ``.env`` (which holds your API keys). Without this, ``load_settings``
would re-load keys that a test deliberately removed with ``monkeypatch``.
Integration tests (tests/integration) still read ``.env`` because they need
the real MongoDB credentials.
"""

from __future__ import annotations

import pytest

from finsight.common import config as config_module


@pytest.fixture(autouse=True)
def _ignore_real_dotenv(monkeypatch):
    monkeypatch.setattr(config_module, "load_dotenv", lambda *args, **kwargs: False)
    config_module.get_settings.cache_clear()
    yield
    config_module.get_settings.cache_clear()