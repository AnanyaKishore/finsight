"""Configuration loading.

Two sources, deliberately separated:

* **Secrets and deployment wiring** (API keys, Mongo credentials, hosts) come
  from environment variables, optionally loaded from a git-ignored ``.env``.
* **Tunable, versioned parameters** come from ``params.yaml`` (tracked by DVC as
  stage params) and the ticker universe from ``configs/tickers.yaml``.

This keeps secrets out of Git while every value that can change a result is
versioned and reproducible from a commit hash.
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

import yaml
from dotenv import load_dotenv

from finsight.common.exceptions import ConfigError

PLACEHOLDERS = frozenset({"", "replace-me", "changeme", "change-me"})


def usable_secret(value: str | None, min_length: int = 16) -> str | None:
    """A shared secret, or ``None`` if it is missing, a placeholder or too short.

    Used for ``SCHEDULER_TOKEN``: the copy of ``.env.example`` says
    ``replace-me``, and accepting that would make the trigger API guessable.
    """
    value = (value or "").strip()
    if value.lower() in PLACEHOLDERS or len(value) < min_length:
        return None
    return value


def find_project_root(start: Path | None = None) -> Path:
    """Locate the project root.

    Order: ``$FINSIGHT_ROOT`` if set, otherwise the first ancestor of ``start``
    (default: CWD) that contains ``params.yaml``.

    Raises:
        ConfigError: If no root can be found.
    """
    env_root = os.getenv("FINSIGHT_ROOT")
    if env_root:
        return Path(env_root).resolve()
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "params.yaml").is_file():
            return candidate
    raise ConfigError("Cannot find project root (no params.yaml); set FINSIGHT_ROOT.")


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except FileNotFoundError as exc:
        raise ConfigError(f"Missing config file: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"Top level of {path} must be a mapping")
    return data


@dataclass(frozen=True)
class Ticker:
    """One stock in the fixed universe."""

    symbol: str
    name: str
    aliases: tuple[str, ...] = ()
    sector: str = ""


@dataclass(frozen=True)
class Settings:
    """Resolved runtime settings (immutable)."""

    root: Path
    params: dict[str, Any]
    tickers: tuple[Ticker, ...]
    mongo_host: str = "localhost"
    mongo_port: int = 27017
    mongo_user: str | None = None
    mongo_password: str | None = field(default=None, repr=False)
    mongo_db: str = "finsight"
    marketstack_api_key: str | None = field(default=None, repr=False)
    marketstack_base_url: str = "https://api.marketstack.com/v2"
    finnhub_api_key: str | None = field(default=None, repr=False)
    finnhub_base_url: str = "https://finnhub.io/api/v1"
    log_level: str = "INFO"

    @property
    def mongo_uri(self) -> str:
        """Build the MongoDB URI, URL-escaping credentials."""
        if self.mongo_user and self.mongo_password:
            creds = f"{quote_plus(self.mongo_user)}:{quote_plus(self.mongo_password)}@"
            return f"mongodb://{creds}{self.mongo_host}:{self.mongo_port}/?authSource=admin"
        return f"mongodb://{self.mongo_host}:{self.mongo_port}/"

    @property
    def symbols(self) -> list[str]:
        """Ticker symbols in configured order."""
        return [t.symbol for t in self.tickers]

    def section(self, name: str) -> dict[str, Any]:
        """Return a ``params.yaml`` section, raising a clear error if absent."""
        try:
            value = self.params[name]
        except KeyError as exc:
            raise ConfigError(f"params.yaml has no '{name}' section") from exc
        if not isinstance(value, dict):
            raise ConfigError(f"params.yaml section '{name}' must be a mapping")
        return value

    def require(self, attr: str) -> str:
        """Return a secret/setting that must be present, e.g. an API key.

        Raises:
            ConfigError: If the value is empty or still the ``replace-me`` placeholder.
        """
        value = getattr(self, attr)
        if not value or str(value).strip().lower() in {"replace-me", "changeme"}:
            raise ConfigError(f"Setting '{attr}' is not configured; fill it in .env")
        return str(value)


def _parse_tickers(raw: dict[str, Any]) -> tuple[Ticker, ...]:
    entries = raw.get("tickers")
    if not entries:
        raise ConfigError("configs/tickers.yaml must define a non-empty 'tickers' list")
    tickers = []
    for entry in entries:
        try:
            tickers.append(
                Ticker(
                    symbol=str(entry["symbol"]).upper(),
                    name=str(entry["name"]),
                    aliases=tuple(str(a).lower() for a in entry.get("aliases", [])),
                    sector=str(entry.get("sector", "")),
                )
            )
        except (KeyError, TypeError) as exc:
            raise ConfigError(f"Bad ticker entry {entry!r}: {exc}") from exc
    symbols = [t.symbol for t in tickers]
    if len(set(symbols)) != len(symbols):
        raise ConfigError("Duplicate symbols in configs/tickers.yaml")
    return tuple(tickers)


def load_settings(root: Path | None = None) -> Settings:
    """Load settings from ``.env`` / environment, ``params.yaml`` and ``tickers.yaml``.

    Variables already present in the environment win over ``.env`` (so Docker
    Compose can override ``MONGO_HOST=mongo`` for containers).
    """
    project_root = root or find_project_root()
    load_dotenv(project_root / ".env", override=False)
    env = os.environ
    return Settings(
        root=project_root,
        params=_load_yaml(project_root / "params.yaml"),
        tickers=_parse_tickers(_load_yaml(project_root / "configs" / "tickers.yaml")),
        mongo_host=env.get("MONGO_HOST", "localhost"),
        mongo_port=int(env.get("MONGO_PORT", "27017")),
        mongo_user=env.get("MONGO_ROOT_USER") or None,
        mongo_password=env.get("MONGO_ROOT_PASSWORD") or None,
        mongo_db=env.get("MONGO_DB_NAME", "finsight"),
        marketstack_api_key=env.get("MARKETSTACK_API_KEY") or None,
        marketstack_base_url=env.get("MARKETSTACK_BASE_URL", "https://api.marketstack.com/v2"),
        finnhub_api_key=env.get("FINNHUB_API_KEY") or None,
        finnhub_base_url=env.get("FINNHUB_BASE_URL", "https://finnhub.io/api/v1"),
        log_level=env.get("LOG_LEVEL", "INFO"),
    )


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide cached settings (call ``get_settings.cache_clear()`` in tests)."""
    return load_settings()
