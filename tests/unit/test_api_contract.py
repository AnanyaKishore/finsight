"""The published API contract (docs/api/openapi.json, LLD §2) matches the running code."""

from __future__ import annotations

import json
import re
from pathlib import Path

from finsight.api.app import create_app

ROOT = Path(__file__).resolve().parents[2]


def test_committed_openapi_matches_the_code():
    """If this fails, run `python scripts/export_openapi.py` and update LLD §2."""
    committed = json.loads((ROOT / "docs" / "api" / "openapi.json").read_text(encoding="utf-8"))
    generated = json.loads(json.dumps(create_app().openapi(), sort_keys=True))
    assert generated == committed


def test_every_endpoint_is_documented_in_the_lld():
    lld = (ROOT / "docs" / "LLD.md").read_text(encoding="utf-8")
    for path in create_app().openapi()["paths"]:
        stem = re.sub(r"/\{[^}]+\}.*$", "", path)
        assert stem in lld, f"{path} is missing from docs/LLD.md"
