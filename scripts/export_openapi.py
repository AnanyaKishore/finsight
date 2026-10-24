"""Write the backend's OpenAPI schema to docs/api/openapi.json (no services needed).

    python scripts/export_openapi.py

The schema is generated from the Pydantic models in finsight.api.schemas, so the
LLD's endpoint tables and this file cannot drift apart from the running code.
"""

import json
from pathlib import Path

from finsight.api.app import create_app

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    schema = create_app().openapi()  # MongoDB is connected lazily: nothing is contacted
    out = ROOT / "docs" / "api" / "openapi.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)}: {len(schema['paths'])} paths")
