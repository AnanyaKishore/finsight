"""DVC stage ``fetch_sentiment``: download the pinned model and evaluation data once.

* **Model** -- ``distilbert/distilbert-base-uncased-finetuned-sst-2-english`` at
  a pinned Hub *commit* (``sentiment.base_revision``), PyTorch safetensors
  weights and tokenizer only (~268 MB). Saved to ``models/sentiment/base``.
* **Evaluation data** -- Financial PhraseBank v1.0 (Malo et al., 2014; CC BY-NC-SA
  3.0): 4,846 English financial-news sentences labelled positive / negative /
  neutral by 5-8 annotators. Only the requested agreement level
  (``AllAgree`` = 100 % annotator agreement, 2,264 sentences) is extracted to
  ``data/external/financial_phrasebank.csv``.

Both outputs are DVC-cached, so after the first run every other stage (and
every later machine, via ``dvc pull``) works offline; the stage re-runs only
if the pinned revisions in ``params.yaml`` change. The raw ZIP is read
directly (the Hub's old ``datasets`` loading script is no longer supported).
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
import tempfile
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from finsight.common.config import get_settings
from finsight.common.exceptions import FinSightError
from finsight.common.logging import configure_logging, get_logger
from finsight.common.retry import retry_with_backoff
from finsight.common.runs import stage_run

log = get_logger(__name__)

MODEL_FILES = ["config.json", "model.safetensors", "tokenizer_config.json", "vocab.txt"]
PHRASEBANK_ZIP = "data/FinancialPhraseBank-v1.0.zip"
LABELS = {"positive", "negative", "neutral"}


def _network_error(exc: BaseException) -> bool:
    """Retry connection problems and server errors, not 404s or bad revisions."""
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status is not None:
        return status >= 500 or status == 429
    return isinstance(exc, OSError | TimeoutError | ConnectionError)


_download = retry_with_backoff(max_retries=4, base_delay=2.0, is_retryable=_network_error)


def fetch_model(repo_id: str, revision: str, dest: Path) -> dict[str, Any]:
    """Download the model files at ``revision`` into ``dest`` (replacing it)."""
    from huggingface_hub import snapshot_download  # noqa: PLC0415

    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True)
    _download(snapshot_download)(
        repo_id=repo_id, revision=revision, local_dir=str(dest), allow_patterns=MODEL_FILES
    )
    shutil.rmtree(dest / ".cache", ignore_errors=True)  # Hub bookkeeping, not model content
    missing = [name for name in MODEL_FILES if not (dest / name).exists()]
    if missing:
        raise FinSightError(f"model download incomplete, missing {missing}")
    source = {"repo_id": repo_id, "revision": revision, "files": MODEL_FILES}
    (dest / "SOURCE.json").write_text(json.dumps(source, indent=2) + "\n", encoding="utf-8")
    size_mb = sum(p.stat().st_size for p in dest.iterdir()) / 1e6
    log.info("sentiment model fetched", extra={"repo": repo_id, "size_mb": round(size_mb, 1)})
    return {**source, "size_mb": round(size_mb, 1)}


def parse_phrasebank(raw: bytes, agreement: str) -> pd.DataFrame:
    """Parse one ``Sentences_<agreement>.txt`` file (ISO-8859-1, ``sentence@label``).

    Raises:
        FinSightError: If a line has an unknown label or the file is empty.
    """
    rows = []
    for number, line in enumerate(raw.decode("iso-8859-1").splitlines(), start=1):
        if not line.strip():
            continue
        sentence, sep, label = line.rpartition("@")
        label = label.strip().lower()
        if not sep or label not in LABELS:
            raise FinSightError(f"PhraseBank line {number}: cannot parse {line[:60]!r}")
        rows.append({"id": len(rows), "sentence": sentence.strip(), "label": label})
    if not rows:
        raise FinSightError("PhraseBank file is empty")
    frame = pd.DataFrame(rows)
    frame["agreement"] = agreement
    return frame


def phrasebank_from_zip(archive: bytes, agreement: str) -> pd.DataFrame:
    """Extract and parse the requested agreement level from the v1.0 ZIP.

    Raises:
        FinSightError: If the ZIP has no file for ``agreement``.
    """
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        names = [n for n in zf.namelist() if n.endswith(f"Sentences_{agreement}.txt")]
        if not names:
            raise FinSightError(f"no Sentences_{agreement}.txt in the PhraseBank archive")
        return parse_phrasebank(zf.read(names[0]), agreement)


def fetch_phrasebank(repo_id: str, revision: str, agreement: str, dest: Path) -> dict[str, Any]:
    """Download the PhraseBank ZIP at ``revision`` and write the labelled CSV."""
    from huggingface_hub import hf_hub_download  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as tmp:
        path = _download(hf_hub_download)(
            repo_id=repo_id,
            filename=PHRASEBANK_ZIP,
            repo_type="dataset",
            revision=revision,
            local_dir=tmp,
        )
        frame = phrasebank_from_zip(Path(path).read_bytes(), agreement)
    dest.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(dest, index=False, lineterminator="\n")
    counts = frame["label"].value_counts().sort_index().to_dict()
    log.info("phrasebank fetched", extra={"rows": len(frame), "labels": counts})
    return {"repo_id": repo_id, "revision": revision, "rows": len(frame), "labels": counts}


def main(argv: Sequence[str] | None = None) -> int:
    """Stage entry point (``python -m finsight.sentiment.fetch``)."""
    argparse.ArgumentParser(description="Fetch the sentiment model + PhraseBank").parse_args(argv)
    settings = get_settings()
    configure_logging("fetch_sentiment", settings.log_level, settings.root / "logs")
    params = settings.section("sentiment")
    try:
        with stage_run("fetch_sentiment", settings) as run:
            model = fetch_model(
                params["base_model"], params["base_revision"], settings.root / params["base_dir"]
            )
            data = fetch_phrasebank(
                params["phrasebank_repo"],
                params["phrasebank_revision"],
                params["phrasebank_agreement"],
                settings.root / params["phrasebank_path"],
            )
            run.metrics = {"model_size_mb": model["size_mb"], "phrasebank_rows": data["rows"]}
    except Exception as exc:  # noqa: BLE001 -- report any download/parse failure cleanly
        print(f"fetch_sentiment failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"model": model, "phrasebank": data}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
