"""DVC stages ``fetch_sentiment@<candidate>`` and ``fetch_sentiment_data``.

Everything is downloaded once at a pinned Hugging Face Hub *commit* and then
DVC-cached, so every later stage (and every other machine, via ``dvc pull``)
works offline and byte-identically.

Models (``sentiment.candidates``), weights + tokenizer only:

* ``distilbert_sst2`` -- DistilBERT fine-tuned on SST-2 movie reviews (the
  proposal's model, ~268 MB, Apache-2.0).
* ``finbert`` -- ProsusAI/FinBERT, BERT-base further trained on financial text
  and fine-tuned on Financial PhraseBank (~440 MB, CC BY-NC-SA 4.0).

Evaluation data (``sentiment.datasets``):

* ``twitter_financial_news`` -- zeroshot/twitter-financial-news-sentiment
  (MIT): finance news headlines from Twitter labelled bearish / bullish /
  neutral. Neither candidate was trained on it, so it is the **fair** dataset
  used for band calibration and the promotion gate. URLs are stripped (our
  articles contain none).
* ``financial_phrasebank`` -- Financial PhraseBank v1.0 AllAgree (Malo et al.,
  2014; CC BY-NC-SA 3.0). Reported for every candidate, but **FinBERT was
  fine-tuned on it**, so its PhraseBank score is optimistic and never used to
  choose between models. The raw ZIP is parsed directly (the Hub's old
  ``datasets`` loading script is no longer supported).
"""

from __future__ import annotations

import argparse
import io
import json
import re
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
OPTIONAL_MODEL_FILES = ["special_tokens_map.json"]
PHRASEBANK_ZIP = "data/FinancialPhraseBank-v1.0.zip"
TWITTER_LABELS = {0: "negative", 1: "positive", 2: "neutral"}  # bearish, bullish, neutral
_URL = re.compile(r"https?://\S+")
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
        repo_id=repo_id,
        revision=revision,
        local_dir=str(dest),
        allow_patterns=MODEL_FILES + OPTIONAL_MODEL_FILES,
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


def parse_twitter(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalise the Twitter financial-news CSV to ``id, sentence, label``.

    Raises:
        FinSightError: On unexpected columns or labels.
    """
    if not {"text", "label"} <= set(frame.columns):
        raise FinSightError(f"unexpected columns {list(frame.columns)}")
    if not set(frame["label"].unique()) <= set(TWITTER_LABELS):
        raise FinSightError(f"unexpected labels {sorted(frame['label'].unique())}")
    text = frame["text"].astype(str).map(lambda t: " ".join(_URL.sub(" ", t).split()))
    out = pd.DataFrame({"sentence": text, "label": frame["label"].map(TWITTER_LABELS)})
    out = out[out["sentence"].str.len() > 0].drop_duplicates("sentence").reset_index(drop=True)
    out.insert(0, "id", range(len(out)))
    return out


def fetch_twitter(repo_id: str, revision: str, filename: str, dest: Path) -> dict[str, Any]:
    """Download one split of the Twitter financial-news dataset and normalise it."""
    from huggingface_hub import hf_hub_download  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as tmp:
        path = _download(hf_hub_download)(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            revision=revision,
            local_dir=tmp,
        )
        frame = parse_twitter(pd.read_csv(path))
    dest.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(dest, index=False, lineterminator="\n")
    counts = frame["label"].value_counts().sort_index().to_dict()
    log.info("twitter financial news fetched", extra={"rows": len(frame), "labels": counts})
    return {"repo_id": repo_id, "revision": revision, "rows": len(frame), "labels": counts}


def main(argv: Sequence[str] | None = None) -> int:
    """Stage entry point: ``python -m finsight.sentiment.fetch {model <key> | data}``."""
    parser = argparse.ArgumentParser(description="Fetch sentiment models / evaluation data")
    sub = parser.add_subparsers(dest="what", required=True)
    sub.add_parser("model").add_argument("candidate")
    sub.add_parser("data")
    args = parser.parse_args(argv)
    settings = get_settings()
    configure_logging("fetch_sentiment", settings.log_level, settings.root / "logs")
    params = settings.section("sentiment")
    try:
        if args.what == "model":
            cand = params["candidates"][args.candidate]
            with stage_run(f"fetch_sentiment@{args.candidate}", settings) as run:
                result = fetch_model(
                    cand["repo"], cand["revision"], settings.root / cand["base_dir"]
                )
                run.metrics = {"model_size_mb": result["size_mb"]}
        else:
            ds = params["datasets"]
            with stage_run("fetch_sentiment_data", settings) as run:
                pb = ds["financial_phrasebank"]
                tw = ds["twitter_financial_news"]
                result = {
                    "financial_phrasebank": fetch_phrasebank(
                        pb["repo"], pb["revision"], pb["agreement"], settings.root / pb["path"]
                    ),
                    "twitter_financial_news": fetch_twitter(
                        tw["repo"], tw["revision"], tw["file"], settings.root / tw["path"]
                    ),
                }
                run.metrics = {k: v["rows"] for k, v in result.items()}
    except KeyError as exc:
        print(f"fetch_sentiment failed: unknown key {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 -- report any download/parse failure cleanly
        print(f"fetch_sentiment failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
