"""Encrypted snapshots at rest: MongoDB, reports, configuration and (optionally) models.

    python -m finsight.security.backup create                   # data + reports + config
    python -m finsight.security.backup create --include-models  # + models/ (LSTM, sentiment)
    python -m finsight.security.backup list
    python -m finsight.security.backup verify backups/<file>.tar.enc
    python -m finsight.security.backup restore backups/<file>.tar.enc --to /tmp/finsight-restore
    python -m finsight.security.backup restore <file> --to <dir> --mongo-db finsight_restore

A snapshot is a gzipped tar streamed straight through
:class:`~finsight.security.crypto.EncryptingWriter`, so no unencrypted copy is
ever written to disk. It contains:

* ``mongo/<collection>.jsonl``: every document, as MongoDB Extended JSON
  (keeps dates and ObjectIds exact);
* ``reports/``, ``params.yaml``, ``dvc.lock``, ``configs/``: what produced the data;
* ``models/`` with ``--include-models``: the trained artifacts DVC tracks;
* ``manifest.json``: SHA-256 and size of every file, checked by ``verify``.

``restore`` never touches the live database: it unpacks into a directory and,
if asked, loads the collections into a *separate* database to inspect.
The key is the one ``finsight.security.secrets`` uses.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import tarfile
import time
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bson import json_util
from cryptography.fernet import Fernet

from finsight.common.config import get_settings
from finsight.common.exceptions import FinSightError, SecurityError
from finsight.common.mongo import get_client
from finsight.security.crypto import DecryptingReader, EncryptingWriter, load_key

BACKUP_DIR = Path("backups")
PROJECT_FILES = ("reports", "params.yaml", "dvc.lock", "configs")
SKIP_COLLECTIONS = ("monitoring_reports",)  # recomputed every 2 minutes; 30-day TTL


def _add_bytes(tar: tarfile.TarFile, name: str, data: bytes, manifest: dict[str, Any]) -> None:
    info = tarfile.TarInfo(name)
    info.size, info.mtime, info.mode = len(data), int(time.time()), 0o600
    tar.addfile(info, io.BytesIO(data))
    manifest[name] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def _add_file(tar: tarfile.TarFile, path: Path, name: str, manifest: dict[str, Any]) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    info = tar.gettarinfo(str(path), arcname=name)
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    with path.open("rb") as handle:
        tar.addfile(info, handle)
    manifest[name] = {"sha256": digest.hexdigest(), "bytes": info.size}


def _walk(root: Path, entries: Iterable[str]) -> Iterator[Path]:
    for entry in entries:
        path = root / entry
        if path.is_file():
            yield path
        elif path.is_dir():
            yield from sorted(p for p in path.rglob("*") if p.is_file() and not p.is_symlink())


def dump_collections(db: Any) -> Iterator[tuple[str, bytes, int]]:
    """``(collection, jsonl bytes, count)`` for every collection worth keeping."""
    for name in sorted(db.list_collection_names()):
        if name in SKIP_COLLECTIONS or name.startswith("system."):
            continue
        lines = [
            json_util.dumps(doc, json_options=json_util.CANONICAL_JSON_OPTIONS)
            for doc in db[name].find()
        ]
        yield name, ("\n".join(lines) + "\n").encode() if lines else b"", len(lines)


def create(
    db: Any, root: Path, fernet: Fernet, target: Path, include_models: bool = False
) -> dict[str, Any]:
    """Write one encrypted snapshot to ``target``; return its manifest."""
    manifest: dict[str, Any] = {}
    counts: dict[str, int] = {}
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    with partial.open("wb") as sink, EncryptingWriter(sink, fernet) as enc:
        with tarfile.open(fileobj=enc, mode="w|gz") as tar:
            for name, data, count in dump_collections(db):
                _add_bytes(tar, f"mongo/{name}.jsonl", data, manifest)
                counts[name] = count
            entries = [*PROJECT_FILES, *(("models",) if include_models else ())]
            for path in _walk(root, entries):
                _add_file(tar, path, path.relative_to(root).as_posix(), manifest)
            meta = {
                "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "database": db.name,
                "collections": counts,
                "include_models": include_models,
                "files": manifest,
            }
            _add_bytes(tar, "manifest.json", json.dumps(meta, indent=2).encode(), {})
    partial.replace(target)
    target.chmod(0o600)
    return meta


def _open(path: Path, fernet: Fernet) -> tarfile.TarFile:
    reader = DecryptingReader(path.open("rb"), fernet)
    return tarfile.open(fileobj=io.BufferedReader(reader, buffer_size=1 << 20), mode="r|gz")


def verify(path: Path, fernet: Fernet) -> dict[str, Any]:
    """Decrypt the whole snapshot and check every file against the manifest.

    Raises:
        SecurityError: Wrong key, tampering, or a file that does not match.
    """
    seen: dict[str, str] = {}
    meta: dict[str, Any] | None = None
    try:
        with _open(path, fernet) as tar:
            for member in tar:
                handle = tar.extractfile(member)
                if handle is None:
                    continue
                if member.name == "manifest.json":
                    meta = json.loads(handle.read())
                    continue
                digest = hashlib.sha256()
                while block := handle.read(1 << 20):
                    digest.update(block)
                seen[member.name] = digest.hexdigest()
    except (tarfile.TarError, OSError, EOFError) as exc:
        raise SecurityError(f"{path} is not a readable snapshot: {exc}") from exc
    if meta is None:
        raise SecurityError("snapshot has no manifest.json")
    expected = {name: info["sha256"] for name, info in meta["files"].items()}
    if expected != seen:
        bad = sorted(set(expected.items()) ^ set(seen.items()))
        raise SecurityError(f"snapshot does not match its manifest: {bad[:3]}")
    return meta


def restore(path: Path, fernet: Fernet, to: Path, db: Any | None = None) -> dict[str, Any]:
    """Unpack into ``to`` (and load collections into ``db`` when given)."""
    meta = verify(path, fernet)
    to.mkdir(parents=True, exist_ok=True)
    with _open(path, fernet) as tar:
        tar.extractall(to, filter="data")  # refuses absolute paths, "..", devices, links out
    if db is not None:
        for name in meta["collections"]:
            text = (to / "mongo" / f"{name}.jsonl").read_text(encoding="utf-8")
            docs = [json_util.loads(line) for line in text.splitlines() if line.strip()]
            db[name].drop()
            if docs:
                db[name].insert_many(docs)
    return meta


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="FinSight encrypted backups")
    parser.add_argument("--key-file", type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("create", help="write a new encrypted snapshot to backups/")
    make.add_argument("--include-models", action="store_true", help="also models/ (can be large)")
    sub.add_parser("list", help="snapshots in backups/")
    check = sub.add_parser("verify", help="decrypt and check a snapshot")
    check.add_argument("file", type=Path)
    back = sub.add_parser("restore", help="unpack a snapshot (never into the live database)")
    back.add_argument("file", type=Path)
    back.add_argument("--to", type=Path, required=True, help="directory to unpack into")
    back.add_argument("--mongo-db", help="also load the collections into this (separate) database")
    args = parser.parse_args(argv)

    settings = get_settings()
    root = settings.root
    try:
        if args.command == "list":
            for path in sorted((root / BACKUP_DIR).glob("*.tar.enc")):
                print(f"{path.relative_to(root)}  {path.stat().st_size / 1e6:8.1f} MB")
            return 0
        fernet = load_key(args.key_file)
        if args.command == "create":
            client = get_client(settings)
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            target = root / BACKUP_DIR / f"finsight-{stamp}.tar.enc"
            meta = create(client[settings.mongo_db], root, fernet, target, args.include_models)
            docs = sum(meta["collections"].values())
            size_mb = target.stat().st_size / 1e6
            print(
                f"wrote {target.relative_to(root)} ({size_mb:.1f} MB, "
                f"{len(meta['collections'])} collections, {docs} documents, "
                f"{len(meta['files'])} files)"
            )
        elif args.command == "verify":
            meta = verify(args.file, fernet)
            print(
                f"OK: {len(meta['files'])} files match the manifest; created {meta['created_at']}"
            )
        else:
            if args.mongo_db == settings.mongo_db:
                print("error: --mongo-db must not be the live database", file=sys.stderr)
                return 1
            db = None
            if args.mongo_db:
                db = get_client(settings)[args.mongo_db]
            meta = restore(args.file, fernet, args.to, db)
            loaded = ""
            if db is not None:
                loaded = f" and {len(meta['collections'])} collections into {args.mongo_db}"
            print(f"restored {len(meta['files'])} files into {args.to}{loaded}")
    except (FinSightError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
