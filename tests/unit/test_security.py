"""Security tests (Day 9): encryption at rest, secrets handling, and the deployment's
security properties (localhost-only ports, non-root containers, no plaintext secrets).
"""

from __future__ import annotations

import io
import os
import re
import stat
import struct
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml
from cryptography.fernet import Fernet

from finsight.common.exceptions import SecurityError
from finsight.security import backup, secrets
from finsight.security.crypto import (
    LENGTH,
    MAGIC,
    DecryptingReader,
    EncryptingWriter,
    create_key,
    decrypt_bytes,
    encrypt_bytes,
    load_key,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def key_file(tmp_path):
    return create_key(tmp_path / "keys" / "secret.key")


@pytest.fixture
def fernet(key_file):
    return load_key(key_file)


# -------------------------------------------------------------------- crypto
@pytest.mark.parametrize("size", [0, 1, 1000, 64 * 1024 + 7])
def test_encrypt_round_trip_across_chunk_boundaries(size):
    f = Fernet(Fernet.generate_key())
    data = os.urandom(size)
    out = io.BytesIO()
    with EncryptingWriter(out, f, chunk_size=16 * 1024) as writer:
        writer.write(data[: size // 2])
        writer.write(data[size // 2 :])
    blob = out.getvalue()
    assert blob.startswith(MAGIC)
    if size >= 16:
        assert data[:16] not in blob  # no plaintext leaks into the file
    assert DecryptingReader(io.BytesIO(blob), f).read() == data


def test_wrong_key_and_modified_bytes_are_refused():
    f = Fernet(Fernet.generate_key())
    blob = encrypt_bytes(f, b"MONGO_ROOT_PASSWORD=hunter2hunter2\n")
    with pytest.raises(SecurityError, match="wrong key"):
        decrypt_bytes(Fernet(Fernet.generate_key()), blob)
    tampered = bytearray(blob)
    tampered[-5] ^= 1
    with pytest.raises(SecurityError, match="modified"):
        decrypt_bytes(f, bytes(tampered))
    with pytest.raises(SecurityError, match="bad header"):
        decrypt_bytes(f, b"plain text")


def _chunks(blob):
    pos, out = len(MAGIC), []
    while pos < len(blob):
        (n,) = LENGTH.unpack(blob[pos : pos + 4])
        out.append(blob[pos : pos + 4 + n])
        pos += 4 + n
    return out


def test_dropped_reordered_or_truncated_chunks_are_detected():
    f = Fernet(Fernet.generate_key())
    out = io.BytesIO()
    with EncryptingWriter(out, f, chunk_size=10) as writer:
        writer.write(b"x" * 35)  # 4 chunks
    parts = _chunks(out.getvalue())
    assert len(parts) == 4
    with pytest.raises(SecurityError, match="truncated"):
        decrypt_bytes(f, MAGIC + b"".join(parts[:3]))  # last chunk cut off
    with pytest.raises(SecurityError, match="expected"):
        decrypt_bytes(f, MAGIC + parts[1] + parts[0] + b"".join(parts[2:]))
    with pytest.raises(SecurityError, match="after the final chunk"):
        decrypt_bytes(f, MAGIC + b"".join(parts) + struct.pack(">I", 0))


def test_key_is_private_never_overwritten_and_never_in_the_repo(tmp_path, key_file):
    assert stat.S_IMODE(key_file.stat().st_mode) == 0o600
    with pytest.raises(SecurityError, match="already exists"):
        create_key(key_file)
    with pytest.raises(SecurityError, match="inside the repository"):
        create_key(tmp_path / "repo" / "secret.key", repo_root=tmp_path / "repo")
    key_file.chmod(0o644)
    with pytest.raises(SecurityError, match="readable by other users"):
        load_key(key_file)
    with pytest.raises(SecurityError, match="no key"):
        load_key(tmp_path / "missing.key")


# ------------------------------------------------------------------- secrets
def test_env_encrypt_decrypt_and_check(tmp_path, key_file, monkeypatch):
    monkeypatch.setattr(secrets, "git_tracks", lambda root, rel: False)
    repo = tmp_path / "repo"
    repo.mkdir()
    env = repo / ".env"
    env.write_text("FINNHUB_API_KEY=abc123\nMONGO_ROOT_PASSWORD=pw\n")
    env.chmod(0o600)
    enc = secrets.encrypt_env(repo, key_file)
    assert b"abc123" not in enc.read_bytes()
    assert all(ok for ok, _ in secrets.check(repo, key_file))
    env.write_text("FINNHUB_API_KEY=changed\n")
    results = secrets.check(repo, key_file)
    assert [ok for ok, msg in results if "decrypts" in msg] == [True]
    assert [ok for ok, msg in results if "matches .env" in msg] == [False]  # re-encrypt needed
    with pytest.raises(SecurityError, match="--force"):
        secrets.decrypt_env(repo, key_file)
    secrets.decrypt_env(repo, key_file, force=True)
    assert env.read_text() == "FINNHUB_API_KEY=abc123\nMONGO_ROOT_PASSWORD=pw\n"
    assert stat.S_IMODE(env.stat().st_mode) == 0o600


# -------------------------------------------------------------------- backup
def _project(tmp_path):
    root = tmp_path / "proj"
    (root / "reports").mkdir(parents=True)
    (root / "reports" / "lstm_metrics.json").write_text('{"mape": 2.9}')
    (root / "params.yaml").write_text("a: 1\n")
    (root / "models" / "lstm").mkdir(parents=True)
    (root / "models" / "lstm" / "model.bin").write_bytes(os.urandom(300_000))
    return root


def test_backup_round_trip_keeps_types_and_files(db, tmp_path, fernet):
    when = datetime(2026, 10, 27, tzinfo=UTC)
    db.prices.insert_one({"ticker": "AAPL", "date": when, "close": 1.5})
    db.monitoring_reports.insert_one({"generated_at": when})
    root = _project(tmp_path)
    target = tmp_path / "out" / "snap.tar.enc"
    meta = backup.create(db, root, fernet, target, include_models=True)
    assert meta["collections"]["prices"] == 1 and "monitoring_reports" not in meta["collections"]
    assert "models/lstm/model.bin" in meta["files"]
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert b"AAPL" not in target.read_bytes() and b"lstm_metrics" not in target.read_bytes()
    assert backup.verify(target, fernet)["files"] == meta["files"]
    restored_db = db.client["finsight_restore"]
    backup.restore(target, fernet, tmp_path / "restored", restored_db)
    doc = restored_db.prices.find_one({}, {"_id": 0})
    assert doc["date"] == when and doc["close"] == 1.5
    assert (tmp_path / "restored" / "params.yaml").read_text() == "a: 1\n"


def test_backup_without_models_and_tampering(db, tmp_path, fernet):
    root = _project(tmp_path)
    target = tmp_path / "snap.tar.enc"
    meta = backup.create(db, root, fernet, target)
    assert not any(name.startswith("models/") for name in meta["files"])
    blob = bytearray(target.read_bytes())
    blob[len(blob) // 2] ^= 0xFF
    target.write_bytes(bytes(blob))
    with pytest.raises(SecurityError):
        backup.verify(target, fernet)


# ------------------------------------------------- deployment security checks
@pytest.fixture(scope="module")
def compose():
    return yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))["services"]


def test_every_published_port_is_bound_to_localhost(compose):
    for name, service in compose.items():
        for port in service.get("ports", []):
            assert str(port).startswith("127.0.0.1:"), (name, port)


def test_images_we_build_run_as_a_non_root_user(compose):
    for name, service in compose.items():
        build = service.get("build")
        if not build:
            continue
        dockerfile = (ROOT / build["context"] / build["dockerfile"]).read_text(encoding="utf-8")
        users = re.findall(r"^USER\s+(\S+)", dockerfile, flags=re.MULTILINE)
        assert users and users[-1] not in {"root", "0"}, name


def test_no_plaintext_secrets_in_compose_and_mongo_requires_auth(compose):
    secret_like = re.compile(r"(PASSWORD|TOKEN|SECRET|API_KEY)", re.IGNORECASE)
    for name, service in compose.items():
        env = service.get("environment") or {}
        for key, value in env.items():
            if secret_like.search(key):
                assert str(value).startswith("${"), (name, key)
    mongo = compose["mongo"]["environment"]
    assert mongo["MONGO_INITDB_ROOT_USERNAME"] and mongo["MONGO_INITDB_ROOT_PASSWORD"].startswith(
        "${"
    )


def test_secrets_and_keys_are_git_ignored_and_backups_too():
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    for pattern in (".env", "*.key", "secrets/.env", "/backups/"):
        assert pattern in ignored, pattern
    example = (ROOT / ".env.example").read_text(encoding="utf-8")
    for line in example.splitlines():
        if re.match(r"^\w*(PASSWORD|TOKEN|API_KEY)=", line):
            assert line.endswith("=replace-me"), line


def test_backend_cors_is_not_open_to_everyone():
    from finsight.api.config import ApiConfig  # noqa: PLC0415

    assert "*" not in ApiConfig().frontend_origins


def test_no_secret_values_committed_in_tracked_text_files():
    """A cheap scan for real-looking keys in the files we ship."""
    pattern = re.compile(
        r"(api[_-]?key|password|token)\s*[:=]\s*['\"]?([A-Za-z0-9_\-]{20,})", re.IGNORECASE
    )
    allowed = {"replace-me"}
    hits = []
    for path in [
        *ROOT.glob("*.y*ml"),
        *ROOT.glob("*.toml"),
        *(ROOT / "src").rglob("*.py"),
        *(ROOT / "monitoring").rglob("*.yml"),
    ]:
        for match in pattern.finditer(path.read_text(encoding="utf-8", errors="ignore")):
            if match.group(2) not in allowed and "environ" not in match.group(0):
                hits.append(f"{path.relative_to(ROOT)}: {match.group(0)[:40]}")
    assert not hits, hits


def test_encrypted_env_is_the_committed_form():
    enc = ROOT / "secrets" / ".env.enc"
    if enc.exists():
        assert enc.read_bytes().startswith(MAGIC)
