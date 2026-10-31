"""Encrypted copy of ``.env`` that is safe to commit: ``secrets/.env.enc``.

    python -m finsight.security.secrets init-key     # once: ~/.config/finsight/secret.key
    python -m finsight.security.secrets encrypt      # .env -> secrets/.env.enc (commit this)
    python -m finsight.security.secrets decrypt      # secrets/.env.enc -> .env (new machine)
    python -m finsight.security.secrets check        # key, permissions, git, in sync?

``.env`` holds the API keys and passwords and is git-ignored. The encrypted
copy travels with the repository, so a fresh clone plus the key (kept outside
the repository, never committed) restores every secret. ``decrypt`` refuses to
overwrite an existing ``.env`` unless ``--force`` is given, and writes it with
owner-only permissions.
"""

from __future__ import annotations

import argparse
import os
import stat
import subprocess
import sys
from pathlib import Path

from finsight.common.config import find_project_root
from finsight.common.exceptions import SecurityError
from finsight.security.crypto import (
    create_key,
    decrypt_bytes,
    encrypt_bytes,
    key_path,
    load_key,
)

ENC_PATH = Path("secrets/.env.enc")


def encrypt_env(root: Path, key_file: Path | None = None) -> Path:
    """Encrypt ``root/.env`` into ``root/secrets/.env.enc``."""
    source = root / ".env"
    if not source.exists():
        raise SecurityError(f"{source} does not exist")
    blob = encrypt_bytes(load_key(key_file), source.read_bytes())
    target = root / ENC_PATH
    target.parent.mkdir(exist_ok=True)
    target.write_bytes(blob)
    return target


def decrypt_env(root: Path, key_file: Path | None = None, force: bool = False) -> Path:
    """Decrypt ``secrets/.env.enc`` back into ``.env`` (mode 600)."""
    blob = (root / ENC_PATH).read_bytes()
    plain = decrypt_bytes(load_key(key_file), blob)
    target = root / ".env"
    if target.exists() and not force:
        raise SecurityError(f"{target} exists; use --force to replace it")
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(plain)
    target.chmod(0o600)
    return target


def git_tracks(root: Path, relative: str) -> bool:
    """Whether git tracks ``relative`` (False outside a git checkout)."""
    try:
        result = subprocess.run(  # noqa: S603
            ["git", "ls-files", "--error-unmatch", relative],  # noqa: S607
            cwd=root,
            capture_output=True,
            check=False,
        )
    except OSError:
        return False
    return result.returncode == 0


def check(root: Path, key_file: Path | None = None) -> list[tuple[bool, str]]:
    """Every check as ``(ok, message)``; nothing is changed."""
    results: list[tuple[bool, str]] = []
    path = (key_file or key_path()).expanduser().resolve()
    results.append(
        (not path.is_relative_to(root.resolve()), f"key kept outside the repository ({path})")
    )
    try:
        fernet = load_key(key_file)
        results.append((True, "key present with owner-only permissions"))
    except SecurityError as exc:
        return [*results, (False, str(exc))]
    env = root / ".env"
    if env.exists():
        mode = env.stat().st_mode
        results.append(
            (not mode & (stat.S_IRWXG | stat.S_IRWXO), ".env readable by you only (chmod 600 .env)")
        )
        results.append((not git_tracks(root, ".env"), ".env is not tracked by git"))
    enc = root / ENC_PATH
    if not enc.exists():
        results.append((False, f"{ENC_PATH} missing: run `encrypt`"))
        return results
    try:
        plain = decrypt_bytes(fernet, enc.read_bytes())
        results.append((True, f"{ENC_PATH} decrypts with this key"))
    except SecurityError as exc:
        return [*results, (False, f"{ENC_PATH}: {exc}")]
    if env.exists():
        results.append(
            (
                plain == env.read_bytes(),
                f"{ENC_PATH} matches .env (re-run `encrypt` after editing .env)",
            )
        )
    return results


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="FinSight secrets (.env encryption)")
    parser.add_argument(
        "--key-file", type=Path, help="default: $FINSIGHT_KEY_FILE or ~/.config/finsight/secret.key"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init-key", help="create a new key (never overwrites)")
    sub.add_parser("encrypt", help=".env -> secrets/.env.enc")
    dec = sub.add_parser("decrypt", help="secrets/.env.enc -> .env")
    dec.add_argument("--force", action="store_true", help="replace an existing .env")
    sub.add_parser("check", help="key, permissions, git and sync checks")
    args = parser.parse_args(argv)
    root = find_project_root()
    try:
        if args.command == "init-key":
            path = create_key(args.key_file or key_path(), repo_root=root)
            print(f"new key written to {path} (mode 600). Keep a second copy somewhere safe:")
            print("without it, secrets/.env.enc and the encrypted backups cannot be decrypted.")
        elif args.command == "encrypt":
            print(f"wrote {encrypt_env(root, args.key_file).relative_to(root)} (safe to commit)")
        elif args.command == "decrypt":
            print(
                f"wrote {decrypt_env(root, args.key_file, args.force).relative_to(root)} (mode 600)"
            )
        else:
            results = check(root, args.key_file)
            for ok, message in results:
                print(f"{'OK  ' if ok else 'FAIL'}  {message}")
            return 0 if all(ok for ok, _ in results) else 1
    except (SecurityError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
