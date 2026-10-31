r"""Fernet keys and streaming (chunked) file encryption.

**Fernet** (from ``cryptography``) is AES-128-CBC encryption plus an
HMAC-SHA256 tag, with a timestamp; a wrong key or any changed byte makes
decryption fail instead of returning garbage.

Fernet works on whole messages in memory, which is fine for ``.env`` but not
for a backup with model files of several hundred MB. :class:`EncryptingWriter`
therefore splits a stream into 4 MB chunks and encrypts each one separately.
Each chunk's plaintext starts with its sequence number and a "last chunk" flag,
so chunks that are reordered, dropped or cut off at the end are detected, not
just chunks whose bytes were changed. File layout::

    b"FSENC1\n" ( <4-byte big-endian length> <Fernet token> )*

The key lives **outside the repository** (default
``~/.config/finsight/secret.key``, mode 600; override with
``FINSIGHT_KEY_FILE``). Lose it and the encrypted files cannot be recovered,
so keep a second copy somewhere safe (a password manager, a USB stick).
"""

from __future__ import annotations

import io
import os
import stat
import struct
from pathlib import Path
from typing import BinaryIO

from cryptography.fernet import Fernet, InvalidToken

from finsight.common.exceptions import SecurityError

MAGIC = b"FSENC1\n"
CHUNK = 4 * 1024 * 1024
HEADER = struct.Struct(">QB")  # chunk index, last-chunk flag
LENGTH = struct.Struct(">I")
DEFAULT_KEY_FILE = Path("~/.config/finsight/secret.key")


# ---------------------------------------------------------------------- keys
def key_path() -> Path:
    """Where the key is kept (``FINSIGHT_KEY_FILE`` or the default)."""
    return Path(os.environ.get("FINSIGHT_KEY_FILE", DEFAULT_KEY_FILE)).expanduser()


def create_key(path: Path, repo_root: Path | None = None) -> Path:
    """Write a new random key with owner-only permissions; never overwrite one.

    Raises:
        SecurityError: If a key already exists or the path is inside the repository.
    """
    path = path.expanduser().resolve()
    if repo_root is not None and path.is_relative_to(repo_root.resolve()):
        raise SecurityError(f"refusing to keep the key inside the repository ({path})")
    if path.exists():
        raise SecurityError(f"a key already exists at {path}; it was not replaced")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(Fernet.generate_key())
    return path


def load_key(path: Path | None = None) -> Fernet:
    """Read the key and build a :class:`Fernet`.

    Raises:
        SecurityError: If the key is missing, malformed or readable by other users.
    """
    path = (path or key_path()).expanduser()
    if not path.exists():
        raise SecurityError(
            f"no key at {path}; create one with `python -m finsight.security.secrets init-key`"
        )
    if path.stat().st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise SecurityError(f"{path} is readable by other users; run `chmod 600 {path}`")
    try:
        return Fernet(path.read_bytes().strip())
    except ValueError as exc:
        raise SecurityError(f"{path} does not contain a valid Fernet key") from exc


# --------------------------------------------------------------- whole files
def encrypt_bytes(fernet: Fernet, data: bytes) -> bytes:
    """Encrypt a small payload as a chunked stream (same format as big files)."""
    out = io.BytesIO()
    with EncryptingWriter(out, fernet) as writer:
        writer.write(data)
    return out.getvalue()


def decrypt_bytes(fernet: Fernet, blob: bytes) -> bytes:
    """Inverse of :func:`encrypt_bytes` (raises :class:`SecurityError`)."""
    return DecryptingReader(io.BytesIO(blob), fernet).read()


# ------------------------------------------------------------------- streams
class EncryptingWriter(io.RawIOBase):
    """Write-only file object that encrypts into ``sink`` chunk by chunk."""

    def __init__(self, sink: BinaryIO, fernet: Fernet, chunk_size: int = CHUNK) -> None:
        """Start the stream (writes the magic header)."""
        super().__init__()
        self.sink, self.fernet, self.chunk_size = sink, fernet, chunk_size
        self.buffer = bytearray()
        self.index = 0
        self.sink.write(MAGIC)

    def writable(self) -> bool:
        """This stream can be written."""
        return True

    def write(self, data: bytes) -> int:  # type: ignore[override]
        """Buffer ``data`` and emit every full chunk."""
        if self.closed:
            raise ValueError("write to a closed EncryptingWriter")
        self.buffer.extend(data)
        while len(self.buffer) > self.chunk_size:
            self._emit(bytes(self.buffer[: self.chunk_size]), last=False)
            del self.buffer[: self.chunk_size]
        return len(data)

    def _emit(self, plain: bytes, last: bool) -> None:
        token = self.fernet.encrypt(HEADER.pack(self.index, int(last)) + plain)
        self.sink.write(LENGTH.pack(len(token)) + token)
        self.index += 1

    def close(self) -> None:
        """Write the final chunk (always present, possibly empty)."""
        if not self.closed:
            self._emit(bytes(self.buffer), last=True)
            self.buffer.clear()
            self.sink.flush()
        super().close()


class DecryptingReader(io.RawIOBase):
    """Read-only file object over an encrypted stream; verifies every chunk."""

    def __init__(self, source: BinaryIO, fernet: Fernet) -> None:
        """Check the header; chunks are decrypted lazily as data is read."""
        super().__init__()
        self.source, self.fernet = source, fernet
        if source.read(len(MAGIC)) != MAGIC:
            raise SecurityError("not a FinSight encrypted file (bad header)")
        self.pending = memoryview(b"")
        self.index = 0
        self.finished = False

    def readable(self) -> bool:
        """This stream can be read."""
        return True

    def _next_chunk(self) -> None:
        raw_len = self.source.read(LENGTH.size)
        if len(raw_len) < LENGTH.size:
            raise SecurityError("encrypted file is truncated (final chunk missing)")
        token = self.source.read(LENGTH.unpack(raw_len)[0])
        try:
            plain = self.fernet.decrypt(token)
        except InvalidToken as exc:
            raise SecurityError("decryption failed: wrong key or the file was modified") from exc
        index, last = HEADER.unpack(plain[: HEADER.size])
        if index != self.index:
            raise SecurityError(f"chunk {index} found where chunk {self.index} was expected")
        self.index += 1
        self.pending = memoryview(plain)[HEADER.size :]
        if last:
            self.finished = True
            if self.source.read(1):
                raise SecurityError("unexpected data after the final chunk")

    def readinto(self, buffer: bytearray) -> int:  # type: ignore[override]
        """Fill ``buffer`` with decrypted bytes (0 at the end of the stream)."""
        while not self.pending and not self.finished:
            self._next_chunk()
        n = min(len(buffer), len(self.pending))
        buffer[:n] = self.pending[:n]
        self.pending = self.pending[n:]
        return n

    def read(self, size: int = -1) -> bytes:  # type: ignore[override]
        """Read ``size`` bytes, or everything that is left."""
        if size is None or size < 0:
            parts = []
            while chunk := super().read(CHUNK):
                parts.append(chunk)
            return b"".join(parts)
        return super().read(size) or b""
