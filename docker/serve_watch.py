"""Serve the @production model and restart when @production moves (Day 7).

``mlflow models serve`` loads a model once, at start-up. After the promotion
gate (or a rollback) moves ``@production`` to another version, the container
would keep serving the old one until someone restarted it. This wrapper:

1. resolves ``models:/<name>@<alias>`` to a concrete version,
2. runs ``mlflow models serve`` on exactly that version,
3. checks the alias every ``WATCH_INTERVAL_S`` seconds (default 60), and
4. exits when the alias points elsewhere -- compose's ``restart:
   unless-stopped`` then starts the container again, which loads the new model.

If the registry is unreachable, it keeps serving what it has (graceful
degradation) and checks again later. It also exits if the server process dies,
so the container restarts instead of sitting there broken.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import time

from mlflow import MlflowClient

URI = os.environ["MODEL_URI"]  # e.g. models:/finsight-lstm@production
PORT = os.environ.get("PORT", "5001")
INTERVAL = float(os.environ.get("WATCH_INTERVAL_S", "60"))


def log(message: str) -> None:
    """Print a tagged line (the container log)."""
    print(f"[serve-watch] {message}", flush=True)


def serve_argv(model: str) -> list[str]:
    """``mlflow models serve`` for one model, on all interfaces of the container."""
    host = "0.0.0.0"  # noqa: S104 -- inside the container; compose binds the port to localhost
    return ["mlflow", "models", "serve", "-m", model, "--env-manager", "local",
            "-h", host, "-p", PORT, "--workers", "1"]  # fmt: skip


def resolve(client: MlflowClient, name: str, alias: str) -> str:
    """The version number an alias points to."""
    return str(client.get_model_version_by_alias(name, alias).version)


def main() -> int:
    """Serve, watch the alias, exit when it moves."""
    match = re.fullmatch(r"models:/([^@/]+)@([\w-]+)", URI)
    if not match:  # a fixed version or a path: just serve it, nothing to watch
        os.execvp("mlflow", serve_argv(URI))  # noqa: S606,S607
    name, alias = match.groups()
    client = MlflowClient()
    for attempt in range(30):  # the registry may still be starting
        try:
            version = resolve(client, name, alias)
            break
        except Exception as exc:  # noqa: BLE001
            log(f"cannot resolve {URI} yet ({exc.__class__.__name__}); retry {attempt + 1}/30")
            time.sleep(10)
    else:
        log("registry unreachable; giving up so the container restarts")
        return 1
    log(f"serving {name} version {version} (@{alias}) on port {PORT}")
    server = subprocess.Popen(serve_argv(f"models:/{name}/{version}"))  # noqa: S603

    def stop(*_: object) -> None:
        server.terminate()
        try:
            server.wait(timeout=20)
        except subprocess.TimeoutExpired:
            server.kill()

    def on_sigterm(*_: object) -> None:
        stop()
        sys.exit(0)

    signal.signal(signal.SIGTERM, on_sigterm)
    while True:
        time.sleep(INTERVAL)
        if server.poll() is not None:
            log(f"model server exited with code {server.returncode}; restarting the container")
            return 1
        try:
            current = resolve(client, name, alias)
        except Exception as exc:  # noqa: BLE001 -- keep serving through registry outages
            log(f"registry check failed ({exc.__class__.__name__}); still serving v{version}")
            continue
        if current != version:
            log(f"@{alias} moved v{version} -> v{current}; restarting to load it")
            stop()
            return 0


if __name__ == "__main__":
    sys.exit(main())
