"""Atomic experiment artifacts and verified HTTPS downloads."""

import json
import os
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from clickseg.checkpoint import download_ssl_context


def progress(message):
    print(message, file=sys.stderr, flush=True)


def atomic_json(path: Path, value):
    atomic_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode())


def atomic_bytes(path: Path, value: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(value)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def fetch(url: str, path: Path):
    """Reuse existing files; consumers validate ZIP CRC, images, and recorded hashes."""
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    context = download_ssl_context()
    for attempt in range(3):
        fd, name = tempfile.mkstemp(dir=path.parent, suffix=".part")
        temporary = Path(name)
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "clickseg/0.2"})
            with os.fdopen(fd, "wb") as output:
                with urllib.request.urlopen(request, timeout=60, context=context) as response:
                    expected = int(response.headers.get("Content-Length", 0))
                    received = 0
                    milestone = 32 * 1024 * 1024
                    for chunk in iter(lambda: response.read(1024 * 1024), b""):
                        output.write(chunk)
                        received += len(chunk)
                        if received >= milestone:
                            progress(f"{path.name}: {received / 1024**2:.0f} MiB downloaded")
                            milestone += 32 * 1024 * 1024
                    if expected and received != expected:
                        raise OSError(f"Incomplete download: {path.name}")
            os.replace(temporary, path)
            return
        except (OSError, urllib.error.URLError):
            if attempt == 2:
                raise
            time.sleep(attempt + 1)
        finally:
            temporary.unlink(missing_ok=True)
