"""Download and verify the official checkpoint from the pinned upstream revision."""

import hashlib
import os
import ssl
import tempfile
import urllib.request
from pathlib import Path

import certifi

UPSTREAM_REV = "f706ad9c4eb7f219c00d9050e46328518ffb65d2"
CHECKPOINT_SHA256 = "6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f"
CHECKPOINT_URL = (
    f"https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/{UPSTREAM_REV}"
    "/weights/mobile_sam.pt"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {path}. Run: uv run clickseg download")
    if sha256_file(path) != CHECKPOINT_SHA256:
        raise ValueError(
            f"Checkpoint SHA-256 mismatch: {path}. Expected official MobileSAM weights."
        )


def download_ssl_context() -> ssl.SSLContext:
    """Keep system trust settings and add bundled public roots for Python on macOS."""
    context = ssl.create_default_context()
    context.load_verify_locations(cafile=certifi.where())
    return context


def download_checkpoint(path: Path) -> Path:
    if path.exists():
        verify_checkpoint(path)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="mobile-sam-", suffix=".part", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as output:
            request = urllib.request.Request(CHECKPOINT_URL, headers={"User-Agent": "clickseg/0.1"})
            with urllib.request.urlopen(
                request, timeout=60, context=download_ssl_context()
            ) as response:
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    output.write(chunk)
        verify_checkpoint(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path
