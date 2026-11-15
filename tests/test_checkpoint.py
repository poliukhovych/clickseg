import ssl
from pathlib import Path

import pytest

from clickseg.checkpoint import download_ssl_context, verify_checkpoint


def test_invalid_checkpoint_is_rejected_before_model_loading(tmp_path: Path):
    path = tmp_path / "mobile_sam.pt"
    path.write_bytes(b"not a checkpoint")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_checkpoint(path)


def test_download_trusts_public_roots_when_system_ca_locations_are_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("SSL_CERT_FILE", str(tmp_path / "missing-ca.pem"))
    monkeypatch.setenv("SSL_CERT_DIR", str(tmp_path))
    assert ssl.create_default_context().cert_store_stats()["x509_ca"] == 0
    context = download_ssl_context()
    assert context.cert_store_stats()["x509_ca"] > 0
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname is True
