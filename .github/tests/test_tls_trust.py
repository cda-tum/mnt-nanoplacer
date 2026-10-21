"""Offline checks using public affected roots and an unaffected ISRG root."""

import hashlib
import importlib
import json
import runpy
import ssl
from pathlib import Path
from unittest.mock import Mock

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_tls_trust.py"
FIXTURES = json.loads((Path(__file__).parent / "fixtures" / "tls_roots.json").read_text(encoding="utf-8"))


@pytest.fixture
def bundles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    paths = {label: tmp_path / f"{label}.pem" for label in ("system", "pip")}
    for path in paths.values():
        path.write_text(FIXTURES["unaffected"], encoding="utf-8")
    monkeypatch.setenv("SSL_CERT_FILE", str(paths["system"]))
    monkeypatch.setattr(importlib.import_module("pip._vendor.certifi"), "where", lambda: str(paths["pip"]))
    return paths


def test_tls_guard_accepts_clean_bundles(bundles: dict[str, Path], capsys: pytest.CaptureFixture[str]) -> None:
    assert all(path.is_file() for path in bundles.values())
    runpy.run_path(str(SCRIPT), run_name="__main__")
    output = capsys.readouterr().out
    assert "System bundle: trust check passed" in output
    assert "pip bundled certifi: trust check passed" in output


@pytest.mark.parametrize("store", ["system", "pip"])
@pytest.mark.parametrize("root", FIXTURES["affected"], ids=lambda root: root["name"])
def test_tls_guard_rejects_each_affected_root(bundles: dict[str, Path], store: str, root: dict[str, str]) -> None:
    # Retain a legitimate CA to prove that a mixed bundle is still rejected.
    bundles[store].write_text(FIXTURES["unaffected"] + root["pem"], encoding="utf-8")
    with pytest.raises(SystemExit, match=root["sha256"]):
        runpy.run_path(str(SCRIPT), run_name="__main__")


@pytest.mark.parametrize("store", ["system", "pip"])
@pytest.mark.parametrize("contents", ["", "not a certificate", None], ids=["empty", "malformed", "missing"])
def test_tls_guard_rejects_unusable_bundles(bundles: dict[str, Path], store: str, contents: str | None) -> None:
    if contents is None:
        bundles[store].unlink()
    else:
        bundles[store].write_text(contents, encoding="utf-8")
    label = "System bundle" if store == "system" else "pip bundled certifi"
    with pytest.raises(SystemExit, match=f"{label}: cannot load CA bundle"):
        runpy.run_path(str(SCRIPT), run_name="__main__")


def test_tls_guard_requires_explicit_system_bundle(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    with pytest.raises(SystemExit, match="SSL_CERT_FILE must explicitly select"):
        runpy.run_path(str(SCRIPT), run_name="__main__")


def test_tls_guard_rejects_store_with_no_ca_certificates(monkeypatch: pytest.MonkeyPatch) -> None:
    guard = runpy.run_path(str(SCRIPT))
    monkeypatch.setattr(ssl, "create_default_context", Mock(return_value=Mock(get_ca_certs=Mock(return_value=[]))))
    with pytest.raises(SystemExit, match="CA bundle is empty"):
        guard["check_bundle"]("Empty store", "empty.pem")


def test_tls_guard_fingerprints_match_public_fixtures() -> None:
    guard = runpy.run_path(str(SCRIPT))
    fingerprints = {hashlib.sha256(ssl.PEM_cert_to_DER_cert(root["pem"])).hexdigest() for root in FIXTURES["affected"]}
    assert fingerprints == guard["BLOCKED_ROOTS"]
