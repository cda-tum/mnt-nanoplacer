"""Disposable branch-only TLS measurements. Never installs from a test host."""

import argparse
import concurrent.futures
import hashlib
import io
import importlib.metadata
import json
import os
from pathlib import Path
import re
import ssl
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile

HERE = Path(__file__).resolve().parent
HOSTS = json.loads((HERE / "hosts.json").read_text())
CONTROLS = ["github.com", "pypi.org", "files.pythonhosted.org", "probe-emudhra.we-have-a-problem.com"]
OVERRIDES = {
    "NODE_OPTIONS", "NODE_EXTRA_CA_CERTS", "SSL_CERT_FILE", "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "PIP_CERT", "GIT_SSL_CAINFO",
    "UV_SYSTEM_CERTS", "UV_NATIVE_TLS", "PIP_TRUSTED_HOST", "UV_INSECURE_HOST",
    "NODE_TLS_REJECT_UNAUTHORIZED", "GIT_SSL_NO_VERIFY",
}
REJECTIONS = (
    "unable to get local issuer", "unable to get issuer", "unable to verify the first certificate",
    "self-signed certificate", "self signed certificate", "certificate revoked",
    "certificate untrusted", "certificate rejected", "certificate is not trusted",
    "unknownissuer", "unknown issuer", "revoked", "blocked public key", "public key is blocked",
    "compromised", "denylist", "distrusted", "untrustedroot", "cert_e_untrustedroot",
    "server certificate verification failed", "sslverificationerror",
)


def classify_error(message):
    lower = message.lower()
    if any(word in lower for word in ("expired", "not yet valid", "hostname mismatch")):
        return "inconclusive"
    return "rejected" if any(word in lower for word in REJECTIONS) else "inconclusive"


def emit(value):
    print("TLS_DIAGNOSTIC " + json.dumps(value, sort_keys=True), flush=True)


def request(client, host):
    url = f"https://{host}/"
    try:
        if client == "python":
            with urllib.request.urlopen(url, timeout=15) as response:
                body = response.read(8192).decode(errors="replace")
        elif client in ("pip", "requests", "twine", "httpx"):
            if client == "pip":
                from pip._internal.commands import create_command
                command = create_command("index")
                options, _ = command.parse_args(["versions", "certifi", "--retries", "0", "--timeout", "15", "--no-cache-dir"])
                session = command._build_session(options)
            elif client == "twine":
                from twine.repository import Repository
                session = Repository("https://upload.pypi.org/legacy/", "", "", False).session
            elif client == "httpx":
                import httpx
                session = httpx.Client()
            else:
                import requests
                session = requests.Session()
            with session:
                response = session.get(url, timeout=15)
                body = response.text[:8192]
        else:
            raise ValueError(client)
        if "emudhra.we-have-a-problem.com" in host and "Your client successfully connected to the site." not in body:
            return "inconclusive", "Unexpected response body; possible interception"
        return "connected", "HTTPS response received"
    except urllib.error.HTTPError as error:
        return ("connected" if host in CONTROLS else "inconclusive"), f"HTTP {error.code}"
    except Exception as error:
        message = f"{type(error).__name__}: {error}"
        return classify_error(message), message[:2000]


def cli_request(client, host):
    url = f"https://{host}/"
    if client == "curl":
        command = ["curl", "--silent", "--show-error", "--max-time", "20", url]
    elif client == "git":
        url = "https://github.com/cda-tum/mnt-bench.git" if host == "github.com" else url
        command = ["git", "-c", "credential.helper=", "-c", "http.sslVerify=true", "ls-remote", url]
    else:
        # A wheel URL prevents build-backend execution, even if an affected host is trusted.
        command = ["uv", "pip", "install", "--python", sys.executable, "--dry-run", "--no-deps",
                   "--reinstall", "--no-cache", url + "certifi-0.0.0-py3-none-any.whl"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=35)
        output = result.stdout + result.stderr
        if client == "curl":
            if result.returncode == 0:
                if host in CONTROLS or "Your client successfully connected to the site." in output:
                    return "connected", "HTTPS response received"
            return classify_error(output), output[-2000:]
        if client == "git" and (result.returncode == 0 or re.search(r"<= Recv header: HTTP/(?:2|1\.[01]) [2-5]\d\d", output)):
            return "connected", "Verified HTTPS connection established"
        if client == "uv" and ("HTTP status" in output or "Invalid ZIP" in output or "invalid Zip" in output or "Failed to read from zip" in output):
            return "connected", output[-2000:]
        return classify_error(output), output[-2000:]
    except subprocess.TimeoutExpired:
        return "inconclusive", "Process timeout"


def clients(label, names):
    removed = sorted(name for name in OVERRIDES if name in os.environ)
    for name in OVERRIDES:
        os.environ.pop(name, None)
    # Do not allow proxy trust or user package-index configuration to masquerade as the client.
    for name in list(os.environ):
        if name.lower().endswith("_proxy"):
            os.environ.pop(name)
    os.environ.update(PIP_CONFIG_FILE=os.devnull, PIP_DISABLE_PIP_VERSION_CHECK="1",
                      UV_NO_CONFIG="1", UV_HTTP_RETRIES="0", UV_HTTP_TIMEOUT="15",
                      UV_PYTHON_DOWNLOADS="never", GIT_TERMINAL_PROMPT="0", GIT_TRACE_CURL="1", GIT_TRACE_CURL_NO_DATA="1")
    packages = {}
    for name in ("pip", "certifi", "virtualenv", "pre-commit", "requests", "httpx", "twine", "sigstore"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    emit(dict(kind="metadata", label=label, python=sys.version, executable=sys.executable,
              openssl=ssl.OPENSSL_VERSION, trust_paths=ssl.get_default_verify_paths()._asdict(),
              removed_overrides=removed, packages=packages))
    results = []
    for client in names.split(","):
        if client in ("curl", "git", "uv"):
            emit(dict(kind="client-version", client=client, label=label,
                      output=subprocess.check_output([client, "--version"], text=True).strip()))
        probe = cli_request if client in ("curl", "git", "uv") else request
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            values = list(pool.map(lambda host: probe(client, host), CONTROLS + HOSTS))
        for host, (status, detail) in zip(CONTROLS + HOSTS, values):
            row = dict(kind="result", label=label, client=client, host=host, status=status,
                       expected="connected" if host in CONTROLS else "rejected", detail=detail)
            results.append(row)
            emit(row)
    failures = sum(row["status"] != row["expected"] for row in results)
    emit(dict(kind="summary", label=label, measurements=len(results), nonpassing=failures))
    # Evidence collection completes even when trust is unsafe; summaries never call that a pass.
    if "GITHUB_STEP_SUMMARY" in os.environ:
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as summary:
            summary.write(f"\n### {label}\n\n{len(results)} measurements; **{failures} unsafe or inconclusive**.\n")


def isolated():
    root = Path(tempfile.mkdtemp(prefix="tls-isolation-", dir=os.environ.get("RUNNER_TEMP")))
    config = root / "pre-commit.json"
    entry = f'python "{HERE / "probe.py"}" clients --label pre-commit --only python,pip'
    config.write_text(json.dumps({"repos": [{"repo": "local", "hooks": [{
        "id": "temporary-tls-measurement", "name": "Temporary TLS measurement", "entry": entry,
        "language": "python", "pass_filenames": False, "always_run": True}]}]}))
    subprocess.run([sys.executable, "-m", "pre_commit", "run", "--all-files", "--verbose", "--config", str(config)], check=True)
    project = root / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text('[build-system]\nrequires = ["setuptools"]\nbuild-backend = "probe_backend"\nbackend-path = ["."]\n')
    (project / "probe_backend.py").write_text(
        'import os, subprocess, sys\n'
        'def get_requires_for_build_wheel(config_settings=None):\n'
        f'    subprocess.run([sys.executable, {str(HERE / "probe.py")!r}, "clients", "--label", os.environ["PROBE_BUILD_LABEL"], "--only", "python"], check=True)\n'
        '    raise RuntimeError("TLS_DIAGNOSTIC_BUILD_COMPLETE")\n')
    for name, command in (
        ("pip-build-isolation", [sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-cache-dir", str(project)]),
        ("uv-build-isolation", ["uv", "build", "--wheel", "--no-cache", "--python", sys.executable, str(project)]),
    ):
        environment = dict(os.environ, PROBE_BUILD_LABEL=name)
        result = subprocess.run(command, env=environment, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180)
        output = result.stdout + result.stderr
        print(output, flush=True)
        if "TLS_DIAGNOSTIC_BUILD_COMPLETE" not in output or '"kind": "summary"' not in output:
            raise RuntimeError(f"{name} did not reach the isolated probe")


def offline():
    with urllib.request.urlopen("https://emudhra.we-have-a-problem.com/chains.zip", timeout=60) as response:
        archive = response.read()
    assert hashlib.sha256(archive).hexdigest() == "afb9d69e8e5fc9eb0c08d5fe4baafc67edbcced2d63af2f9deb62edb6431d4a9", "Incident chain archive changed"
    root = Path(tempfile.mkdtemp(prefix="tls-chains-"))
    environment = {key: value for key, value in os.environ.items() if key not in OVERRIDES}
    stores = {
        "original-system-store": [],
        "verified-certifi-store": ["-CAfile", os.environ["SSL_CERT_FILE"], "-no-CApath", "-no-CAstore"],
        "embedded-node-root-bundle": ["-CAfile", str(Path(os.environ["RUNNER_TEMP"]) / "node-roots.pem"), "-no-CApath", "-no-CAstore"],
    }
    with zipfile.ZipFile(io.BytesIO(archive)) as archive_zip:
        names = [name for name in archive_zip.namelist() if name.endswith(".pem")]
        assert len(names) == 16 and sum("rollover/" in name for name in names) == 2
        for name in names:
            pem = root / Path(name).name
            pem.write_bytes(archive_zip.read(name))
            for store, options in stores.items():
                result = subprocess.run(["openssl", "verify", "-show_chain", "-purpose", "sslserver", *options, "-untrusted", str(pem), str(pem)], env=environment, capture_output=True, text=True, timeout=20)
                detail = result.stdout + result.stderr
                status = "connected" if result.returncode == 0 else classify_error(detail)
                emit(dict(kind="result", label="offline-store-only", client=store, host=Path(name).name, expected="rejected", status=status, detail=detail))
    print("Offline bundle measurements do not exercise runtime public-key denylists.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["clients", "isolated", "bundled-pip", "offline", "self-check"])
    parser.add_argument("--label", default="selected-python")
    parser.add_argument("--only", default="python,pip")
    args = parser.parse_args()
    if args.mode == "self-check":
        assert len(HOSTS) == len(set(HOSTS)) == 14
        assert classify_error("CERTIFICATE_VERIFY_FAILED: unable to get local issuer certificate") == "rejected"
        assert classify_error("Connection timed out") == "inconclusive"
        assert classify_error("certificate expired") == "inconclusive"
        assert classify_error("UnknownIssuer") == "rejected"
        print("Diagnostic classification checks passed")
    elif args.mode == "isolated":
        isolated()
    elif args.mode == "offline":
        offline()
    elif args.mode == "bundled-pip":
        import ensurepip
        wheels = list((Path(ensurepip.__file__).parent / "_bundled").glob("pip-*.whl"))
        assert len(wheels) == 1, "Cannot identify archived ensurepip wheel"
        environment = dict(os.environ, PYTHONPATH=str(wheels[0]))
        subprocess.run([sys.executable, str(HERE / "probe.py"), "clients", "--label", "archived-ensurepip", "--only", "pip"], env=environment, check=True)
    else:
        clients(args.label, args.only)
