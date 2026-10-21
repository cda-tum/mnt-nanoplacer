"""Check the two CI certificate bundles after bootstrapping patched pip.

This inspects trust anchors, not Node, uv, containers, or live TLS handshakes.
"""

import hashlib
import os
import ssl

from pip._vendor import certifi

# https://emudhra.we-have-a-problem.com/roots.json (including Sectigo trust paths)
BLOCKED_ROOTS = {
    "7dd78d5f4f13459a83dff9abbba62edbaf6f2d102bf257fd712f4d9f2746ed8d",
    "f2edaaff3d9b701fb4d14c9f285f8c0bad90fc3bcc1fa07d47f4718edaba34a5",
    "d7a7a0fb5d7e2731d771e9484ebcdef71d5f0c3e0a2948782bc83ee0ea699ef4",
    "86a1ecba089c4a8d3bbe2734c612ba341d813e043cf9e8a862cd5c57a36bbe6b",
}


def check_bundle(label: str, filename: str) -> None:
    try:
        roots = ssl.create_default_context(cafile=filename).get_ca_certs(binary_form=True)
    except OSError as error:
        message = f"{label}: cannot load CA bundle {filename}: {error}"
        raise SystemExit(message) from error
    if not roots:
        message = f"{label}: CA bundle is empty"
        raise SystemExit(message)
    blocked = BLOCKED_ROOTS & {hashlib.sha256(root).hexdigest() for root in roots}
    if blocked:
        message = f"{label}: affected eMudhra trust anchors remain: {sorted(blocked)}"
        raise SystemExit(message)
    print(f"{label}: trust check passed ({len(roots)} certificates)")


def main() -> None:
    bundle = os.environ.get("SSL_CERT_FILE")
    if not bundle:
        message = "SSL_CERT_FILE must explicitly select the CI system bundle"
        raise SystemExit(message)
    check_bundle("System bundle", bundle)
    # pip combines system roots with its own vendored certifi; check both separately.
    check_bundle("pip bundled certifi", certifi.where())


if __name__ == "__main__":
    main()
