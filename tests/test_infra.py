import os
import socket

import pytest


@pytest.mark.parametrize("port", [5432, 6379, 6333, 7700])
def test_service_reachable_on_localhost(port):
    with socket.create_connection(("127.0.0.1", port), timeout=2):
        pass


@pytest.mark.parametrize("port", [5432, 6333, 7700])
def test_service_not_exposed_externally(port):
    """Regression guard: a compose edit must not republish these to 0.0.0.0."""
    public_ip = os.environ.get("MOC_PUBLIC_IP")
    if not public_ip:
        pytest.skip("MOC_PUBLIC_IP not set")
    with pytest.raises((ConnectionRefusedError, TimeoutError, OSError)):
        socket.create_connection((public_ip, port), timeout=2)


# ─────────────────────── what compose publishes ───────────────────────
#
# CLAUDE.md: compose ports bind to 127.0.0.1 only, because this is a public
# VPS. Task 39 added the one deliberate exception — a webhook is a URL a
# vendor's servers have to reach, so something must answer on 443 — and an
# exception that lives only in a comment is a rule that erodes. These assert
# it about the file, so they hold whether or not the stack is running.


def _compose() -> dict:
    import pathlib

    import yaml

    root = pathlib.Path(__file__).resolve().parents[1]
    return yaml.safe_load((root / "compose.yaml").read_text(encoding="utf-8"))


#: The only service allowed to listen outside loopback, and the reason: TLS
#: termination for the webhook. Anything else publishing publicly is a backing
#: store or an application process exposed to the internet directly.
PUBLIC_SERVICE = "caddy"


def test_only_the_reverse_proxy_publishes_outside_loopback():
    offenders = {}
    for name, service in _compose()["services"].items():
        if name == PUBLIC_SERVICE:
            continue
        for published in service.get("ports", []) or []:
            if not str(published).startswith("127.0.0.1:"):
                offenders.setdefault(name, []).append(published)
    assert offenders == {}, (
        f"{offenders} publish outside loopback on a public VPS. Only "
        f"{PUBLIC_SERVICE!r} may, and only because a webhook URL has to be "
        "reachable by the vendor."
    )


def test_the_application_containers_do_not_inherit_the_hosts_loopback_addresses():
    """`.env` points at 127.0.0.1 because that is where the host reaches the
    stores. A container inheriting that resolves 127.0.0.1 to itself and
    reports the database as down — which reads as an outage rather than as
    configuration, and costs an hour on the wrong thing."""
    services = _compose()["services"]
    for name in ("api", "worker-inbound", "worker-outbound"):
        environment = services[name].get("environment") or {}
        assert environment.get("MOC_PG_HOST") == "postgres", (
            f"{name} would resolve the database to its own loopback"
        )
        assert environment.get("MOC_VALKEY_HOST") == "valkey"
