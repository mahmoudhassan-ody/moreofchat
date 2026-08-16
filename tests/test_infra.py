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
