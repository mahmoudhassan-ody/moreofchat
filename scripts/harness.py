"""Starting the real programs, for the scripts that drive them.

Shared by `drive_the_path.py` and `rehearse.py`. Extracted rather than copied:
two scripts starting the same three processes two ways is two places for
"the worker never came up" to mean different things, and the second copy is
always the one that drifts.

Nothing here is imported by `moc`. These are operator tools that happen to be
written in Python, and the thing they are testing is the package.
"""

import json
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread
from urllib.parse import parse_qsl

import yaml

ROOT = Path(__file__).resolve().parents[1]


class Vendor:
    """Stands in for api.twilio.com and messaging.twilio.com, on loopback.

    Two hosts in the real config and therefore two shapes here: replies arrive
    form-encoded on the message host, typing indicators as JSON on the
    messaging host. Recorded separately, because an indicator counted as a
    reply would make a turn that only said "typing" look answered.
    """

    def __init__(self) -> None:
        self.messages: list[dict] = []
        self.indicators: list[dict] = []
        self._server: HTTPServer | None = None
        self.port = free_port()

    def start(self) -> None:
        vendor = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 - the base class's shape
                length = int(self.headers.get("content-length", 0))
                body = self.rfile.read(length).decode("utf-8")
                if self.headers.get("content-type", "").startswith("application/json"):
                    vendor.indicators.append(json.loads(body))
                    payload = json.dumps({"success": True}).encode()
                    status = 200
                else:
                    vendor.messages.append(dict(parse_qsl(body)))
                    payload = json.dumps({"sid": "SM-stub", "status": "queued"}).encode()
                    status = 201
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args) -> None:  # noqa: A002 - silence the default
                return

        self._server = HTTPServer(("127.0.0.1", self.port), Handler)
        Thread(target=self._server.serve_forever, daemon=True).start()

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()

    def wait_for_message(self, *, seconds: float = 90.0) -> dict | None:
        """The next reply, or None if the turn produced nothing in time.

        The caller has to treat None as a finished turn rather than waiting
        again: a turn that died produces no reply ever, and a caller that keeps
        waiting matches the *next* turn's reply to this one and shifts every
        line after it.
        """
        before = len(self.messages)
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if len(self.messages) > before:
                return self.messages[-1]
            time.sleep(0.25)
        return None


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def isolate_queues(root: Path, tag: str) -> None:
    """Give this run its own streams, in its own copy of the config.

    A rehearsal must not share a consumer group with anything, and "anything"
    includes the workers of a *previous* rehearsal: a run killed by a timeout
    leaves its children behind, nothing reaps them, and they keep consuming.
    Two generations then answer alternate messages and the older one wins some,
    so a run reports a defect that was fixed an hour earlier — which cost two
    rounds of diagnosis before the orphans were spotted.

    It is also what a deploy that forgets to stop the old containers does to
    real customers, which is the version that matters.
    """
    queues = root / "workers" / "queues.yaml"
    document = yaml.safe_load(queues.read_text(encoding="utf-8"))
    for name in ("inbound", "outbound"):
        section = document[name]
        section["stream"] = f"{section['stream']}:{tag}"
        section["group"] = f"{section['group']}-{tag}"
        section["dead_letter_stream"] = f"{section['dead_letter_stream']}:{tag}"
    document["idempotency"]["key_prefix"] = (
        f"{document['idempotency']['key_prefix']}{tag}:"
    )
    queues.write_text(yaml.safe_dump(document, allow_unicode=True), encoding="utf-8")


def config_tree_pointing_at(vendor_port: int) -> Path:
    """A copy of `config/` with both Twilio hosts swapped for the stub.

    Through `MOC_CONFIG_DIR`, which exists precisely so nothing has to be
    patched at runtime. The adapter, the signature, the form encoding and the
    error handling are all the real ones; only the hostnames differ.

    **Both hosts.** The typing indicator is a different host in the real
    config, so a copy that rewrote only the message base left every run quietly
    posting to the real api.twilio.com with fixture credentials — a live
    external call from a script whose whole point is that nothing leaves the
    machine.
    """
    root = Path(tempfile.mkdtemp(prefix="moc-harness-")) / "config"
    shutil.copytree(ROOT / "config", root)
    whatsapp = root / "channels" / "whatsapp.yaml"
    document = yaml.safe_load(whatsapp.read_text(encoding="utf-8"))
    document["api_base"] = f"http://127.0.0.1:{vendor_port}"
    document["typing_indicator"]["api_base"] = f"http://127.0.0.1:{vendor_port}"
    whatsapp.write_text(yaml.safe_dump(document, allow_unicode=True), encoding="utf-8")
    return root


class Processes:
    """The programs, started the way compose starts them."""

    def __init__(self, *, environment: dict[str, str]) -> None:
        self._environment = environment
        self._started: list[subprocess.Popen] = []
        self.logs: dict[str, Path] = {}
        self._log_dir = Path(tempfile.mkdtemp(prefix="moc-harness-logs-"))

    def start(self, name: str, command: list[str]) -> int:
        self.logs[name] = self._log_dir / f"{name}.log"
        handle = self.logs[name].open("w", encoding="utf-8")
        self._started.append(
            subprocess.Popen(  # noqa: S603 - fixed argv, no shell
                command, cwd=ROOT, env=self._environment, stdout=handle, stderr=handle
            )
        )
        return self._started[-1].pid

    def api(self, module: str, port: int) -> int:
        return self.start(
            module.split(":")[-1],
            [
                sys.executable, "-m", "uvicorn", "--factory", module,
                "--host", "127.0.0.1", "--port", str(port),
            ],
        )

    def worker(self, which: str) -> int:
        return self.start(f"worker-{which}", [sys.executable, "-m", "moc.workers.run", which])

    def stop(self) -> None:
        for process in self._started:
            process.terminate()
        for process in self._started:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()

    def tails(self, lines: int = 25) -> None:
        """What each program said. The point of running them separately is that
        a failure names which one."""
        for name, path in self.logs.items():
            body = path.read_text(encoding="utf-8").strip().splitlines()[-lines:]
            print(f"\n--- {name} ---")
            print("\n".join(body))


def wait_for_port(port: int, seconds: float = 30.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return True
        except OSError:
            time.sleep(0.2)
    return False


def say(step: str, detail: str = "") -> None:
    print(f"  {step:<54} {detail}", flush=True)


__all__ = [
    "ROOT",
    "Processes",
    "Vendor",
    "config_tree_pointing_at",
    "free_port",
    "say",
    "wait_for_port",
]
