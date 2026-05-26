#!/usr/bin/env python3

import json
import os
import socket
import sys
import threading
from typing import Any


class EmulatorServiceError(RuntimeError):
    """Raised when the online emulator returns an error response."""


def parse_opcode_batch(text: str) -> list[str]:
    tokens = []
    for chunk in text.replace(",", " ").split():
        token = chunk.strip()
        if token:
            tokens.append(token)
    return tokens


def _default_label() -> str:
    script = os.path.basename(sys.argv[0]) if sys.argv and sys.argv[0] else "python"
    return f"{script}[{os.getpid()}]"


class EmulatorClient:
    """TCP client for the PLENA online emulator.

    Two connection modes:

    - Ephemeral (default): every `send_command` opens a fresh TCP connection.
      Backwards-compatible, but each call shows up as a separate session in
      the server's session registry.

    - Persistent: call `connect()` (or use as a context manager) to keep a
      single socket open for the lifetime of the client. The server will see
      one stable session, and the client may auto-`set_label` itself so it is
      identifiable in the GUI's session list.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7878,
        timeout: float = 10.0,
        *,
        label: str | None = None,
        auto_label: bool = False,
    ):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._label = label
        self._auto_label = auto_label or label is not None
        self._sock: socket.socket | None = None
        self._rw = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ low-level

    def _open_socket(self) -> tuple[socket.socket, Any]:
        sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        return sock, sock.makefile("rwb")

    def _exchange(self, rw, request: dict[str, Any]) -> dict[str, Any]:
        payload = json.dumps(request)
        rw.write((payload + "\n").encode("utf-8"))
        rw.flush()
        line = rw.readline()
        if not line:
            raise RuntimeError("emulator service closed the connection")
        return json.loads(line.decode("utf-8"))

    # ------------------------------------------------------------------ persistent mode

    def connect(self) -> "EmulatorClient":
        """Open a persistent connection. Idempotent."""
        with self._lock:
            if self._sock is not None:
                return self
            self._sock, self._rw = self._open_socket()
            if self._auto_label:
                label = self._label or _default_label()
                self._exchange(self._rw, {"cmd": "set_label", "label": label})
        return self

    def close(self) -> None:
        with self._lock:
            if self._rw is not None:
                try:
                    self._rw.close()
                except Exception:
                    pass
                self._rw = None
            if self._sock is not None:
                try:
                    self._sock.close()
                except Exception:
                    pass
                self._sock = None

    def __enter__(self) -> "EmulatorClient":
        return self.connect()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ------------------------------------------------------------------ public API

    def request_raw(self, request: dict[str, Any]) -> dict[str, Any]:
        # Persistent connection path: reuse the open socket under a lock.
        if self._sock is not None:
            with self._lock:
                if self._sock is None:
                    return self._ephemeral_request(request)
                try:
                    return self._exchange(self._rw, request)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    # Drop the dead socket and fall back to ephemeral mode for this call.
                    try:
                        self._sock.close()
                    except Exception:
                        pass
                    self._sock = None
                    self._rw = None
            return self._ephemeral_request(request)

        return self._ephemeral_request(request)

    def _ephemeral_request(self, request: dict[str, Any]) -> dict[str, Any]:
        sock, rw = self._open_socket()
        try:
            return self._exchange(rw, request)
        finally:
            try:
                rw.close()
            finally:
                sock.close()

    def request(self, request: dict[str, Any]) -> Any:
        response = self.request_raw(request)
        if not response.get("ok"):
            raise EmulatorServiceError(response.get("error", "unknown emulator error"))
        return response.get("data")

    def send_command(self, cmd: str, **kwargs: Any) -> Any:
        return self.request({"cmd": cmd, **kwargs})
