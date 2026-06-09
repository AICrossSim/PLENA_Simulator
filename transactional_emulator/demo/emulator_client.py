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
      identifiable in the monitor GUI.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        # 7979 is the ooo_arch worktree's gateway default — see
        # ../start_online_sim.sh for the layout. The sibling
        # yw/online_emulator worktree uses 7878. Override via the
        # `port=` kwarg when targeting a non-default gateway.
        port: int = 7979,
        timeout: float = 10.0,
        *,
        label: str | None = None,
        auto_label: bool = False,
        config_toml: str | None = None,
        config_name: str | None = None,
    ):
        """
        Parameters
        ----------
        config_toml
            Full TOML text for this session's hardware config. When set,
            the persistent ``connect()`` flow sends a ``set_session_config``
            handshake before any hardware-touching command. The gateway
            writes the TOML to a tempfile and exposes it as
            ``PLENA_CONFIG=<tempfile>`` to the spawned backend, so each
            session can run on its own hardware footprint without
            restarting the gateway. Must be supplied *with* ``label`` (or
            ``auto_label=True``) since per-session config only makes sense
            in persistent mode.
        config_name
            Optional short tag for the WebGUI (e.g. ``"config_2"``). Just a
            display label — the actual config comes from ``config_toml``.
        """
        self.host = host
        self.port = port
        self.timeout = timeout
        self._label = label
        self._auto_label = auto_label or label is not None
        self._config_toml = config_toml
        self._config_name = config_name
        self._config_handshake_done = False
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
        """Open a persistent connection. Idempotent.

        Handshake order (only the steps that apply are sent):
          1. ``set_label`` — if a label was given (or ``auto_label=True``).
          2. ``set_session_config`` — if ``config_toml`` was given. This
             MUST happen before any hardware-touching command, since the
             gateway is in lazy-spawn mode and we lose the chance to
             inject ``PLENA_CONFIG`` into the backend once it's running.
        """
        with self._lock:
            if self._sock is not None:
                return self
            self._sock, self._rw = self._open_socket()
            if self._auto_label:
                label = self._label or _default_label()
                self._exchange(self._rw, {"cmd": "set_label", "label": label})
            if self._config_toml is not None and not self._config_handshake_done:
                payload = {
                    "cmd": "set_session_config",
                    "config_toml": self._config_toml,
                }
                if self._config_name:
                    payload["config_name"] = self._config_name
                resp = self._exchange(self._rw, payload)
                if not resp.get("ok", False):
                    raise RuntimeError(
                        f"set_session_config rejected by gateway: {resp.get('error', resp)}"
                    )
                self._config_handshake_done = True
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
