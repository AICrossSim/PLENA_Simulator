#!/usr/bin/env python3

import json
import socket
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


class EmulatorClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 7878, timeout: float = 10.0):
        self.host = host
        self.port = port
        self.timeout = timeout

    def request_raw(self, request: dict[str, Any]) -> dict[str, Any]:
        payload = json.dumps(request)
        with socket.create_connection((self.host, self.port), timeout=self.timeout) as sock:
            sock_file = sock.makefile("rwb")
            sock_file.write((payload + "\n").encode("utf-8"))
            sock_file.flush()

            line = sock_file.readline()
            if not line:
                raise RuntimeError("emulator service closed the connection")

        response = json.loads(line.decode("utf-8"))
        return response

    def request(self, request: dict[str, Any]) -> Any:
        response = self.request_raw(request)
        if not response.get("ok"):
            raise EmulatorServiceError(response.get("error", "unknown emulator error"))
        return response.get("data")

    def send_command(self, cmd: str, **kwargs: Any) -> Any:
        return self.request({"cmd": cmd, **kwargs})
