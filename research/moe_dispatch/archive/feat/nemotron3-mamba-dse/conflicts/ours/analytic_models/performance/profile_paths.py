"""Portable path handling for manifests produced on profiling hosts."""

from __future__ import annotations

from pathlib import PurePosixPath


def profile_relative_path(remote_path: str) -> str:
    """Return the path below ``~/plena-profiles`` without retaining a username."""
    if not isinstance(remote_path, str):
        raise ValueError("profile artifact path must be a string")
    parts = PurePosixPath(remote_path).parts
    if len(parts) < 5 or parts[:2] != ("/", "home") or parts[3] != "plena-profiles":
        raise ValueError("profile artifact must be an absolute /home/<user>/plena-profiles path")
    if not parts[2] or any(part in {"", ".", ".."} for part in parts[4:]):
        raise ValueError("profile artifact path contains an invalid component")
    return PurePosixPath(*parts[4:]).as_posix()


__all__ = ["profile_relative_path"]
