from __future__ import annotations

import pytest

from .profile_paths import profile_relative_path


def test_profile_path_drops_the_profiling_username() -> None:
    assert profile_relative_path(
        "/home/another-user/plena-profiles/formal-runs/campaign/manifest.json"
    ) == "formal-runs/campaign/manifest.json"


@pytest.mark.parametrize(
    "path",
    (
        "relative/plena-profiles/file.json",
        "/tmp/plena-profiles/file.json",
        "/home/user/results/file.json",
        "/home/user/plena-profiles/../private/file.json",
    ),
)
def test_profile_path_rejects_locations_outside_the_campaign_root(path: str) -> None:
    with pytest.raises(ValueError, match="profile artifact"):
        profile_relative_path(path)
