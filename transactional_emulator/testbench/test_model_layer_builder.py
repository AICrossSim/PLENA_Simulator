"""Compatibility wrapper for the old sliced-layer builder unit-test filename."""

from transactional_emulator.testbench.test_sliced_layer_builder import *  # noqa: F403
from transactional_emulator.testbench.test_sliced_layer_builder import main


if __name__ == "__main__":
    raise SystemExit(main())
