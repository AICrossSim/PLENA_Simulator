from pathlib import Path

# Canonical build directory for the `just build-emulator` flow. A single shared
# location (transactional_emulator/build) so the justfile recipe, view_mem, and
# every testbench agree on where generated artifacts live. This lives on the
# transactional_emulator side (not in the generic PLENA_Tools package) because
# the build location is a property of this consumer, not of the shared tools.
# Returned as a Path so callers can use the `/` operator.
BUILD_DIR = Path(__file__).resolve().parent.parent / "build"
