"""Compatibility wrapper for the old model-layer testbench module name.

Use ``transactional_emulator.testbench.sliced_layer_test_builder`` for new code.
"""

from transactional_emulator.testbench import sliced_layer_test_builder as _impl

for _name, _value in vars(_impl).items():
    if _name not in {
        "__builtins__",
        "__cached__",
        "__doc__",
        "__file__",
        "__loader__",
        "__name__",
        "__package__",
        "__spec__",
    }:
        globals()[_name] = _value

for _compat_local in ("_impl", "_name", "_value"):
    globals().pop(_compat_local, None)

__all__ = [name for name in globals() if not name.startswith("__")]
