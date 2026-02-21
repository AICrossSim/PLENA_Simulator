"""
PLENA ATen-style operator registration package.

Usage:
    from plena.ops.registry import OpRegistry, Backend
    import plena.ops as ops

    OpRegistry.load()  # loads native_ops.yaml from this package
    prog = PLENAProgram(mlen=64, blen=4)
    result = ops.softmax(prog, X_batch, scale=1.0)
"""
from pathlib import Path

PLENA_PKG_DIR = Path(__file__).parent
NATIVE_OPS_YAML = PLENA_PKG_DIR / "native_ops.yaml"
