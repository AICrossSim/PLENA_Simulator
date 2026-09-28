"""Load the dispatch planner from the pinned Compiler submodule.

PLENA_DISPATCH_COMPILER may point at its research/moe_dispatch directory during
paired-repository development. Every experiment hashes the selected source.
"""
import importlib.util
import os
from pathlib import Path


HERE = Path(__file__).resolve().parent
COMPILER_DIR = Path(os.environ.get(
    "PLENA_DISPATCH_COMPILER",
    str(HERE.parents[1] / "PLENA_Compiler/research/moe_dispatch"),
)).resolve()
COMPILER_PATH = COMPILER_DIR / "compiler.py"
if not COMPILER_PATH.is_file():
    raise ImportError(
        "Dispatch Compiler missing. Run git submodule update --init PLENA_Compiler "
        "or set PLENA_DISPATCH_COMPILER to its research/moe_dispatch directory."
    )
spec = importlib.util.spec_from_file_location("plena_dispatch_compiler", COMPILER_PATH)
compiler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compiler)


def default_binary() -> Path:
    """Use Cargo's configured target directory or this standalone crate's target."""
    override = os.environ.get("PLENA_DISPATCH_TEST_BINARY")
    if override:
        return Path(override).resolve()
    target = Path(os.environ.get("CARGO_TARGET_DIR", str(HERE / "rust/target")))
    return target.resolve() / "release/moe-dispatch-analytical-v1"
