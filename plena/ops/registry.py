"""
OpRegistry — PLENA ATen-style operator dispatch registry.

Loads operator declarations from native_ops.yaml and routes calls
to the correct backend implementation (CPU or PLENA).
"""

from __future__ import annotations

import importlib
import yaml
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


class Backend(Enum):
    """Available dispatch backends."""
    CPU = "cpu"
    PLENA = "plena"


class MemoryPattern(Enum):
    BATCH_ONLY = "batch_only"
    SUB_MATRIX_COL = "sub_matrix_col"
    SUB_MATRIX_ROW = "sub_matrix_row"
    FLASH_ATTN = "flash_attn"


class TileLoopMode(Enum):
    AUTO = "auto"
    MANUAL = "manual"
    NONE = "none"


@dataclass
class PlenaBackendInfo:
    """PLENA-specific metadata for an operator."""
    asm_template: Optional[str]
    memory_pattern: MemoryPattern
    tile_loops: TileLoopMode = TileLoopMode.NONE
    uses_fpram: bool = False
    uses_mram: bool = False


@dataclass
class OpSchema:
    """Parsed operator declaration from native_ops.yaml."""
    name: str
    func_signature: str
    category: str          # "primitive" or "composite"
    in_place: bool
    dispatch: Dict[str, str]   # backend_name -> "module.path.function"
    plena_backend: PlenaBackendInfo
    doc: str
    _resolved: Dict[str, Callable] = field(default_factory=dict, repr=False)

    def resolve(self, backend: str) -> Callable:
        """Lazily import and cache the backend implementation."""
        if backend not in self._resolved:
            path = self.dispatch[backend]
            module_path, func_name = path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            self._resolved[backend] = getattr(module, func_name)
        return self._resolved[backend]


class OpRegistry:
    """
    Central PLENA operator registry.

    Usage:
        registry = OpRegistry.load()          # load default native_ops.yaml
        registry.set_backend(Backend.PLENA)   # or Backend.CPU
        result = registry.dispatch("softmax", prog, X_batch, scale=1.0)

        # Or via plena.ops:
        import plena.ops as ops
        ops.softmax(prog, X_batch, scale=1.0)
    """

    _instance: Optional["OpRegistry"] = None

    def __init__(self):
        self._ops: Dict[str, OpSchema] = {}
        self._backend: Backend = Backend.PLENA

    @classmethod
    def load(cls, yaml_path: Optional[str] = None) -> "OpRegistry":
        """
        Load registry from YAML.

        Args:
            yaml_path: Path to native_ops.yaml. Defaults to the file
                       bundled with this package.
        """
        if yaml_path is None:
            # Default: same directory as this package
            yaml_path = Path(__file__).parent.parent / "native_ops.yaml"
        else:
            yaml_path = Path(yaml_path)

        registry = cls()
        with open(yaml_path, "r") as f:
            entries = yaml.safe_load(f)

        for entry in entries:
            schema = cls._parse(entry)
            registry._ops[schema.name] = schema

        cls._instance = registry
        return registry

    @classmethod
    def get(cls) -> "OpRegistry":
        """Return the singleton registry, loading defaults if needed."""
        if cls._instance is None:
            cls.load()
        return cls._instance

    @staticmethod
    def _parse(entry: dict) -> OpSchema:
        func_str = entry["func"]
        name = func_str.split("(")[0].strip()
        pb = entry.get("plena_backend", {})
        return OpSchema(
            name=name,
            func_signature=func_str,
            category=entry.get("category", "primitive"),
            in_place=entry.get("in_place", False),
            dispatch=entry.get("dispatch", {}),
            plena_backend=PlenaBackendInfo(
                asm_template=pb.get("asm_template"),
                memory_pattern=MemoryPattern(pb.get("memory_pattern", "batch_only")),
                tile_loops=TileLoopMode(pb.get("tile_loops", "none")),
                uses_fpram=pb.get("uses_fpram", False),
                uses_mram=pb.get("uses_mram", False),
            ),
            doc=entry.get("doc", ""),
        )

    def set_backend(self, backend: Backend) -> None:
        """Set the active backend for dispatch calls."""
        self._backend = backend

    def get_backend(self) -> Backend:
        """Return the currently active backend."""
        return self._backend

    def get_op(self, name: str) -> OpSchema:
        if name not in self._ops:
            available = list(self._ops.keys())
            raise KeyError(
                f"Operator '{name}' not in registry. Available: {available}"
            )
        return self._ops[name]

    def list_ops(self, category: Optional[str] = None) -> List[str]:
        if category is None:
            return list(self._ops.keys())
        return [n for n, s in self._ops.items() if s.category == category]

    def dispatch(
        self,
        op_name: str,
        *args,
        backend: Optional[Backend] = None,
        **kwargs,
    ) -> Any:
        """
        Dispatch an operator call to the active (or specified) backend.

        Args:
            op_name: Registered operator name (e.g. "softmax")
            *args:   Positional args forwarded to the implementation.
                     For PLENA backend, first arg MUST be PLENAProgram.
                     For CPU backend, args are torch.Tensor objects.
            backend: Override active backend for this single call.
            **kwargs: Additional keyword arguments forwarded to impl.

        Returns:
            CPU backend:   torch.Tensor (golden reference value)
            PLENA backend: TensorVar proxy (ISA has been generated in prog)
        """
        schema = self.get_op(op_name)
        target = (backend or self._backend).value
        impl = schema.resolve(target)
        return impl(*args, **kwargs)
