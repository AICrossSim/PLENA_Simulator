"""Kernel compilation runner: subprocess into the compiler CLI with
manager-planned address overrides (MANAGER_DESIGN.md §2.4).

The compiler runs in .venv via `python -m tilelang_tvm_compiler compile`. The
manager pins every HBM (and FPRAM) tensor to a pre-planned address by writing
the overrides to a JSON file and passing --hbm-address-overrides /
--fpram-address-overrides. IR artifacts (isa/hlir/lowir/midir) are dumped into
managerbuild/ir/<kernel>/ (auto-refreshed each call).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Mapping, Optional

from . import env as _env
from .geometry import BehaviorSettings


@dataclass
class CompiledKernel:
    name: str
    isa_text: str
    isa_path: Path
    hlir_path: Path
    buffer_addrs: Dict[str, dict]   # name -> {address, shape, scope, dtype, [value]}
    ir_dir: Path

    def address_of(self, buf_name: str) -> int:
        return int(self.buffer_addrs[buf_name]["address"])

    def hoisted_constants(self) -> Dict[str, float]:
        """name -> value for every auto-hoisted FP constant buffer."""
        return {
            name: float(entry["value"])
            for name, entry in self.buffer_addrs.items()
            if isinstance(entry, dict) and "value" in entry
        }


def _fmt_kwargs(kwargs: Mapping[str, object]) -> str:
    return ",".join(f"{k}={v}" for k, v in kwargs.items())


def compile_kernel(
    kernel: str,
    *,
    asm_name: str,
    settings: BehaviorSettings,
    kernel_kwargs: Optional[Mapping[str, object]] = None,
    hbm_overrides: Optional[Mapping[str, int]] = None,
    fpram_overrides: Optional[Mapping[str, int]] = None,
    use_v2: bool = True,
) -> CompiledKernel:
    """Compile ``kernel`` (``module:factory`` spec) with pinned addresses.

    Geometry (mlen/blen/hlen) comes from ``settings`` (derived from the toml),
    never hardcoded. Returns a CompiledKernel with the ISA text and the parsed
    buffer address table.
    """
    ir_dir = _env.ir_dir_for(asm_name)
    if ir_dir.exists():
        shutil.rmtree(ir_dir)          # ir/ is auto-refreshed per call
    ir_dir.mkdir(parents=True, exist_ok=True)

    hlir_path = ir_dir / f"{asm_name}.hlir.txt"
    addrs_path = ir_dir / f"{asm_name}.buffer_addrs.json"
    isa_path = ir_dir / f"{asm_name}.isa"

    cmd = [
        str(_env.VENV_PYTHON), "-m", "tilelang_tvm_compiler", "compile",
        "--kernel", kernel,
        "--asm-name", asm_name,
        "--mlen", str(settings.mlen),
        "--blen", str(settings.blen),
        "--btmm-hlen", str(settings.hlen),
        "--use-v2" if use_v2 else "--no-use-v2",
        "--dump-hlir", str(hlir_path),
        "--dump-buffer-addrs", str(addrs_path),
        "--output", str(isa_path),
    ]
    if kernel_kwargs:
        cmd += ["--kernel-kwargs", _fmt_kwargs(kernel_kwargs)]

    if hbm_overrides:
        p = ir_dir / "hbm_overrides.json"
        p.write_text(json.dumps({k: int(v) for k, v in hbm_overrides.items()}, indent=2))
        cmd += ["--hbm-address-overrides", str(p)]
    if fpram_overrides:
        p = ir_dir / "fpram_overrides.json"
        p.write_text(json.dumps({k: int(v) for k, v in fpram_overrides.items()}, indent=2))
        cmd += ["--fpram-address-overrides", str(p)]

    res = subprocess.run(cmd, env=_env.compile_env(),
                         capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(
            f"compile failed for {asm_name} (rc={res.returncode}).\n"
            f"--- stderr ---\n{res.stderr}\n--- cmd ---\n{' '.join(cmd)}"
        )

    isa_text = isa_path.read_text() if isa_path.exists() else res.stdout
    buffer_addrs = json.loads(addrs_path.read_text()) if addrs_path.exists() else {}

    return CompiledKernel(
        name=asm_name,
        isa_text=isa_text,
        isa_path=isa_path,
        hlir_path=hlir_path,
        buffer_addrs=buffer_addrs,
        ir_dir=ir_dir,
    )
