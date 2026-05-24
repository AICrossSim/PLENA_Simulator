"""Manager: end-to-end orchestration over the HBM bin (MANAGER_DESIGN.md §2.5).

Single-kernel flow (run_kernel):
  1. plan HBM addresses for the kernel's tensors (HbmLayout)
  2. collect + allocate the kernel's hoisted FP constants (ConstPool)
  3. seek-write input tensors into the HBM bin; write fp_sram.bin
  4. compile the kernel with hbm/fpram address overrides -> ISA
  5. assemble ISA -> machine code
  6. run the emulator (cargo) -> hbm_dump.bin
  7. read back the output tensor from hbm_dump.bin and compare to golden

All geometry/precision comes from plena_settings.toml (settings); nothing is
hardcoded. Artifacts go to managerbuild/ (hbm_bin/ persistent, ir/ refreshed).
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Mapping, Optional

import numpy as np

from . import env as _env
from .geometry import BehaviorSettings, load_behavior_settings
from .tensor import HbmLayout, ManagedTensor, Role
from .binio import write_tensor, read_tensor
from .const_pool import ConstPool
from .runner import compile_kernel, CompiledKernel


@dataclass
class CompareResult:
    name: str
    cosine: float
    nrmse: float

    def ok(self, cos_thresh: float = 0.85) -> bool:
        return self.cosine >= cos_thresh


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else float("nan")


def _nrmse(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    rng = b.max() - b.min()
    if rng == 0:
        rng = np.abs(b).max() or 1.0
    return float(np.sqrt(np.mean((a - b) ** 2)) / rng)


class Manager:
    def __init__(self, settings: Optional[BehaviorSettings] = None,
                 build_root: Optional[Path] = None):
        self.s = settings or load_behavior_settings()
        self.layout = HbmLayout(self.s, base=0)
        self.const_pool = ConstPool()
        self.build_root = Path(build_root) if build_root else _env.MANAGERBUILD
        self.hbm_dir = self.build_root / "hbm_bin"
        self.ir_dir = self.build_root / "ir"
        self.hbm_bin = self.hbm_dir / "hbm_for_behave_sim.bin"
        self.fp_sram = self.hbm_dir / "fp_sram.bin"

    # ---- HBM placement ----
    def place(self, name, shape, role: Role, data=None) -> ManagedTensor:
        return self.layout.place(name, shape, role, data=data)

    def init_bin(self) -> None:
        """Zero-fill the HBM bin up to the planned upper bound, then seek-write
        every tensor that has data. SCRATCH (data=None) stays zero."""
        self.hbm_dir.mkdir(parents=True, exist_ok=True)
        total = max(self.layout.total_bytes(), 64)
        self.hbm_bin.write_bytes(b"\x00" * total)
        for t in self.layout.tensors.values():
            if t.data is not None:
                write_tensor(self.hbm_bin, t.hbm_addr, t.data, self.s)

    # ---- assemble + run ----
    def _assemble(self, isa_text: str, ir_dir: Path) -> Path:
        """ISA text -> machine code .mem via AssemblyToBinary."""
        import sys
        sys.path.insert(0, str(_env.COMPILER_DIR))
        from compiler.assembler.assembly_to_binary import AssemblyToBinary

        asm_path = ir_dir / "generated_asm_code.asm"
        mem_path = ir_dir / "generated_machine_code.mem"
        asm_path.write_text(isa_text)
        doc = _env.COMPILER_DIR / "doc"
        asm = AssemblyToBinary(str(doc / "operation.svh"), str(doc / "configuration.svh"))
        asm.generate_binary(str(asm_path), str(mem_path))
        return mem_path

    def _run_emulator(self, mem_path: Path) -> Path:
        """Run the emulator; returns the path to its hbm_dump.bin.

        Uses the prebuilt release binary directly (no cargo dependency). The
        binary writes hbm_dump.bin / vram_dump.bin into its CWD, so we run it
        from transactional_emulator/.
        """
        emu_dir = _env.PROJECT_ROOT / "transactional_emulator"
        binary = emu_dir / "target" / "release" / "transactional_emulator"
        if not binary.exists():
            raise RuntimeError(
                f"prebuilt emulator binary not found at {binary}. "
                f"Build it once with `cargo build --release` in {emu_dir}."
            )
        cmd = [
            str(binary),
            "--opcode", str(mem_path),
            "--hbm", str(self.hbm_bin),
            "--fpsram", str(self.fp_sram),
            "--quiet",
        ]
        res = subprocess.run(cmd, cwd=str(emu_dir), env=_env.emulator_env(),
                            capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(
                f"emulator run failed (rc={res.returncode}).\n--- stderr ---\n"
                f"{res.stderr[-3000:]}"
            )
        dump = emu_dir / "hbm_dump.bin"
        if not dump.exists():
            raise RuntimeError(f"emulator produced no hbm_dump.bin at {dump}")
        return dump

    # ---- end-to-end single kernel ----
    def run_kernel(
        self,
        kernel: str,
        *,
        asm_name: str,
        kernel_kwargs: Mapping[str, object],
        compare: Optional[Mapping[str, np.ndarray]] = None,
    ) -> Dict[str, object]:
        """Compile + run one kernel over the current HBM layout.

        ``compare``: {tensor_name -> golden_fp32_array}. Each is read back from
        the emulator's hbm_dump at the tensor's planned address and compared.
        Returns {"compiled": CompiledKernel, "compares": [CompareResult], ...}.
        """
        # 1+2. collect/allocate this kernel's constants (compile once to discover)
        probe = compile_kernel(kernel, asm_name=asm_name, settings=self.s,
                              kernel_kwargs=kernel_kwargs)
        self.const_pool.collect(asm_name, probe.hoisted_constants())

        # 3. write bin + fp_sram
        self.init_bin()
        self.const_pool.write_fp_sram(self.fp_sram)

        # 4. compile with hbm + fpram overrides
        ck = compile_kernel(
            kernel, asm_name=asm_name, settings=self.s,
            kernel_kwargs=kernel_kwargs,
            hbm_overrides=self.layout.overrides(),
            fpram_overrides=self.const_pool.overrides_for(asm_name),
        )

        # 5. assemble
        mem_path = self._assemble(ck.isa_text, ck.ir_dir)
        # 6. run emulator
        dump = self._run_emulator(mem_path)

        # 7. read back + compare
        compares = []
        if compare:
            for name, golden in compare.items():
                t = self.layout.tensors[name]
                got = read_tensor(dump, t.hbm_addr, t.shape, self.s)
                cmp = CompareResult(
                    name=name,
                    cosine=_cosine(got, golden),
                    nrmse=_nrmse(got, golden),
                )
                compares.append(cmp)

        return {"compiled": ck, "mem_path": mem_path,
                "hbm_dump": dump, "compares": compares}
