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

import os
import shutil
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
    got: "np.ndarray | None" = None
    golden: "np.ndarray | None" = None

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
                 build_root: Optional[Path] = None,
                 compile_only: Optional[bool] = None):
        self.s = settings or load_behavior_settings()
        self.layout = HbmLayout(self.s, base=0)
        self.const_pool = ConstPool()
        self.build_root = Path(build_root) if build_root else _env.MANAGERBUILD
        self.hbm_dir = self.build_root / "hbm_bin"
        self.ir_dir = self.build_root / "ir"
        self.hbm_bin = self.hbm_dir / "hbm_for_behave_sim.bin"
        self.fp_sram = self.hbm_dir / "fp_sram.bin"
        # compile_only: 只编译 (compile + write bin + assemble -> .isa/.mem),
        # 跳过 emulator 运行和所有比对。便于只产 ISA / 量 cycle / 喂能耗工具,
        # 不付昂贵的 simulator 时间。默认从 env PLENA_COMPILE_ONLY 读 (1/true/yes)。
        if compile_only is None:
            compile_only = os.environ.get("PLENA_COMPILE_ONLY", "").lower() in (
                "1", "true", "yes", "on")
        self.compile_only = compile_only

    # ---- HBM placement ----
    def place(self, name, shape, role: Role, data=None) -> ManagedTensor:
        return self.layout.place(name, shape, role, data=data)

    def init_bin(self, *, weights: bool = True) -> None:
        """Zero-fill the HBM bin up to the planned upper bound, then seek-write
        the initial tensors. SCRATCH/ACTIVATION (data=None) stay zero.

        ``weights=False``: write only IO inputs, NOT weights. Weights are then
        written just-in-time per kernel (write_weights), so they can share a
        reused region and not bloat HBM. See [[feedback_weights_just_in_time]].
        """
        self.hbm_dir.mkdir(parents=True, exist_ok=True)
        total = max(self.layout.total_bytes(), 64)
        self.hbm_bin.write_bytes(b"\x00" * total)
        for t in self.layout.tensors.values():
            if t.data is None:
                continue
            if not weights and t.role is Role.WEIGHT:
                continue
            write_tensor(self.hbm_bin, t.hbm_addr, t.data, self.s)

    def write_fp_sram_for(self, ck) -> None:
        """Write fp_sram.bin for one compiled kernel, the way test_helper does:
        read each hoisted constant's COMPILER-ASSIGNED slot + value from the
        buffer-addrs dump and write the value at that slot. The compiler puts
        scratch fragments low (from FPRAM_USER_BASE) and consts above them, so
        we must NOT pre-allocate const slots ourselves (that collided with the
        scratch region). fp_sram is per-kernel (just-in-time), since each kernel
        runs as its own emulator invocation.
        """
        import numpy as np
        consts = [
            (int(e["address"]), float(e["value"]))
            for e in ck.buffer_addrs.values()
            if isinstance(e, dict) and "value" in e
        ]
        n = (max(a for a, _ in consts) + 1) if consts else 1
        arr = np.zeros(n, dtype=np.float16)
        for addr, val in consts:
            arr[addr] = np.float16(val)
        self.hbm_dir.mkdir(parents=True, exist_ok=True)
        self.fp_sram.write_bytes(arr.tobytes())

    def write_weights(self, bin_path: Path, names) -> None:
        """Seek-write the given WEIGHT tensors into ``bin_path`` (the CURRENT
        relay bin). Used just before running the kernel that consumes them, so
        weights overwrite a prior kernel's in the shared weight region without
        disturbing the activations the relay carries."""
        for name in names:
            t = self.layout.tensors[name]
            if t.data is not None:
                write_tensor(bin_path, t.hbm_addr, t.data, self.s)

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

    def _run_emulator(self, mem_path: Path, input_bin: Path,
                      out_dump: Optional[Path] = None,
                      latency_out: Optional[list] = None) -> Path:
        """Run the emulator on ``input_bin`` (the --hbm image); returns the path
        to the resulting HBM dump.

        If ``latency_out`` is a list, the hardware latency parsed from the
        emulator's ``Simulation completed. Latency ...`` stderr line (the
        modeled cycle/time cost, NOT wall-clock) is appended to it as a string.

        The emulator loads --hbm into its internal HBM, executes (writing each
        producer's output to its address), then dumps the WHOLE HBM image to
        hbm_dump.bin in its CWD. That full image is the relay medium: feeding
        kernel N's dump as kernel N+1's --hbm carries N's outputs forward with
        no asm concatenation (MANAGER_DESIGN.md §2.5). We copy the CWD dump to
        ``out_dump`` so the next run's hbm_dump.bin doesn't clobber it.

        Uses the prebuilt release binary directly (no cargo dependency).
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
            "--hbm", str(input_bin),
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
        if latency_out is not None:
            # "Simulation completed. Latency <value>" — printed unconditionally
            # in main() (not gated by --quiet). Capture the value verbatim.
            lat = "?"
            for ln in res.stderr.splitlines():
                idx = ln.find("Latency")
                if idx != -1:
                    lat = ln[idx + len("Latency"):].strip()
                    break
            latency_out.append(lat)
        dump = emu_dir / "hbm_dump.bin"
        if not dump.exists():
            raise RuntimeError(f"emulator produced no hbm_dump.bin at {dump}")
        if out_dump is not None:
            shutil.copyfile(str(dump), str(out_dump))
            return out_dump
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
        # 1. compile with hbm overrides only. NO fpram override: the compiler
        #    places FP constants itself (scratch low from FPRAM_USER_BASE,
        #    consts above) — we read those slots back and fill fp_sram.
        ck = compile_kernel(
            kernel, asm_name=asm_name, settings=self.s,
            kernel_kwargs=kernel_kwargs,
            hbm_overrides=self.layout.overrides(),
            ir_base=self.ir_dir,
        )

        # 2. write bin (io+weights) + fp_sram from the compiled const slots
        self.init_bin()
        self.write_fp_sram_for(ck)

        # 3. assemble
        mem_path = self._assemble(ck.isa_text, ck.ir_dir)

        # compile_only: 产出 ISA + .mem 即停, 不跑 emulator / 不比对。
        if self.compile_only:
            print(f"  [compile-only] {asm_name}: compiled + assembled -> "
                  f"{ck.ir_dir} (skipped emulator)")
            return {"compiled": ck, "mem_path": mem_path,
                    "hbm_dump": None, "compares": []}

        # 4. run emulator (single kernel: input bin is the manager's own bin)
        dump = self._run_emulator(mem_path, self.hbm_bin)

        # 5. read back + compare
        compares = []
        if compare:
            for name, golden in compare.items():
                t = self.layout.tensors[name]
                got = read_tensor(dump, t.hbm_addr, t.shape, self.s)
                cmp = CompareResult(
                    name=name,
                    cosine=_cosine(got, golden),
                    nrmse=_nrmse(got, golden),
                    got=np.asarray(got).reshape(-1),
                    golden=np.asarray(golden).reshape(-1),
                )
                compares.append(cmp)

        return {"compiled": ck, "mem_path": mem_path,
                "hbm_dump": dump, "compares": compares}

    # ---- graph-driven entry (MANAGER_DESIGN.md §8) ----
    def run_graph(self, graph, *, data=None, compare=None) -> Dict[str, object]:
        """Run a declarative compute graph: place tensors, topo-sort nodes,
        derive each node's tensor_map automatically, then relay-run.

        ``data``: {tensor_name -> torch.Tensor} for io/weight values.
        ``compare``: {tensor_name -> golden} attached to its producing node.
        """
        from .graph import ComputeGraph
        g = ComputeGraph.load(graph)
        steps = g.to_steps(self, data=data, compare=compare)
        return self.run_pipeline(steps)

    # ---- multi-kernel pipeline via HBM-bin relay (no asm concat) ----
    def run_pipeline(self, steps: "list[KernelStep]") -> Dict[str, object]:
        """Run a chain of kernels by relaying the HBM bin between independent
        emulator runs — NO asm concatenation.

        Each kernel is compiled + assembled + run on its own. Kernel N reads
        the HBM image kernel N-1 dumped (the full HBM is dumped every run), so a
        producer's output — written at an address the manager also pinned as the
        consumer's input — flows forward through the shared bin. The manager
        must have already ``place``d every tensor so producer.out_addr ==
        consumer.in_addr (and weights/inputs have data).

        ``steps``: ordered list of KernelStep. Returns per-step CompiledKernel +
        the final dump + all compares.
        """
        # 1. write initial bin: IO inputs only (NOT weights). Weights and
        #    fp_sram are written just-in-time per kernel below.
        #    compile_only: 不跑 emulator, 不需要 HBM 输入 bin (zero-fill 整个
        #    HBM 很慢且无用), 跳过。
        if not self.compile_only:
            self.init_bin(weights=False)

        # 2. relay: each kernel reads the previous dump, writes a new one.
        #    Each kernel runs as its own emulator invocation, so its weights
        #    (shared reused region) AND its fp_sram (compiler-assigned const
        #    slots) are written right before it runs.
        import time as _time
        results = {}
        compares = []
        input_bin = self.hbm_bin
        prev_dump = None        # the dump produced by step i-1 (relay source)
        for i, st in enumerate(steps):
            # per-step HBM overrides: map THIS kernel's buffer names to the
            # shared managed tensors' addresses (producer Y_hbm and consumer
            # X_hbm can point to the same managed tensor -> same address).
            hbm_ov = {buf: self.layout.tensors[mt].hbm_addr
                     for buf, mt in st.tensor_map.items()}
            _t0 = _time.time()
            # Deep-nesting kernels (s_concat / s_split, esp. asymmetric streams)
            # exhaust gp_only_spill's pin-reserve -> force the stable allocator
            # for just those. Everything else keeps the default (gp_only_spill).
            _alloc = "stable" if ("s_concat" in st.kernel or "s_split" in st.kernel) else None
            ck = compile_kernel(
                st.kernel, asm_name=st.asm_name, settings=self.s,
                kernel_kwargs=st.kernel_kwargs,
                hbm_overrides=hbm_ov,
                alloc_mode=_alloc,
                ir_base=self.ir_dir,
            )
            _t_compile = _time.time() - _t0
            # just-in-time weights + fp_sram: only needed to feed the emulator.
            # compile_only 跳过 (不写任何 HBM bin)。
            _t0 = _time.time()
            if not self.compile_only:
                self.write_weights(input_bin, st.weight_tensors)
                self.write_fp_sram_for(ck)
            _t_write = _time.time() - _t0

            # compile_only: 只到 .isa 文本即停。不 assemble (.isa→.mem 需要
            # .svh opcode 数字, 新加的 S_DIV_INT/S_REM_INT 还没填), 不跑 emulator,
            # 不比对, 不 relay。.isa 已由 compile 写到 ck.ir_dir/<asm>.isa。
            if self.compile_only:
                timing = {"compile": _t_compile, "write": _t_write,
                          "assemble": 0.0, "emulator": 0.0,
                          "total": _t_compile + _t_write}
                results[st.asm_name] = {"compiled": ck, "hbm_dump": None,
                                        "timing": timing, "latency": None}
                print(f"  [{i+1}/{len(steps)}] {st.asm_name}: compiled "
                      f"({timing['total']:.1f}s = compile {_t_compile:.1f} + "
                      f"write {_t_write:.1f}) [compile-only, .isa only] -> {ck.ir_dir.name}")
                continue

            _t0 = _time.time()
            mem_path = self._assemble(ck.isa_text, ck.ir_dir)
            _t_asm = _time.time() - _t0

            out_dump = ck.ir_dir / "hbm_dump.bin"
            _t0 = _time.time()
            _lat = []
            dump = self._run_emulator(mem_path, input_bin, out_dump=out_dump,
                                      latency_out=_lat)
            _t_emu = _time.time() - _t0
            timing = {"compile": _t_compile, "write": _t_write,
                     "assemble": _t_asm, "emulator": _t_emu,
                     "total": _t_compile + _t_write + _t_asm + _t_emu}
            hw_latency = _lat[0] if _lat else "?"
            results[st.asm_name] = {"compiled": ck, "hbm_dump": dump,
                                    "timing": timing, "latency": hw_latency}
            print(f"  [{i+1}/{len(steps)}] {st.asm_name}: ran ({timing['total']:.1f}s "
                  f"= compile {_t_compile:.1f} + write {_t_write:.1f} + asm {_t_asm:.1f} "
                  f"+ emu {_t_emu:.1f}) | hw latency {hw_latency} -> {dump.name}")

            # compare THIS kernel's outputs now, straight from its own dump,
            # so the dump can be freed right after (no need to keep every 1 GB
            # intermediate around until the end).
            if st.compare:
                for name, golden in st.compare.items():
                    t = self.layout.tensors[name]
                    got = read_tensor(dump, t.hbm_addr, t.shape, self.s)
                    compares.append(CompareResult(
                        name=f"{st.asm_name}:{name}",
                        cosine=_cosine(got, golden),
                        nrmse=_nrmse(got, golden),
                    ))

            # free the previous step's dump: it has now been consumed both as
            # this step's relay input AND for its own (already-done) compare.
            # Never delete self.hbm_bin (the initial input bin).
            if prev_dump is not None and prev_dump != self.hbm_bin:
                try:
                    prev_dump.unlink()
                except OSError:
                    pass
            prev_dump = dump
            input_bin = dump            # relay forward

        final_dump = input_bin

        return {"results": results, "final_dump": final_dump, "compares": compares}


@dataclass
class KernelStep:
    kernel: str                         # "module:factory" spec
    asm_name: str
    kernel_kwargs: Mapping[str, object]
    # Per-step buffer-name -> managed-tensor-name. Kernels reuse generic buffer
    # names (X_hbm/Y_hbm), so each step maps its own buffers to the shared
    # managed tensors. A producer's Y_hbm and a consumer's X_hbm map to the SAME
    # managed tensor -> same address -> relay through the bin.
    tensor_map: Mapping[str, str] = field(default_factory=dict)
    # managed-tensor-names that are WEIGHTs for this kernel — written
    # just-in-time into the relay bin right before this kernel runs.
    weight_tensors: tuple = ()
    # compare: managed-tensor-name -> golden array
    compare: Optional[Mapping[str, np.ndarray]] = None
