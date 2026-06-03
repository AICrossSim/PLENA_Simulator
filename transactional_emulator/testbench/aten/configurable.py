"""Reusable ATen testbench configuration.

The ATen template tests need the compiler and Rust emulator to agree on the
same physical hardware dimensions.  This base class owns that setup and writes
a per-build TOML override instead of mutating the repository-wide
``plena_settings.toml``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomlkit
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
COMPILER_ROOT = REPO_ROOT / "PLENA_Compiler"
for path in (REPO_ROOT, COMPILER_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from compiler.aten.plena import PlenaCompiler  # noqa: E402
from transactional_emulator.testbench.emulator_runner import run_and_assert  # noqa: E402
from transactional_emulator.testbench.sim_env_utils import create_mem_for_sim  # noqa: E402
from transactional_emulator.tools.create_sim_env import create_sim_env  # noqa: E402


DEFAULT_REAL_DATA_RATIO = (8 * 8 + 8) / (8 * 8)

LATENCY_PROFILE_PRESETS: dict[str, dict[str, int]] = {
    # Measured on current main + khl/addr-pipeline-fix RTL, behavioral mode.
    # See ~/docs/plena_latency_audit_20260520.md.
    "nexys_video_150mhz": {
        "SCALAR_FP_EXP_CYCLES": 18,
        "SCALAR_FP_RECI_CYCLES": 6,
        "VECTOR_ADD_CYCLES": 11,
        "VECTOR_EXP_CYCLES": 18,
    },
}


def apply_latency_profile_config(
    config: dict[str, Any],
    *,
    dc_en: int | None,
    latency_profile: str | None,
) -> None:
    """Apply latency-profile CLI options to a per-build TOML dict.

    Unknown profiles are still allowed; the Rust emulator will fall back to
    `dc_lib_dis` for entries without a matching profile key.
    """

    for mode in ("TRANSACTIONAL", "ANALYTIC"):
        if mode not in config:
            continue
        mode_config = config[mode].setdefault("CONFIG", {})
        if dc_en is not None:
            mode_config.setdefault("DC_EN", {})["value"] = int(dc_en)
        if latency_profile:
            mode_config["LATENCY_PROFILE"] = {"value": latency_profile}
            latency = config[mode].setdefault("LATENCY", {})
            for name, value in LATENCY_PROFILE_PRESETS.get(latency_profile, {}).items():
                if name in latency:
                    latency[name][latency_profile] = int(value)


@dataclass(frozen=True)
class HardwareConfig:
    mlen: int
    vlen: int
    blen: int
    hlen: int
    broadcast_amount: int
    dc_en: int | None
    latency_profile: str | None
    hbm_m_prefetch_amount: int | None
    hbm_v_prefetch_amount: int | None
    hbm_v_writeback_amount: int | None
    real_data_ratio: float = DEFAULT_REAL_DATA_RATIO

    @classmethod
    def from_args(
        cls,
        args: argparse.Namespace,
        *,
        default_hlen: int | None = None,
        default_broadcast_amount: int | None = None,
    ) -> HardwareConfig:
        base = read_behavior_config()
        mlen = int(args.mlen)
        vlen = int(args.vlen if args.vlen is not None else mlen)
        blen = int(args.blen)
        # Sub-64 MLEN is supported: the emulator packs HBM rows tightly and reads
        # aligned 64-byte words with byte-granular extraction (transfer_mx_from_hbm).
        # The only HBM requirement is a byte-aligned element row, which 8-bit MXFP
        # always satisfies. (mlen % blen alignment is checked in setup_hw.)
        hlen = int(args.hlen if args.hlen is not None else default_hlen or base["HLEN"])
        broadcast_amount = int(
            args.broadcast_amount
            if args.broadcast_amount is not None
            else default_broadcast_amount or base["BROADCAST_AMOUNT"]
        )
        return cls(
            mlen=mlen,
            vlen=vlen,
            blen=blen,
            hlen=hlen,
            broadcast_amount=broadcast_amount,
            dc_en=args.dc_en if args.dc_en is not None else (0 if args.latency_profile else None),
            latency_profile=args.latency_profile,
            hbm_m_prefetch_amount=args.hbm_m_prefetch,
            hbm_v_prefetch_amount=args.hbm_v_prefetch,
            hbm_v_writeback_amount=args.hbm_v_writeback,
            real_data_ratio=float(args.real_data_ratio),
        )

    def compiler_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "mlen": self.mlen,
            "blen": self.blen,
            "real_data_ratio": self.real_data_ratio,
        }
        if self.hbm_v_prefetch_amount is not None:
            kwargs["hbm_v_prefetch_amount"] = self.hbm_v_prefetch_amount
        if self.hbm_v_writeback_amount is not None:
            kwargs["hbm_v_writeback_amount"] = self.hbm_v_writeback_amount
        return kwargs

    def write_toml(self, build_dir: Path) -> Path:
        source_path = Path(os.environ.get("PLENA_BASE_SETTINGS_TOML", REPO_ROOT / "plena_settings.toml"))
        with source_path.open() as f:
            config = tomlkit.load(f)

        txn = config["TRANSACTIONAL"]["CONFIG"]
        txn["MLEN"]["value"] = self.mlen
        txn["VLEN"]["value"] = self.vlen
        txn["BLEN"]["value"] = self.blen
        txn["HLEN"]["value"] = self.hlen
        txn["BROADCAST_AMOUNT"]["value"] = self.broadcast_amount
        txn["HBM_M_Prefetch_Amount"]["value"] = self.hbm_m_prefetch_amount or self.mlen
        txn["HBM_V_Prefetch_Amount"]["value"] = self.hbm_v_prefetch_amount or self.blen
        txn["HBM_V_Writeback_Amount"]["value"] = self.hbm_v_writeback_amount or self.blen
        txn["HBM_WIDTH"]["value"] = max(self.mlen * 8, txn["HBM_WIDTH"]["value"])
        apply_latency_profile_config(
            config,
            dc_en=self.dc_en,
            latency_profile=self.latency_profile,
        )

        # PlenaCompiler reads from BEHAVIOR.CONFIG — mirror tile dimensions AND
        # the HBM prefetch/writeback amounts there. The compiler's preload
        # codegen (load_batch) advances the destination by
        # BEHAVIOR.CONFIG.HBM_V_Prefetch_Amount rows per H_PREFETCH_V, while the
        # emulator's H_PREFETCH_V writes TRANSACTIONAL.CONFIG.HBM_V_Prefetch_Amount
        # rows. If these disagree, the preload writes overlapping windows that
        # spill past the tensor into adjacent VRAM (e.g. a freshly-alloc'd output
        # region) with NaN over-read data — silently corrupting ops that read
        # their output region before writing (e.g. softmax's S += X).
        if "BEHAVIOR" not in config:
            config["BEHAVIOR"] = {}
        if "CONFIG" not in config["BEHAVIOR"]:
            config["BEHAVIOR"]["CONFIG"] = {}
        beh = config["BEHAVIOR"]["CONFIG"]
        for key in (
            "MLEN",
            "VLEN",
            "BLEN",
            "HLEN",
            "BROADCAST_AMOUNT",
            "HBM_M_Prefetch_Amount",
            "HBM_V_Prefetch_Amount",
            "HBM_V_Writeback_Amount",
        ):
            beh[key] = {"value": txn[key]["value"]}

        build_dir.mkdir(parents=True, exist_ok=True)
        out_path = build_dir / "plena_settings.toml"
        with out_path.open("w") as f:
            tomlkit.dump(config, f)
        return out_path


def add_hw_args(parser: argparse.ArgumentParser) -> None:
    """Add standard hardware tile-size arguments to an argparse parser."""
    parser.add_argument("--mlen", type=int, default=64)
    parser.add_argument("--vlen", type=int, default=None)
    parser.add_argument("--blen", type=int, default=4)
    parser.add_argument("--hlen", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)


def resolve_rows(args, default_seq):
    """Resolve the total token-row count for the unified [batch_size, seq_len, hidden]
    interface: rows = batch_size * seq_len.

    `batch_size` defaults to 1 and `seq_len` defaults to `default_seq` (each test's
    historical row count), so a no-arg invocation reproduces the previous behavior.
    The non-attention ops are per-row independent, so a [batch, seq, hidden] input is
    numerically identical to a flat [rows, hidden] one.

    Returns (rows, batch_size, seq_len); raises if rows is not a positive multiple of blen.
    """
    batch_size = args.batch_size if args.batch_size is not None else 1
    seq_len = args.seq_len if args.seq_len is not None else default_seq
    rows = batch_size * seq_len
    if rows <= 0 or rows % args.blen != 0:
        raise ValueError(
            f"rows = batch_size*seq_len = {batch_size}*{seq_len} = {rows} "
            f"must be a positive multiple of BLEN ({args.blen})"
        )
    return rows, batch_size, seq_len


def setup_hw(args: argparse.Namespace, build_dir: Path) -> HardwareConfig:
    """Create a HardwareConfig from parsed args, write per-build TOML, set env var.

    Returns the HardwareConfig for use in test setup.
    """
    mlen = args.mlen
    vlen = args.vlen if args.vlen is not None else mlen
    blen = args.blen

    if mlen % blen != 0:
        raise ValueError(f"MLEN ({mlen}) must be divisible by BLEN ({blen})")
    if vlen != mlen:
        raise ValueError(f"VLEN ({vlen}) must equal MLEN ({mlen}) for ATen tests")
    # Sub-64 MLEN is supported: the Python HBM writer packs rows tightly and the
    # emulator reads aligned 64-byte words with byte-granular extraction
    # (transfer_mx_from_hbm / transfer_mx_to_hbm). 8-bit MXFP rows are always
    # byte-aligned, so no MLEN%64 constraint is needed.

    base = read_behavior_config()
    hlen = args.hlen if args.hlen is not None else base["HLEN"]
    broadcast_amount = mlen // hlen

    hw = HardwareConfig(
        mlen=mlen,
        vlen=vlen,
        blen=blen,
        hlen=hlen,
        broadcast_amount=broadcast_amount,
        dc_en=None,
        latency_profile=None,
        hbm_m_prefetch_amount=None,
        hbm_v_prefetch_amount=None,
        hbm_v_writeback_amount=None,
    )
    toml_path = hw.write_toml(build_dir)
    os.environ["PLENA_SETTINGS_TOML"] = str(toml_path)
    os.environ.setdefault("PLENA_TEXT_DUMP_MAX_VALUES", "10000000")
    return hw


class AtenTemplateTestbench:
    """Base class for configurable ATen template tests."""

    def __init__(
        self,
        args: argparse.Namespace,
        *,
        name: str,
        default_build_dir: Path,
        default_hlen: int | None = None,
        default_broadcast_amount: int | None = None,
    ) -> None:
        self.args = args
        self.name = name
        self.build_dir = (args.build_dir or default_build_dir).resolve()
        self.hw = HardwareConfig.from_args(
            args,
            default_hlen=default_hlen,
            default_broadcast_amount=default_broadcast_amount,
        )
        settings_path = self.hw.write_toml(self.build_dir)
        os.environ["PLENA_SETTINGS_TOML"] = str(settings_path)
        torch.manual_seed(args.seed)

    @staticmethod
    def add_common_args(parser: argparse.ArgumentParser, *, default_build_dir: Path) -> None:
        parser.add_argument("--mlen", type=int, default=64)
        parser.add_argument("--vlen", type=int, default=None)
        parser.add_argument("--blen", type=int, default=4)
        parser.add_argument("--hlen", type=int, default=None)
        parser.add_argument("--broadcast-amount", type=int, default=None)
        parser.add_argument("--hbm-m-prefetch", type=int, default=None)
        parser.add_argument("--hbm-v-prefetch", type=int, default=None)
        parser.add_argument("--hbm-v-writeback", type=int, default=None)
        parser.add_argument("--real-data-ratio", type=float, default=DEFAULT_REAL_DATA_RATIO)
        parser.add_argument(
            "--dc-en",
            type=int,
            choices=(0, 1),
            default=None,
            help="Override BEHAVIOR/ANALYTIC CONFIG.DC_EN in the per-build TOML.",
        )
        parser.add_argument(
            "--latency-profile",
            default=None,
            help=("Optional per-FPGA latency profile for DC_EN=0. Known preset: nexys_video_150mhz."),
        )
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--no-run", action="store_true")
        parser.add_argument("--build-dir", type=Path, default=default_build_dir)
        parser.add_argument(
            "--atol",
            type=float,
            default=0.2,
            help="Absolute tolerance for emulator/golden allclose checks.",
        )
        parser.add_argument(
            "--rtol",
            type=float,
            default=0.2,
            help="Relative tolerance for emulator/golden allclose checks.",
        )
        parser.add_argument(
            "--min-allclose-rate",
            type=float,
            default=90.0,
            help="Required percentage of values satisfying the allclose tolerance.",
        )
        parser.add_argument(
            "--strict-allclose",
            action="store_true",
            help="Require 100% of compared values to satisfy allclose tolerance.",
        )

    def compiler(self, **kwargs: Any) -> PlenaCompiler:
        compiler_kwargs = self.hw.compiler_kwargs()
        compiler_kwargs.update(kwargs)
        return PlenaCompiler(**compiler_kwargs)

    def comparison_params(
        self,
        *,
        vram_addr: int,
        rows: int,
        logical_cols: int,
        physical_cols: int | None = None,
        physical_rows: int | None = None,
        use_stride_mode: bool | None = None,
        use_slice_mode: bool = False,
        slice_per_row: int | None = None,
    ) -> dict[str, Any]:
        row_dim = self.hw.mlen
        storage_cols = physical_cols or logical_cols
        storage_rows = physical_rows or rows
        num_rows = (storage_rows * storage_cols + row_dim - 1) // row_dim
        if use_stride_mode is None:
            use_stride_mode = logical_cols > row_dim
        min_allclose_rate = 100.0 if self.args.strict_allclose else float(self.args.min_allclose_rate)
        params = {
            "start_row_idx": vram_addr // row_dim,
            "num_rows": num_rows,
            "num_batches": rows,
            "elements_per_batch": logical_cols,
            "row_dim": row_dim,
            "atol": float(self.args.atol),
            "rtol": float(self.args.rtol),
            "min_allclose_match_rate": min_allclose_rate,
            "use_stride_mode": use_stride_mode,
            "use_slice_mode": use_slice_mode,
            "slice_per_row": slice_per_row,
        }
        if physical_rows is not None:
            params["physical_rows"] = storage_rows
        return params

    def prepare_and_run(
        self,
        *,
        asm_name: str,
        gen_code: str,
        input_tensor: dict[str, torch.Tensor],
        golden_result: dict[str, Any],
        fp_preload: list[float] | torch.Tensor | None,
        specified_data_order: list[str],
        comparison_params: dict[str, Any],
        vram_preload: torch.Tensor | None = None,
        tensor_layouts: dict[str, Any] | None = None,
    ) -> None:
        create_sim_env(
            input_tensor,
            gen_code,
            golden_result,
            fp_preload,
            build_dir=str(self.build_dir),
            vram_preload=vram_preload,
            tensor_layouts=tensor_layouts,
        )
        create_mem_for_sim(
            data_size=256,
            mode="behave_sim",
            asm=asm_name,
            data=None,
            specified_data_order=specified_data_order,
            build_path=self.build_dir,
        )
        with (self.build_dir / "comparison_params.json").open("w") as f:
            json.dump(comparison_params, f, indent=2)
        with (self.build_dir / "generated_asm_code.asm").open("w") as f:
            f.write(gen_code)

        print(f"Generated {len(gen_code.splitlines())} lines of ISA")
        print(f"Hardware config: {self.hw}")
        print(f"Simulation environment created: {self.build_dir}")
        if not self.args.no_run:
            run_and_assert(self.build_dir, self.name, mlen=self.hw.mlen, blen=self.hw.blen)


def read_behavior_config() -> dict[str, int]:
    source_path = Path(os.environ.get("PLENA_BASE_SETTINGS_TOML", REPO_ROOT / "plena_settings.toml"))
    with source_path.open() as f:
        config = tomlkit.load(f)
    behavior = config["TRANSACTIONAL"]["CONFIG"]
    out = {}
    for key in (
        "MLEN",
        "VLEN",
        "BLEN",
        "HLEN",
        "BROADCAST_AMOUNT",
        "HBM_M_Prefetch_Amount",
        "HBM_V_Prefetch_Amount",
        "HBM_V_Writeback_Amount",
    ):
        out[key] = int(behavior[key]["value"])
    return out
