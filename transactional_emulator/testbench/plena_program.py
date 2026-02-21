"""PLENAProgram high-level compiler interface."""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable, Union
from functools import wraps

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from developer_compiler import DeveloperCompiler
# ISA generation is fully delegated to DeveloperCompiler; this file does not directly generate ISA


# ============================================================================
# TensorVar Proxy Object Hierarchy
# ============================================================================

class TensorVar:
    """
    Tensor proxy object base class

    All tensor variables inherit from this class.
    Supports __matmul__ (`@`) operator, which automatically dispatches to appropriate PLENAProgram methods.

    Dual naming:
    - display_name: User-visible name (e.g., "temp", "Q", "S")
    - internal_name: System internal name (e.g., "my_func_0/temp"), used for symbol table and ISA generation
    """

    def __init__(self, program: "PLENAProgram", internal_name: str, kind: str,
                 shape: Tuple[int, int], display_name: Optional[str] = None):
        self._program = program
        self.internal_name = internal_name   # System internal name (with scope prefix), used by symbol table
        self.display_name = display_name if display_name is not None else internal_name  # User-visible name
        self.kind = kind                     # "input", "batch", "matrix", "vram_matrix"
        self.shape = shape

    @property
    def name(self) -> str:
        """Compatibility property: returns internal_name for internal system use"""
        return self.internal_name

    def __matmul__(self, other):
        """A @ B: Dispatch to appropriate computation based on operand types"""
        return self._program._dispatch_matmul(self, other)

    def __repr__(self):
        if self.display_name != self.internal_name:
            return (f"{self.__class__.__name__}(display={self.display_name!r}, "
                    f"internal={self.internal_name!r}, shape={self.shape})")
        return f"{self.__class__.__name__}({self.display_name!r}, shape={self.shape})"


class InputVar(TensorVar):
    """
    Input variable: tensor declared in HBM

    Not yet loaded to VRAM; needs to be loaded via load_batch / load_matrix.
    """

    def __init__(self, program: "PLENAProgram", name: str, shape: Tuple[int, int],
                 hbm_addr: int, hbm_size: int, display_name: Optional[str] = None):
        super().__init__(program, name, "input", shape, display_name=display_name)
        self.hbm_addr = hbm_addr
        self.hbm_size = hbm_size


class FPVar:
    """
    FP variable: maps to a contiguous region in FPRAM

    Declared via prog.fp_var("scale", size=1), automatically allocates FPRAM space.
    Provides .address for ISA generation (S_LD_FP / S_ST_FP).

    Usage:
        scale = prog.fp_var("scale", size=1)
        m_old = prog.fp_var("m_old", size=64)

        scale.address   # -> FPRAM address (int)
        scale.size      # -> number of elements
        scale[3]        # -> address + 3 (element offset)
    """

    def __init__(self, program: "PLENAProgram", internal_name: str, address: int,
                 size: int, display_name: Optional[str] = None):
        self._program = program
        self.internal_name = internal_name
        self.display_name = display_name if display_name is not None else internal_name
        self.address = address
        self.size = size

    @property
    def name(self) -> str:
        return self.internal_name

    def __getitem__(self, idx: int) -> int:
        """Element offset: fp_var[i] -> address + i"""
        if idx < 0 or idx >= self.size:
            raise IndexError(f"FPVar '{self.display_name}' index {idx} out of range [0, {self.size})")
        return self.address + idx

    def __repr__(self):
        return f"FPVar({self.display_name!r}, addr={self.address}, size={self.size})"


class VRAMMatrixVar(TensorVar):
    """
    VRAM matrix variable: large matrix allocated via alloc

    Used to store intermediate results (e.g., S block, PV, O).
    Supports sub-block indexed writes: `O[r][c] = ...`
    """

    def __init__(self, program: "PLENAProgram", name: str, shape: Tuple[int, int],
                 display_name: Optional[str] = None):
        super().__init__(program, name, "vram_matrix", shape, display_name=display_name)


# ============================================================================
# PLENAProgram Main Class
# ============================================================================

class PLENAProgram:
    """
    PLENA High-level Compiler Interface

    Wraps DeveloperCompiler with a Pythonic API.
    All operations are eager evaluation: ISA code is generated immediately upon call.
    """

    def __init__(self, mlen: int = 64, blen: int = 4, real_data_ratio: float = 1.125):
        """
        Args:
            mlen: Matrix tile size (default 64)
            blen: Vector tile size (default 4)
            real_data_ratio: HBM 数据比例（MXFP 格式 = 1.125）
        """
        self._compiler = DeveloperCompiler(mlen=mlen, blen=blen)
        self._mlen = mlen
        self._blen = blen
        self._real_data_ratio = real_data_ratio

        # HBM 地址自动分配
        self._next_hbm_addr: int = 0
        self._hbm_free_blocks: List[Tuple[int, int]] = []  # (addr, size)

        # 变量注册表
        self._inputs: Dict[str, InputVar] = {}
        self._tensors: Dict[str, TensorVar] = {}
        self._fp_vars: Dict[str, FPVar] = {}
        self._functions: Dict[str, Callable] = {}
        self._registered_hbm_sub_matrices: Dict[str, bool] = {}
        self._registered_vram_sub_matrices: Dict[str, bool] = {}

        # 结果
        self._result_tensor: Optional[TensorVar] = None

        # Auto-generated name counter
        self._auto_name_counter: int = 0

        # Function scope namespace
        # Push a prefix on each function call (e.g., "linear_0/"), pop on exit
        # _auto_name will automatically add current scope prefix, avoiding name conflicts when calling the same function multiple times
        self._scope_stack: List[str] = []
        self._function_call_counters: Dict[str, int] = {}  # func_name -> call count

    # ========================================================================
    # Property Access
    # ========================================================================

    @property
    def mlen(self) -> int:
        return self._mlen

    @property
    def blen(self) -> int:
        return self._blen

    @property
    def compiler(self) -> DeveloperCompiler:
        """Access underlying DeveloperCompiler (advanced usage)"""
        return self._compiler

    @property
    def symbol_table(self):
        """Access symbol table"""
        return self._compiler.get_symbol_table()

    # ========================================================================
    # Input Declaration
    # ========================================================================

    def input(self, name: str, shape: Tuple[int, int],
              hbm_addr: Optional[int] = None) -> InputVar:
        """
        Declare an input tensor (in HBM)

        Args:
            name: tensor 名称
            shape: (height, width) 形状
            hbm_addr: HBM 地址（None = 自动分配）

        Returns:
            InputVar 代理对象
        """
        h, w = shape
        size = h * w
        hbm_size = int(size * self._real_data_ratio)

        # 自动分配 HBM 地址
        if hbm_addr is None:
            hbm_addr = self._allocate_hbm(hbm_size)

        var = InputVar(self, name, shape, hbm_addr, hbm_size)
        self._inputs[name] = var
        self._compiler.add_hbm_object(
            name=name,
            hbm_addr=hbm_addr,
            shape=shape,
            real_data_ratio=self._real_data_ratio,
        )
        return var

    # ========================================================================
    # Load Operations
    # ========================================================================

    def load_batch(
        self,
        input_var: InputVar,
        name: Optional[str] = None,
    ) -> VRAMMatrixVar:
        """
        Load tensor from HBM to VRAM (Batch type)

        Generates ISA: HBM → VRAM prefetch

        Args:
            input_var: 输入变量（必须是 InputVar）
            name: 结果名称（None = 使用输入名称 + "_batch"）

        Returns:
            VRAMMatrixVar 代理对象
        """
        if not isinstance(input_var, InputVar):
            raise TypeError(f"Expected InputVar, got {type(input_var)}")

        # 双层命名：display_name 给用户看，internal_name 给系统用
        display_name = name if name is not None else input_var.display_name
        internal_name = self._scoped_name(display_name)

        # 调用 DeveloperCompiler 生成 ISA（HBM 来源与 VRAM 目标使用不同名字）
        self._compiler.load_batch(
            hbm_object_name=input_var.name,
            vram_object_name=internal_name,
            vlen=64,
            preload_len=4
        )

        var = VRAMMatrixVar(self, internal_name, input_var.shape, display_name=display_name)
        self._tensors[internal_name] = var
        return var

    # ========================================================================
    # Store Operations
    # ========================================================================

    def store(self, tensor_var, name: Optional[str] = None,
              hbm_addr: Optional[int] = None) -> InputVar:
        """
        Write tensor from VRAM back to HBM

        Args:
            tensor_var: 要存储的变量（VRAMMatrixVar）
            name: 存储名称（之后可通过此名称 load）
            hbm_addr: 目标 HBM 地址（None = 自动分配）

        Returns:
            InputVar 代理对象（可以之后 load 回来）
        """
        if not isinstance(tensor_var, VRAMMatrixVar):
            raise TypeError(f"Store requires VRAMMatrixVar, got {type(tensor_var)}")

        display_name = name if name is not None else f"{tensor_var.display_name}_stored"
        internal_name = self._scoped_name(display_name)

        # 确定 HBM 地址
        if hbm_addr is None:
            h, w = tensor_var.shape
            size = h * w
            hbm_size = int(size * self._real_data_ratio)
            hbm_addr = self._allocate_hbm(hbm_size)
        else:
            h, w = tensor_var.shape
            hbm_size = int(h * w * self._real_data_ratio)

        # 生成 ISA（使用源 tensor 的 internal name）
        self._compiler.store_to_hbm(
            tensor_name=tensor_var.name,  # 这里用 internal name 查 symbol table
            hbm_addr=hbm_addr,
            hbm_object_name=internal_name,
        )

        # 返回 InputVar（可以之后 load 回来）
        var = InputVar(self, internal_name, tensor_var.shape, hbm_addr, hbm_size,
                       display_name=display_name)
        self._inputs[internal_name] = var
        return var

    # ========================================================================
    # VRAM Matrix Allocation
    # ========================================================================

    def alloc(self, name: str, rows: int, cols: int) -> VRAMMatrixVar:
        """
        Allocate a VRAM matrix

        Used to store intermediate results (e.g., S block, PV, O).
        Within function scope, names are automatically prefixed to avoid conflicts.

        Args:
            name: 矩阵名称（用户看到的名字）
            rows: 行数
            cols: 列数

        Returns:
            VRAMMatrixVar 代理对象
        """
        display_name = name
        internal_name = self._scoped_name(name)
        self._compiler.allocate_vram_matrix(name=internal_name, rows=rows, cols=cols)

        var = VRAMMatrixVar(self, internal_name, (rows, cols), display_name=display_name)
        self._tensors[internal_name] = var
        return var

    def free_tensor(self, tensor_var: TensorVar):
        """
        Free a tensor in VRAM, reclaiming space for subsequent allocations

        Used to free intermediate results that are no longer needed, saving VRAM space.
        Freed space can be reused by new alloc() or other operations.

        Args:
            tensor_var: 要释放的 tensor（必须是 VRAMMatrixVar）

        Example:
            y1 = linear(x1, w_sub, 0)
            prog.free_tensor(y1)  # 不再需要 y1，释放空间
            y2 = linear(x2, w_sub, 1)  # y2 可以重用 y1 的空间
        """
        if not isinstance(tensor_var, VRAMMatrixVar):
            raise TypeError(f"Can only free VRAMMatrixVar, got {type(tensor_var)}")
        
        # 使用 internal_name 释放 VRAM
        self._compiler.free_vram_object(tensor_var.name, strict=False)
        # Keep sub-matrix registration state consistent after free.
        self._registered_vram_sub_matrices[tensor_var.name] = False

    def free_input(self, input_var: InputVar):
        """
        Free an InputVar bookkeeping and recycle its HBM range for future auto-allocation.

        Notes:
        - This only affects PLENAProgram's address management state.
        - If a freed input is referenced again later, caller is responsible for correctness.
        """
        if not isinstance(input_var, InputVar):
            raise TypeError(f"Can only free InputVar, got {type(input_var)}")

        self._compiler.free_hbm_object(input_var.name, strict=False)
        self._registered_hbm_sub_matrices[input_var.name] = False
        self._recycle_hbm(input_var.hbm_addr, input_var.hbm_size)
        self._inputs.pop(input_var.name, None)

    def free_fp_var(self, fp_var: FPVar):
        """
        Free an FPVar and return its block to FPRAM free pool.
        """
        if not isinstance(fp_var, FPVar):
            raise TypeError(f"Can only free FPVar, got {type(fp_var)}")
        self.free_fpram(fp_var.name, strict=True)

    # ========================================================================
    # Normalization Operations
    # ========================================================================

    def norm(
        self,
        tensor_var: TensorVar,
        mode: str = "rms",
        eps_offset: int = 1,
        reci_hid_offset: int = 2,
        vlen: Optional[int] = None,
        scratchpad_vram_addr: Optional[int] = None,
    ) -> TensorVar:
        """
        Normalize tensor in-place.

        Args:
            tensor_var: tensor to normalize (must have VRAM backing, e.g., VRAMMatrixVar)
            mode: "rms" or "layer"
            eps_offset: FPRAM address of epsilon
            reci_hid_offset: FPRAM address of 1/hidden_dim
            vlen: vector length (default: program mlen)
            scratchpad_vram_addr: optional scratchpad VRAM address

        Returns:
            The same tensor_var (in-place operation)
        """
        if not isinstance(tensor_var, VRAMMatrixVar):
            raise TypeError(f"norm requires VRAMMatrixVar, got {type(tensor_var)}")

        self._compiler.normalize(
            tensor_name=tensor_var.name,
            mode=mode,
            eps_offset=eps_offset,
            reci_hid_offset=reci_hid_offset,
            vlen=vlen,
            scratchpad_vram_addr=scratchpad_vram_addr,
        )
        return tensor_var

    def rms_norm(
        self,
        tensor_var: TensorVar,
        eps_offset: int = 1,
        reci_hid_offset: int = 2,
        vlen: Optional[int] = None,
        scratchpad_vram_addr: Optional[int] = None,
    ) -> TensorVar:
        """RMS normalization (in-place)."""
        return self.norm(
            tensor_var=tensor_var,
            mode="rms",
            eps_offset=eps_offset,
            reci_hid_offset=reci_hid_offset,
            vlen=vlen,
            scratchpad_vram_addr=scratchpad_vram_addr,
        )

    def layer_norm(
        self,
        tensor_var: TensorVar,
        eps_offset: int = 1,
        reci_hid_offset: int = 2,
        vlen: Optional[int] = None,
        scratchpad_vram_addr: Optional[int] = None,
    ) -> TensorVar:
        """Layer normalization (in-place)."""
        return self.norm(
            tensor_var=tensor_var,
            mode="layer",
            eps_offset=eps_offset,
            reci_hid_offset=reci_hid_offset,
            vlen=vlen,
            scratchpad_vram_addr=scratchpad_vram_addr,
        )

    # ========================================================================
    # FP Variable (FPRAM)
    # ========================================================================

    def allocate_fpram(
        self,
        internal_name: str,
        size: int = 1,
        display_name: Optional[str] = None,
    ) -> FPVar:
        """
        Allocate FPRAM with explicit internal name and return FPVar proxy.
        """
        if size <= 0:
            raise ValueError(f"FPRAM allocation size must be positive, got {size}")

        address = self._compiler.allocate_fpram(internal_name, size)
        var = FPVar(
            self,
            internal_name,
            address,
            size,
            display_name=display_name if display_name is not None else internal_name,
        )
        self._fp_vars[internal_name] = var
        return var

    def free_fpram(self, internal_name: str, strict: bool = True):
        """
        Free FPRAM allocation by internal name.
        """
        self._compiler.free_fpram(internal_name, strict=strict)
        self._fp_vars.pop(internal_name, None)

    def fp_var(self, name: str, size: int = 1) -> FPVar:
        """
        Declare an FP variable in FPRAM

        Allocates a contiguous region in FPRAM and returns an FPVar proxy.
        Within function scope, names are automatically prefixed.

        Args:
            name: variable name
            size: number of f16 elements to allocate (default 1)

        Returns:
            FPVar proxy object (use .address for ISA generation)

        Example:
            scale = prog.fp_var("scale")          # 1 element
            m_old = prog.fp_var("m_old", size=64) # 64 elements
            prog.compiler  # access compiler for ISA if needed
        """
        display_name = name
        internal_name = self._scoped_name(name)

        return self.allocate_fpram(
            internal_name=internal_name,
            size=size,
            display_name=display_name,
        )

    def save_fpram_state(self) -> int:
        """Save FPRAM allocator snapshot"""
        return self._compiler.save_fpram_state()

    def restore_fpram_state(self, snapshot: int):
        """Restore FPRAM allocator snapshot"""
        self._compiler.restore_fpram_state(snapshot)
        # Remove FPVar proxies that are no longer allocated in allocator.
        allocations = set(self._compiler.list_fpram_allocations())
        to_remove = [n for n in self._fp_vars if n not in allocations]
        for n in to_remove:
            del self._fp_vars[n]

    # ========================================================================
    # FPRAM Tile Operations
    # ========================================================================

    def _resolve_fpram_addr(self, addr_or_var: Union[int, FPVar], offset: int = 0) -> int:
        if isinstance(addr_or_var, FPVar):
            if offset < 0 or offset >= addr_or_var.size:
                raise ValueError(
                    f"FPVar offset out of range: offset={offset}, size={addr_or_var.size}, var={addr_or_var.name}"
                )
            return addr_or_var.address + offset
        if not isinstance(addr_or_var, int):
            raise TypeError(f"Expected int or FPVar, got {type(addr_or_var)}")
        return addr_or_var + offset

    def _resolve_rows(
        self,
        row_idx: Optional[int] = None,
        rows: Optional[List[int]] = None,
    ) -> List[int]:
        if row_idx is not None and rows is not None:
            raise ValueError("Provide either row_idx or rows, not both")
        if rows is not None:
            return rows
        if row_idx is not None:
            return [row_idx]
        return list(range(self._mlen))

    def tile_row_max(
        self,
        target_fpram_addr: Union[int, FPVar],
        source: VRAMMatrixVar,
        row_idx: Optional[int] = None,
        rows: Optional[List[int]] = None,
        target_offset: int = 0,
        target_base_offset: int = 0,
    ):
        """
        Tile Row Max: reduce a single row to max, store to FPRAM address.

        Args:
            target_fpram_addr: FPRAM address or FPVar to write result
            source: VRAM tile (mlen x mlen)
            row_idx: single row index (legacy path)
            rows: multiple row indices
            target_offset: element offset when target_fpram_addr is FPVar
            target_base_offset: base offset for multi-row writes (contiguous)

        Example:
            m = prog.fp_var("m", size=1)
            S = prog.alloc("S", 64, 64)
            for row in range(64):
                prog.tile_row_max(m, S, rows=list(range(64)))
        """
        resolved_rows = self._resolve_rows(row_idx=row_idx, rows=rows)
        if len(resolved_rows) == 1:
            offsets = [target_offset]
        else:
            offsets = [target_base_offset + i for i in range(len(resolved_rows))]
        row_map = [
            (r, self._resolve_fpram_addr(target_fpram_addr, off))
            for r, off in zip(resolved_rows, offsets)
        ]
        self._compiler.tile_row_max(
            source_matrix=source.name,
            row_map=row_map,
        )

    def tile_row_sum(
        self,
        target_fpram_addr: Union[int, FPVar],
        source: VRAMMatrixVar,
        row_idx: Optional[int] = None,
        rows: Optional[List[int]] = None,
        target_offset: int = 0,
        target_base_offset: int = 0,
    ):
        """
        Tile Row Sum: reduce a single row to sum, store to FPRAM address.

        Args:
            target_fpram_addr: FPRAM address or FPVar to write result
            source: VRAM tile (mlen x mlen)
            row_idx: single row index (legacy path)
            rows: multiple row indices
            target_offset: element offset when target_fpram_addr is FPVar
            target_base_offset: base offset for multi-row writes (contiguous)
        """
        resolved_rows = self._resolve_rows(row_idx=row_idx, rows=rows)
        if len(resolved_rows) == 1:
            offsets = [target_offset]
        else:
            offsets = [target_base_offset + i for i in range(len(resolved_rows))]
        row_map = [
            (r, self._resolve_fpram_addr(target_fpram_addr, off))
            for r, off in zip(resolved_rows, offsets)
        ]
        self._compiler.tile_row_sum(source.name, row_map)

    def tile_row_exp(
        self,
        source: VRAMMatrixVar,
        row_idx: Optional[int] = None,
        rows: Optional[List[int]] = None,
    ):
        """
        Tile Row Exp: in-place exp on specified rows.

        For each row i: source[i, :] = exp(source[i, :])
        """
        resolved_rows = self._resolve_rows(row_idx=row_idx, rows=rows)
        self._compiler.tile_row_exp(source.name, resolved_rows)

    def tile_row_reci(
        self,
        source: VRAMMatrixVar,
        rows: Optional[List[int]] = None,
    ):
        """
        Tile Row Reciprocal: in-place 1/x on specified rows.

        For each row i: source[i, :] = 1.0 / source[i, :]
        """
        if rows is None:
            rows = list(range(self._mlen))
        self._compiler.tile_row_reci(source.name, rows)

    def tile_row_sub_fp(
        self,
        source: VRAMMatrixVar,
        fpram_addr: Union[int, FPVar],
        row_idx: Optional[int] = None,
        rows: Optional[List[int]] = None,
        fpram_offset: int = 0,
        fpram_base_offset: int = 0,
    ):
        """
        Tile Row Sub FP: subtract FPRAM scalar from a single row.

        For row i: source[i, :] = source[i, :] - FPRAM[fpram_addr]
        """
        resolved_rows = self._resolve_rows(row_idx=row_idx, rows=rows)
        if len(resolved_rows) == 1:
            offsets = [fpram_offset]
        else:
            offsets = [fpram_base_offset + i for i in range(len(resolved_rows))]
        row_map = [
            (r, self._resolve_fpram_addr(fpram_addr, off))
            for r, off in zip(resolved_rows, offsets)
        ]
        self._compiler.tile_row_sub_fp(source.name, row_map)

    def tile_row_mul_fp(
        self,
        source: VRAMMatrixVar,
        fpram_addr: Union[int, FPVar],
        row_idx: Optional[int] = None,
        rows: Optional[List[int]] = None,
        fpram_offset: int = 0,
        fpram_base_offset: int = 0,
    ):
        """
        Tile Row Mul FP: multiply a single row by FPRAM scalar.

        For row i: source[i, :] = source[i, :] * FPRAM[fpram_addr]
        """
        resolved_rows = self._resolve_rows(row_idx=row_idx, rows=rows)
        if len(resolved_rows) == 1:
            offsets = [fpram_offset]
        else:
            offsets = [fpram_base_offset + i for i in range(len(resolved_rows))]
        row_map = [
            (r, self._resolve_fpram_addr(fpram_addr, off))
            for r, off in zip(resolved_rows, offsets)
        ]
        self._compiler.tile_row_mul_fp(source.name, row_map)

    def tile_row_add_fp(
        self,
        source: VRAMMatrixVar,
        fp_var: FPVar,
        rows: Optional[List[int]] = None,
    ):
        """
        Tile Row Add FP: add FPRAM scalar to each specified row.

        For each row i: source[i, :] = source[i, :] + fp_var[i]
        """
        if rows is None:
            rows = list(range(self._mlen))
        row_map = [(r, fp_var[r]) for r in rows]
        self._compiler.tile_row_add_fp(source.name, row_map)

    def tile_row_add(
        self,
        dst: VRAMMatrixVar,
        src: VRAMMatrixVar,
        rows: Optional[List[int]] = None,
    ):
        """
        Tile Row Add: dst[i, :] += src[i, :] for specified rows.
        """
        if rows is None:
            rows = list(range(self._mlen))
        self._compiler.tile_row_add(dst.name, src.name, rows)

    def tile_row_sub(
        self,
        dst: VRAMMatrixVar,
        src: VRAMMatrixVar,
        rows: Optional[List[int]] = None,
    ):
        """
        Tile Row Sub: dst[i, :] -= src[i, :] for specified rows.
        """
        if rows is None:
            rows = list(range(self._mlen))
        self._compiler.tile_row_sub(dst.name, src.name, rows)

    def tile_row_mul(
        self,
        dst: VRAMMatrixVar,
        src: VRAMMatrixVar,
        rows: Optional[List[int]] = None,
    ):
        """
        Tile Row Mul: dst[i, :] *= src[i, :] for specified rows.
        """
        if rows is None:
            rows = list(range(self._mlen))
        self._compiler.tile_row_mul(dst.name, src.name, rows)

    def fpvar_reci(
        self,
        src: FPVar,
        dst: FPVar,
        count: Optional[int] = None,
    ):
        """
        FPVar Reciprocal: compute 1/x for FPRAM scalar array.

        For each element i: dst[i] = 1.0 / src[i]

        Args:
            src: source FPVar
            dst: destination FPVar
            count: number of elements (default: min(src.size, dst.size))

        Example:
            l = prog.fp_var("l", size=64)
            inv_l = prog.fp_var("inv_l", size=64)
            prog.fpvar_reci(l, inv_l)  # inv_l = 1/l
        """
        if count is None:
            count = min(src.size, dst.size)
        if count > src.size or count > dst.size:
            raise ValueError(
                f"count={count} exceeds FPVar size: src.size={src.size}, dst.size={dst.size}"
            )
        self._compiler.fpram_reci(src.name, dst.name, count)

    def fpvar_max(
        self,
        src1: FPVar,
        src2: FPVar,
        dst: FPVar,
        count: Optional[int] = None,
    ):
        """
        FPVar Max: element-wise max for FPRAM scalar arrays.

        For each element i: dst[i] = max(src1[i], src2[i])

        Example:
            m_new = prog.fp_var("m_new", size=64)
            prog.fpvar_max(m_old, row_max, m_new)  # m_new = max(m_old, row_max)
        """
        if count is None:
            count = min(src1.size, src2.size, dst.size)
        self._compiler.fpram_max(src1.name, src2.name, dst.name, count)

    def fpvar_sub(
        self,
        src1: FPVar,
        src2: FPVar,
        dst: FPVar,
        count: Optional[int] = None,
    ):
        """
        FPVar Subtract: element-wise subtraction for FPRAM scalar arrays.

        For each element i: dst[i] = src1[i] - src2[i]

        Example:
            diff = prog.fp_var("diff", size=64)
            prog.fpvar_sub(m_old, m_new, diff)  # diff = m_old - m_new
        """
        if count is None:
            count = min(src1.size, src2.size, dst.size)
        self._compiler.fpram_sub(src1.name, src2.name, dst.name, count)

    def fpvar_exp(
        self,
        src: FPVar,
        dst: FPVar,
        count: Optional[int] = None,
    ):
        """
        FPVar Exp: element-wise exp for FPRAM scalar array.

        For each element i: dst[i] = exp(src[i])

        Example:
            m_res = prog.fp_var("m_res", size=64)
            prog.fpvar_exp(diff, m_res)  # m_res = exp(diff)
        """
        if count is None:
            count = min(src.size, dst.size)
        self._compiler.fpram_exp(src.name, dst.name, count)

    def fpvar_mul(
        self,
        src1: FPVar,
        src2: FPVar,
        dst: FPVar,
        count: Optional[int] = None,
    ):
        """
        FPVar Multiply: element-wise multiplication for FPRAM scalar arrays.

        For each element i: dst[i] = src1[i] * src2[i]

        Example:
            result = prog.fp_var("result", size=64)
            prog.fpvar_mul(l_old, m_res, result)  # result = l_old * m_res
        """
        if count is None:
            count = min(src1.size, src2.size, dst.size)
        self._compiler.fpram_mul(src1.name, src2.name, dst.name, count)

    def fpvar_add(
        self,
        src1: FPVar,
        src2: FPVar,
        dst: FPVar,
        count: Optional[int] = None,
    ):
        """
        FPVar Add: element-wise addition for FPRAM scalar arrays.

        For each element i: dst[i] = src1[i] + src2[i]

        Example:
            l_new = prog.fp_var("l_new", size=64)
            prog.fpvar_add(l_old, sum_p, l_new)  # l_new = l_old + sum_p
        """
        if count is None:
            count = min(src1.size, src2.size, dst.size)
        self._compiler.fpram_add(src1.name, src2.name, dst.name, count)

    def fpvar_copy(
        self,
        src: FPVar,
        dst: FPVar,
        count: Optional[int] = None,
    ):
        """
        FPVar Copy: copy FPRAM scalar array.

        For each element i: dst[i] = src[i]

        Example:
            m_old_saved = prog.fp_var("m_old_saved", size=64)
            prog.fpvar_copy(m_old, m_old_saved)  # backup m_old
        """
        if count is None:
            count = min(src.size, dst.size)
        self._compiler.fpram_copy(src.name, dst.name, count)

    def fpvar_sum(
        self,
        src: FPVar,
        dst: FPVar,
        count: Optional[int] = None,
    ):
        """
        FPVar Sum: reduction sum of src into dst[0] (via compiler FPRAM op).
        """
        if count is None:
            count = src.size
        self._compiler.fpram_sum(src.name, dst.name, count)

    def fpvar_shift(
        self,
        src: FPVar,
        dst: FPVar,
        shift: int,
        count: Optional[int] = None,
        fill: Optional[FPVar] = None,
    ):
        """
        FPVar Shift: shift src into dst, filling out-of-range slots with fill (default FPRAM zero).
        """
        if count is None:
            count = min(src.size, dst.size)
        fill_name = None if fill is None else fill.name
        self._compiler.fpram_shift(
            src_name=src.name,
            dst_name=dst.name,
            shift=shift,
            count=count,
            fill_fpram_name=fill_name,
        )

    def tile_row_mul_fp_broadcast(
        self,
        source: VRAMMatrixVar,
        fpram_scalar_addr: Union[int, FPVar],
        row_idx: Optional[int] = None,
        rows: Optional[List[int]] = None,
        fpram_offset: int = 0,
    ):
        """
        Tile Row Mul FP Broadcast: multiply a single row by a FPRAM scalar.

        For row i: source[i, :] = source[i, :] * FPRAM[fpram_scalar_addr]

        Args:
            source: VRAM tile (mlen x mlen)
            fpram_scalar_addr: FPRAM address or FPVar of the scalar
            row_idx: single row index (legacy path)
            rows: multiple row indices

        Example:
            scale_fp = prog.fp_var("scale", size=1)
            for row in range(64):
                prog.tile_row_mul_fp_broadcast(S, scale_fp, rows=list(range(64)))
        """
        resolved_rows = self._resolve_rows(row_idx=row_idx, rows=rows)
        scalar_addr = self._resolve_fpram_addr(fpram_scalar_addr, fpram_offset)
        self._compiler.tile_row_mul_fp_broadcast(source.name, scalar_addr, resolved_rows)

    def fpvar_fill_from_fpram(
        self,
        dst: FPVar,
        src_fpram_addr: int,
        count: Optional[int] = None,
    ):
        """
        FPVar Fill from FPRAM: fill all elements with a value from FPRAM.

        For each element i: dst[i] = FPRAM[src_fpram_addr]

        Args:
            dst: destination FPVar
            src_fpram_addr: source FPRAM address (e.g., 0 for 0.0, 2 for -inf)
            count: number of elements (default: dst.size)

        Example:
            m_old = prog.fp_var("m_old", size=64)
            prog.fpvar_fill_from_fpram(m_old, 2)  # fill with -inf from address 2
        """
        if count is None:
            count = dst.size
        self._compiler.fpram_fill_from_fpram(dst.name, src_fpram_addr, count)

    def vram_fill_zero(
        self,
        matrix: VRAMMatrixVar,
        rows: Optional[List[int]] = None,
    ):
        """
        VRAM Fill Zero: fill specified rows with 0.

        Args:
            matrix: VRAM matrix
            rows: which rows to fill (default: all rows)

        Example:
            O = prog.alloc("O", 128, 128)
            prog.vram_fill_zero(O, rows=range(64, 128))  # zero out second half
        """
        if rows is None:
            rows = list(range(matrix.shape[0]))
        self._compiler.vram_fill_zero(matrix.name, rows)

    def _ensure_hbm_sub_matrix_registered(self, input_var: InputVar):
        """Ensure an HBM input is registered in compiler sub-matrix manager."""
        if (
            input_var.name in self._registered_hbm_sub_matrices
            and self._registered_hbm_sub_matrices[input_var.name] is True
        ):
            return
        h, w = input_var.shape
        self._compiler.ensure_hbm_sub_matrix(
            name=input_var.name,
            hbm_addr=input_var.hbm_addr,
            shape=(h, w),
            real_data_ratio=self._real_data_ratio,
        )
        self._registered_hbm_sub_matrices[input_var.name] = True

    def _ensure_vram_sub_matrix_registered(self, matrix_var: VRAMMatrixVar):
        """Ensure a VRAM matrix is registered in compiler sub-matrix manager."""
        if (
            matrix_var.name in self._registered_vram_sub_matrices
            and self._registered_vram_sub_matrices[matrix_var.name] is True
        ):
            return
        self._compiler.ensure_vram_matrix_layout(
            name=matrix_var.name,
            shape=matrix_var.shape,
        )
        self._registered_vram_sub_matrices[matrix_var.name] = True

    def vram_sub_projection_to(
        self,
        vram_matrix: VRAMMatrixVar,
        vram_row_idx: int,
        mram_input: InputVar,
        mram_col_idx: int,
        target: VRAMMatrixVar,
        target_row_idx: int,
        target_col_idx: int,
        auto_reset_mram: bool = True,
    ):
        """
        target[target_row_idx][target_col_idx] = vram_matrix[vram_row_idx][:] @ mram_input[:][mram_col_idx]
        """
        if not isinstance(vram_matrix, VRAMMatrixVar):
            raise TypeError(f"vram_matrix must be VRAMMatrixVar, got {type(vram_matrix)}")
        if not isinstance(mram_input, InputVar):
            raise TypeError(f"mram_input must be InputVar, got {type(mram_input)}")
        if not isinstance(target, VRAMMatrixVar):
            raise TypeError(f"target must be VRAMMatrixVar, got {type(target)}")

        self._ensure_vram_sub_matrix_registered(vram_matrix)
        self._ensure_hbm_sub_matrix_registered(mram_input)
        if auto_reset_mram:
            self._compiler.reset_mram()
        self._compiler.load_sub_matrix_col(name=mram_input.name, col_idx=mram_col_idx)
        self._compiler.vram_sub_projection_to(
            vram_mat_name=vram_matrix.name,
            vram_row_idx=vram_row_idx,
            mram_mat_name=mram_input.name,
            mram_col_idx=mram_col_idx,
            target_matrix=target.name,
            target_row_idx=target_row_idx,
            target_col_idx=target_col_idx,
        )

    def vram_sub_projection_T_to(
        self,
        vram_matrix: VRAMMatrixVar,
        vram_row_idx: int,
        mram_input: InputVar,
        mram_row_idx: int,
        target: VRAMMatrixVar,
        target_row_idx: int,
        target_col_idx: int,
        auto_reset_mram: bool = True,
    ):
        """
        target[target_row_idx][target_col_idx] = vram_matrix[vram_row_idx][:] @ mram_input[mram_row_idx][:]^T
        """
        if not isinstance(vram_matrix, VRAMMatrixVar):
            raise TypeError(f"vram_matrix must be VRAMMatrixVar, got {type(vram_matrix)}")
        if not isinstance(mram_input, InputVar):
            raise TypeError(f"mram_input must be InputVar, got {type(mram_input)}")
        if not isinstance(target, VRAMMatrixVar):
            raise TypeError(f"target must be VRAMMatrixVar, got {type(target)}")

        self._ensure_vram_sub_matrix_registered(vram_matrix)
        self._ensure_hbm_sub_matrix_registered(mram_input)
        if auto_reset_mram:
            self._compiler.reset_mram()
        self._compiler.load_sub_matrix_row(name=mram_input.name, row_idx=mram_row_idx)
        self._compiler.vram_sub_projection_T_to(
            vram_mat_name=vram_matrix.name,
            vram_row_idx=vram_row_idx,
            mram_mat_name=mram_input.name,
            mram_row_idx=mram_row_idx,
            target_matrix=target.name,
            target_row_idx=target_row_idx,
            target_col_idx=target_col_idx,
        )

    # ========================================================================
    # VRAM 矩阵加法
    # ========================================================================

    def vram_add(self, dst: VRAMMatrixVar, src: VRAMMatrixVar,
                 dst_row_offset: int = 0, src_row_offset: int = 0,
                 num_rows: Optional[int] = None):
        """
        VRAM 矩阵加法：dst[row_offset:] += src

        Args:
            dst: 目标矩阵
            src: 源矩阵
            dst_row_offset: 目标行偏移
            src_row_offset: 源行偏移
            num_rows: 处理行数
        """
        self._compiler.vram_matrix_add(
            dst_matrix=dst.name,
            src_matrix=src.name,
            dst_row_offset=dst_row_offset,
            src_row_offset=src_row_offset,
            num_rows=num_rows,
        )

    def vram_block_add_to(
        self,
        src1: TensorVar,
        src1_row_idx: int,
        src1_col_idx: int,
        src2: TensorVar,
        src2_row_idx: int,
        src2_col_idx: int,
        target: TensorVar,
        target_row_idx: int,
        target_col_idx: int,
    ):
        """
        mlen x mlen block add:
            target[target_row_idx][target_col_idx] =
                src1[src1_row_idx][src1_col_idx] + src2[src2_row_idx][src2_col_idx]

        Supports writing back to the same matrix/block (in-place overwrite).
        """
        allowed = (VRAMMatrixVar,)
        if not isinstance(src1, allowed):
            raise TypeError(f"src1 must be VRAMMatrixVar, got {type(src1)}")
        if not isinstance(src2, allowed):
            raise TypeError(f"src2 must be VRAMMatrixVar, got {type(src2)}")
        if not isinstance(target, allowed):
            raise TypeError(f"target must be VRAMMatrixVar, got {type(target)}")

        self._compiler.vram_block_add_to(
            src1_matrix=src1.name,
            src1_row_idx=src1_row_idx,
            src1_col_idx=src1_col_idx,
            src2_matrix=src2.name,
            src2_row_idx=src2_row_idx,
            src2_col_idx=src2_col_idx,
            target_matrix=target.name,
            target_row_idx=target_row_idx,
            target_col_idx=target_col_idx,
        )

    # ========================================================================
    # Flash Attention Operations
    # ========================================================================

    def init_online_softmax(self, q_idx: int, o_matrix: VRAMMatrixVar):
        """Initialize Online Softmax state: m=-inf, l=0, O_row=0"""
        o_info = self._compiler.get_tensor_info(o_matrix.name)
        seq_len, head_dim = o_info.shape

        self._compiler.init_online_softmax(
            q_idx=q_idx,
            o_matrix=o_matrix.name,
            seq_len=seq_len,
            head_dim=head_dim,
        )

    def online_softmax_block(self, s_block: VRAMMatrixVar, scale: float):
        """Perform Online Softmax on S block"""
        self._compiler.online_softmax_block(
            s_block_matrix=s_block.name,
            scale=scale,
        )

    def compute_pv(
        self,
        s_block: VRAMMatrixVar,
        v_input: InputVar,
        k_idx: int,
        pv_matrix: VRAMMatrixVar,
        head_dim: int,
    ):
        """Compute PV = P @ V[k_idx] where P is stored in s_block."""
        if not isinstance(s_block, VRAMMatrixVar):
            raise TypeError(f"s_block must be VRAMMatrixVar, got {type(s_block)}")
        if not isinstance(v_input, InputVar):
            raise TypeError(f"v_input must be InputVar, got {type(v_input)}")
        if not isinstance(pv_matrix, VRAMMatrixVar):
            raise TypeError(f"pv_matrix must be VRAMMatrixVar, got {type(pv_matrix)}")

        self._ensure_hbm_sub_matrix_registered(v_input)
        self._compiler.compute_pv(
            s_block_matrix=s_block.name,
            v_sub_matrix=v_input.name,
            k_idx=k_idx,
            pv_matrix=pv_matrix.name,
            head_dim=head_dim,
        )

    def scale_o_row(self, o_matrix: VRAMMatrixVar, q_idx: int):
        """Scale current row block of O by m_res"""
        o_info = self._compiler.get_tensor_info(o_matrix.name)
        seq_len, head_dim = o_info.shape

        self._compiler.scale_o_row(
            o_matrix=o_matrix.name,
            q_idx=q_idx,
            seq_len=seq_len,
            head_dim=head_dim,
        )

    def final_scale_o(self, q_idx: int, o_matrix: VRAMMatrixVar):
        """Final scaling: O[q_idx] = O[q_idx] / l"""
        o_info = self._compiler.get_tensor_info(o_matrix.name)
        seq_len, head_dim = o_info.shape

        self._compiler.final_scale_o(
            q_idx=q_idx,
            o_matrix=o_matrix.name,
            seq_len=seq_len,
            head_dim=head_dim,
        )

    # ========================================================================
    # Function Decorator
    # ========================================================================

    def function(self, func: Callable) -> Callable:
        """
        Decorator: Define reusable functions

        被装饰的函数可以接受 TensorVar 参数，返回 TensorVar 结果。
        每次调用都会生成新的 ISA 代码（eager evaluation）。

        ⚠️ 命名空间机制：
        每次调用函数时，内部产生的中间 tensor 会自动加上作用域前缀，
        确保同一函数多次调用时名字不冲突。

        例如 linear 被调用两次：
        - 第 1 次调用：中间结果名为 "linear_0/proj_1"
        - 第 2 次调用：中间结果名为 "linear_1/proj_1"

        支持嵌套函数调用：
            @prog.function
            def two_layer(x, w1, w2):
                h = linear(x, w1)
                return linear(h, w2)

            # 嵌套命名：two_layer_0/linear_0/proj_1, two_layer_0/linear_1/proj_1
        """
        func_name = func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs):
            # 分配调用编号
            call_idx = self._function_call_counters.get(func_name, 0)
            self._function_call_counters[func_name] = call_idx + 1

            # Push 作用域：如 "linear_0/"
            scope = f"{func_name}_{call_idx}/"
            self._scope_stack.append(scope)

            # 在生成的 ISA 中插入注释
            self._compiler.generated_code += f"; === Enter {func_name} (call #{call_idx}) ===\n"

            # Snapshot: record existing tensors before function execution
            tensors_before = set(self._tensors.keys())
            inputs_before = set(self._inputs.keys())
            fp_vars_before = set(self._fp_vars.keys())

            try:
                result = func(*args, **kwargs)

                # Auto-free: free locally allocated tensors that are not returned
                return_names = set()
                return_fp_names = set()
                if isinstance(result, TensorVar):
                    return_names.add(result.internal_name)
                elif isinstance(result, FPVar):
                    return_fp_names.add(result.internal_name)
                elif isinstance(result, (tuple, list)):
                    for r in result:
                        if isinstance(r, TensorVar):
                            return_names.add(r.internal_name)
                        elif isinstance(r, FPVar):
                            return_fp_names.add(r.internal_name)

                for name in set(self._tensors.keys()) - tensors_before:
                    if name not in return_names:
                        tensor = self._tensors[name]
                        if isinstance(tensor, VRAMMatrixVar):
                            self.free_tensor(tensor)
                            self._registered_vram_sub_matrices[tensor.name] = False

                for name in set(self._inputs.keys()) - inputs_before:
                    if name not in return_names:
                        self.free_input(self._inputs[name])

                local_fp_names = sorted(
                    set(self._fp_vars.keys()) - fp_vars_before,
                    key=lambda n: self._fp_vars[n].address,
                    reverse=True,
                )
                for name in local_fp_names:
                    if name in return_fp_names:
                        continue
                    fp_var = self._fp_vars.get(name)
                    if fp_var is not None:
                        self.free_fp_var(fp_var)
            finally:
                # Pop 作用域
                self._scope_stack.pop()
                self._compiler.generated_code += f"; === Exit {func_name} (call #{call_idx}) ===\n"

            return result

        self._functions[func_name] = wrapper
        wrapper._plena_function = True
        wrapper._plena_name = func_name
        return wrapper

    # ========================================================================
    # Result Marking
    # ========================================================================

    def result(self, tensor_var: TensorVar):
        """
        Mark output result

        Args:
            tensor_var: 结果 tensor
        """
        self._result_tensor = tensor_var

    # ========================================================================
    # Compilation
    # ========================================================================

    def compile(self) -> str:
        """
        Get generated ISA code

        Returns:
            完整的 ISA 代码字符串
        """
        return self._compiler.get_code()

    def print_symbol_table(self):
        """Print symbol table"""
        self._compiler.print_symbol_table()

    def get_symbol_table(self):
        """Get symbol table"""
        return self._compiler.get_symbol_table()

    # ========================================================================
    # 运算符分派（内部方法）
    # ========================================================================

    def _dispatch_matmul(self, left: TensorVar, right) -> TensorVar:
        raise TypeError(
            "@ operator is no longer supported in PLENAProgram. "
            "Use explicit program APIs instead."
        )

    # ========================================================================
    # 工具方法
    # ========================================================================

    def _scoped_name(self, name: str) -> str:
        """
        给用户显式传入的名字加上当前作用域前缀

        在函数作用域内，显式名字也会自动加前缀：
        - 顶层调用 alloc("temp"):             -> "temp"
        - linear 第 0 次调用 alloc("temp"):    -> "linear_0/temp"
        - 嵌套 two_layer→linear alloc("temp"): -> "two_layer_0/linear_0/temp"

        这样同一函数调用两次，内部的 "temp" 不会冲突。
        """
        if not self._scope_stack:
            return name
        scope_prefix = "".join(self._scope_stack)
        return f"{scope_prefix}{name}"

    def _allocate_hbm(self, hbm_size: int) -> int:
        """Allocate HBM range, preferring previously freed blocks."""
        best_idx = None
        best_waste = None
        for i, (addr, size) in enumerate(self._hbm_free_blocks):
            if size >= hbm_size:
                waste = size - hbm_size
                if best_waste is None or waste < best_waste:
                    best_idx = i
                    best_waste = waste

        if best_idx is not None:
            addr, _ = self._hbm_free_blocks.pop(best_idx)
            return addr

        addr = self._next_hbm_addr
        self._next_hbm_addr = ((addr + hbm_size + 63) // 64) * 64
        return addr

    def _recycle_hbm(self, hbm_addr: int, hbm_size: int):
        """Recycle an HBM range for future auto-allocation."""
        if hbm_size <= 0:
            return
        self._hbm_free_blocks.append((hbm_addr, hbm_size))

    def _auto_name(self, prefix: str = "t") -> str:
        """
        生成自动名称（带作用域前缀）

        在函数作用域内，名字会自动加前缀：
        - 顶层: "__proj_1"
        - linear 第 0 次调用: "linear_0/__proj_1"
        - 嵌套: "two_layer_0/linear_0/__proj_1"
        """
        self._auto_name_counter += 1
        scope_prefix = "".join(self._scope_stack)  # 拼接所有层级
        return f"{scope_prefix}__{prefix}_{self._auto_name_counter}"

    def __repr__(self):
        num_inputs = len(self._inputs)
        num_tensors = len(self._tensors)
        num_functions = len(self._functions)
        code_len = len(self._compiler.get_code().splitlines())
        return (
            f"PLENAProgram(mlen={self._mlen}, blen={self._blen}, "
            f"inputs={num_inputs}, tensors={num_tensors}, "
            f"functions={num_functions}, isa_lines={code_len})"
        )
