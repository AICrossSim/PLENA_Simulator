"""
PLENAProgram: High-level Compiler Interface - TileLang-style Python API

A Pythonic wrapper around DeveloperCompiler, supporting:
- TensorVar proxy objects with `A @ B` syntax (sub-matrix operations only)
- Pythonic sub-matrix indexing: `W_sub.col(i)`, `W_sub.row(i).T`
- @prog.function decorator for defining reusable functions
- Automatic HBM address allocation
- Dual naming mechanism (display_name + internal_name)

⚠️ Important: All matrix multiplications must be implemented via sub-matrix operations
    Direct BatchVar @ MatrixVar is not supported; matrices must be registered as sub-matrices first.

Usage Examples
==============

Sub-matrix Projection:
    prog = PLENAProgram(mlen=64, blen=4)

    X = prog.input("X", shape=(64, 128))
    W = prog.input("W", shape=(128, 128))

    A = prog.load_batch(X)
    W_sub = prog.register_sub_matrix(W)
    W_sub.load_col(0)
    
    C = A @ W_sub.col(0)    # sub_projection: A @ W[:, 0:64]

    prog.result(C)
    isa_code = prog.compile()

Sub-matrix Transpose Projection:
    prog = PLENAProgram(mlen=64, blen=4)

    X = prog.input("X", shape=(64, 128))
    W = prog.input("W", shape=(64, 128))  # Shape after transpose: (128, 64)

    A = prog.load_batch(X)
    W_sub = prog.register_sub_matrix(W)
    W_sub.load_row(0)
    
    C = A @ W_sub.row(0).T    # sub_projection_T: A @ W[0:64, :].T

    prog.result(C)
    isa_code = prog.compile()

Using Functions (Auto-scoped Naming):
    prog = PLENAProgram(mlen=64, blen=4)

    @prog.function
    def linear(act, weight_sub, col_idx):
        # Local variables are automatically prefixed to avoid conflicts
        temp = prog.alloc("temp", 64, 64)  # → "linear_0/temp", "linear_1/temp", ...
        return act @ weight_sub.col(col_idx)

    X = prog.input("X", shape=(64, 128))
    W = prog.input("W", shape=(128, 128))

    A = prog.load_batch(X)
    W_sub = prog.register_sub_matrix(W)
    W_sub.load_col(0)
    W_sub.load_col(1)

    Y1 = linear(A, W_sub, 0)    # First call
    Y2 = linear(A, W_sub, 1)    # Second call (no conflict)

    prog.result(Y2)
    isa_code = prog.compile()

Flash Attention (VRAM Sub-matrices):
    prog = PLENAProgram(mlen=64, blen=4)

    Q_in = prog.input("Q", shape=(seq_len, head_dim))
    K_in = prog.input("K", shape=(seq_len, head_dim))
    V_in = prog.input("V", shape=(seq_len, head_dim))

    Q = prog.load_batch(Q_in)
    Q_sub = prog.register_vram_sub_matrix(Q)
    K_sub = prog.register_sub_matrix(K_in)
    V_sub = prog.register_sub_matrix(V_in)

    S = prog.alloc("S", mlen, mlen)
    PV = prog.alloc("PV", mlen, head_dim)
    O = prog.alloc("O", seq_len, head_dim)

    for q_idx in range(num_q_blocks):
        prog.init_online_softmax(q_idx, O)
        for k_idx in range(num_k_blocks):
            prog.reset_mram()
            K_sub.load_row(k_idx)
            prog.vram_sub_projection_T_to(Q_sub.row(q_idx), K_sub.row(k_idx), S, 0, 0)
            prog.online_softmax_block(S, scale)
            prog.compute_pv(S, V_sub, k_idx, PV)
            prog.scale_o_row(O, q_idx)
            prog.vram_add(O, PV, q_idx * mlen)
        prog.final_scale_o(q_idx, O)

    prog.result(O)
    isa_code = prog.compile()
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable, Any, Union
from functools import wraps
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from developer_compiler import DeveloperCompiler
from symbol_table import TensorInfo
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

    @property
    def T(self):
        """Transpose marker: A @ B.T -> tmm_matmul"""
        return TransposedVar(self)

    def __repr__(self):
        if self.display_name != self.internal_name:
            return (f"{self.__class__.__name__}(display={self.display_name!r}, "
                    f"internal={self.internal_name!r}, shape={self.shape})")
        return f"{self.__class__.__name__}({self.display_name!r}, shape={self.shape})"


class TransposedVar:
    """
    Transpose marker proxy

    Used for `A @ B.T` syntax, recognized during __matmul__ dispatch for transpose operations.
    """

    def __init__(self, original: TensorVar):
        self.original = original
        self._program = original._program
        self.name = original.name
        self.kind = original.kind
        self.shape = (original.shape[1], original.shape[0])  # Transposed shape

    def __repr__(self):
        return f"Transposed({self.original})"


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


class BatchVar(TensorVar):
    """
    Batch variable: activation matrix already loaded to VRAM

    Supports `A @ B` matrix multiplication.
    """

    def __init__(self, program: "PLENAProgram", name: str, shape: Tuple[int, int],
                 display_name: Optional[str] = None):
        super().__init__(program, name, "batch", shape, display_name=display_name)


class MatrixVar(TensorVar):
    """
    Matrix variable: weight matrix declared in HBM

    Does not occupy VRAM space; loaded on-demand to MSRAM via H_PREFETCH_M during computation.
    """

    def __init__(self, program: "PLENAProgram", name: str, shape: Tuple[int, int],
                 display_name: Optional[str] = None):
        super().__init__(program, name, "matrix", shape, display_name=display_name)


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

    def __setitem__(self, key, value):
        """
        Supports target[row_idx, col_idx] = sub_projection_result
        """
        if isinstance(key, tuple) and len(key) == 2:
            row_idx, col_idx = key
            if isinstance(value, _DeferredSubProjection):
                value.execute_to(self, row_idx, col_idx)
            else:
                raise TypeError(f"Cannot assign {type(value)} to VRAMMatrixVar element")
        else:
            raise TypeError(f"Invalid index: {key}")


# ============================================================================
# Sub-matrix Proxy Objects
# ============================================================================

class SubMatrixColRef:
    """
    Sub-matrix column reference: SubMatrix(W, col_idx)

    Represents W[:, col_idx*mlen : (col_idx+1)*mlen]
    Used for sub_projection: A @ W_sub.col(0)
    """

    def __init__(self, sub_matrix: "SubMatrixVar", col_idx: int):
        self.sub_matrix = sub_matrix
        self.col_idx = col_idx
        self._program = sub_matrix._program

    def __repr__(self):
        return f"SubMatrixCol({self.sub_matrix.name}[:, {self.col_idx}])"


class SubMatrixRowRef:
    """
    Sub-matrix row reference: SubMatrix(W, row_idx) row

    Represents W[row_idx*mlen : (row_idx+1)*mlen, :]
    Used for sub_projection_T: A @ W_sub.row(0).T
    """

    def __init__(self, sub_matrix: "SubMatrixVar", row_idx: int):
        self.sub_matrix = sub_matrix
        self.row_idx = row_idx
        self._program = sub_matrix._program

    @property
    def T(self):
        """Transpose marker"""
        return TransposedSubMatrixRowRef(self)

    def __repr__(self):
        return f"SubMatrixRow({self.sub_matrix.name}[{self.row_idx}, :])"


class TransposedSubMatrixRowRef:
    """Transposed reference of sub-matrix row"""

    def __init__(self, row_ref: SubMatrixRowRef):
        self.row_ref = row_ref
        self.sub_matrix = row_ref.sub_matrix
        self.row_idx = row_ref.row_idx
        self._program = row_ref._program


class SubMatrixVar:
    """
    Sub-matrix variable: Large matrix registered with SubMatrixManager

    Supports loading sub-blocks by column or row, and performing sub-block projections.

    Usage:
        W_sub = prog.register_sub_matrix(W_input)
        W_sub.load_col(0)        # Load W[:][0] to MRAM
        W_sub.load_row(0)        # Load W[0][:] to MRAM
        C = A @ W_sub.col(0)     # sub_projection
        C = A @ W_sub.row(0).T   # sub_projection_T
    """

    def __init__(self, program: "PLENAProgram", internal_name: str, source_input: InputVar,
                 shape: Tuple[int, int], display_name: Optional[str] = None):
        self._program = program
        self.internal_name = internal_name   # System internal name (with scope prefix)
        self.display_name = display_name if display_name is not None else internal_name
        self.source_input = source_input
        self.shape = shape
        self.loaded_cols: List[int] = []
        self.loaded_rows: List[int] = []

    @property
    def name(self) -> str:
        """Compatibility property: returns internal_name"""
        return self.internal_name

    def load_col(self, col_idx: int):
        """Load entire column sub-blocks to MRAM: W[:][col_idx]"""
        self._program._compiler.load_sub_matrix_col(name=self.internal_name, col_idx=col_idx)
        self.loaded_cols.append(col_idx)

    def load_row(self, row_idx: int):
        """Load entire row sub-blocks to MRAM: W[row_idx][:]"""
        self._program._compiler.load_sub_matrix_row(name=self.internal_name, row_idx=row_idx)
        self.loaded_rows.append(row_idx)

    def col(self, col_idx: int) -> SubMatrixColRef:
        """Get column reference: W_sub.col(0) -> SubMatrix(W, 0) column sub-block"""
        if col_idx not in self.loaded_cols:
            raise ValueError(
                f"SubMatrix column {self.display_name}[:][{col_idx}] not loaded. "
                f"Call W_sub.load_col({col_idx}) first."
            )
        return SubMatrixColRef(self, col_idx)

    def row(self, row_idx: int) -> SubMatrixRowRef:
        """Get row reference: W_sub.row(0) -> SubMatrix(W, 0) row sub-block"""
        if row_idx not in self.loaded_rows:
            raise ValueError(
                f"SubMatrix row {self.display_name}[{row_idx}][:] not loaded. "
                f"Call W_sub.load_row({row_idx}) first."
            )
        return SubMatrixRowRef(self, row_idx)

    def __repr__(self):
        if self.display_name != self.internal_name:
            return (
                f"SubMatrixVar(display={self.display_name!r}, internal={self.internal_name!r}, "
                f"shape={self.shape}, loaded_cols={self.loaded_cols}, loaded_rows={self.loaded_rows})"
            )
        return (
            f"SubMatrixVar({self.display_name!r}, shape={self.shape}, "
            f"loaded_cols={self.loaded_cols}, loaded_rows={self.loaded_rows})"
        )


# ============================================================================
# VRAM Sub-matrix Proxy Objects
# ============================================================================

class VRAMSubMatrixRowRef:
    """
    VRAM sub-matrix row reference

    Represents a row sub-block in VRAM large matrix: A[row_idx*mlen : (row_idx+1)*mlen, :]
    """

    def __init__(self, vram_sub: "VRAMSubMatrixVar", row_idx: int):
        self.vram_sub = vram_sub
        self.row_idx = row_idx
        self._program = vram_sub._program

    def __matmul__(self, other):
        """
        VRAMSubMatrixRow @ SubMatrixColRef  ->  vram_sub_projection
        VRAMSubMatrixRow @ TransposedSubMatrixRowRef  ->  vram_sub_projection_T
        """
        return self._program._dispatch_vram_sub_matmul(self, other)

    def __repr__(self):
        return f"VRAMSubRow({self.vram_sub.name}[{self.row_idx}])"


class VRAMSubMatrixVar:
    """
    VRAM sub-matrix variable: VRAM matrix registered with SubMatrixManager

    Usage:
        Q_sub = prog.register_vram_sub_matrix(Q_batch)
        S = Q_sub.row(0) @ K_sub.row(0).T    # Q[0][:] @ K[0][:]^T
        C = Q_sub.row(0) @ W_sub.col(0)      # Q[0][:] @ W[:][0]
    """

    def __init__(self, program: "PLENAProgram", internal_name: str, source_tensor: str,
                 shape: Tuple[int, int], display_name: Optional[str] = None):
        self._program = program
        self.internal_name = internal_name   # System internal name (with scope prefix)
        self.display_name = display_name if display_name is not None else internal_name
        self.source_tensor = source_tensor
        self.shape = shape

    @property
    def name(self) -> str:
        """Compatibility property: returns internal_name"""
        return self.internal_name

    def row(self, row_idx: int) -> VRAMSubMatrixRowRef:
        """Get row reference: Q_sub.row(0) -> VRAM A[0][:]"""
        return VRAMSubMatrixRowRef(self, row_idx)

    def __repr__(self):
        if self.display_name != self.internal_name:
            return (f"VRAMSubMatrixVar(display={self.display_name!r}, "
                    f"internal={self.internal_name!r}, source={self.source_tensor})")
        return f"VRAMSubMatrixVar({self.display_name!r}, source={self.source_tensor})"


# ============================================================================
# Deferred Execution Objects (for target[r][c] = ... assignment syntax)
# ============================================================================

class _DeferredSubProjection:
    """
    Deferred sub-block projection

    When `Q_sub.row(i) @ K_sub.row(j).T` needs to be written to `S[r, c]`,
    this object is created first and executed upon assignment.
    """

    def __init__(self, program: "PLENAProgram", vram_row_ref: VRAMSubMatrixRowRef,
                 right_ref, op_type: str):
        self._program = program
        self.vram_row_ref = vram_row_ref
        self.right_ref = right_ref
        self.op_type = op_type  # "projection" or "projection_T"

    def execute_to(self, target: VRAMMatrixVar, target_row_idx: int, target_col_idx: int):
        """Execute sub-block projection and write to target matrix"""
        if self.op_type == "projection":
            self._program._compiler.vram_sub_projection_to(
                vram_mat_name=self.vram_row_ref.vram_sub.name,
                vram_row_idx=self.vram_row_ref.row_idx,
                mram_mat_name=self.right_ref.sub_matrix.name,
                mram_col_idx=self.right_ref.col_idx,
                target_matrix=target.name,
                target_row_idx=target_row_idx,
                target_col_idx=target_col_idx,
            )
        elif self.op_type == "projection_T":
            self._program._compiler.vram_sub_projection_T_to(
                vram_mat_name=self.vram_row_ref.vram_sub.name,
                vram_row_idx=self.vram_row_ref.row_idx,
                mram_mat_name=self.right_ref.sub_matrix.name,
                mram_row_idx=self.right_ref.row_idx,
                target_matrix=target.name,
                target_row_idx=target_row_idx,
                target_col_idx=target_col_idx,
            )


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

        # 变量注册表
        self._inputs: Dict[str, InputVar] = {}
        self._tensors: Dict[str, TensorVar] = {}
        self._sub_matrices: Dict[str, SubMatrixVar] = {}
        self._vram_sub_matrices: Dict[str, VRAMSubMatrixVar] = {}
        self._fp_vars: Dict[str, FPVar] = {}
        self._functions: Dict[str, Callable] = {}

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
        return self._compiler.symbol_table

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
            hbm_addr = self._next_hbm_addr
            self._next_hbm_addr = ((hbm_addr + hbm_size + 63) // 64) * 64

        var = InputVar(self, name, shape, hbm_addr, hbm_size)
        self._inputs[name] = var
        return var

    # ========================================================================
    # Load Operations
    # ========================================================================

    def load_batch(self, input_var: InputVar, name: Optional[str] = None) -> BatchVar:
        """
        Load tensor from HBM to VRAM (Batch type)

        Generates ISA: HBM → VRAM prefetch

        Args:
            input_var: 输入变量（必须是 InputVar）
            name: 结果名称（None = 使用输入名称 + "_batch"）

        Returns:
            BatchVar 代理对象
        """
        if not isinstance(input_var, InputVar):
            raise TypeError(f"Expected InputVar, got {type(input_var)}")

        # 双层命名：display_name 给用户看，internal_name 给系统用
        display_name = name if name is not None else input_var.display_name
        internal_name = self._scoped_name(display_name)

        h, w = input_var.shape

        # 调用 DeveloperCompiler 生成 ISA（使用 internal_name）
        self._compiler.load_batch(
            name=internal_name,
            hbm_addr=input_var.hbm_addr,
            h=h,
            w=w,
            real_data_ratio=self._real_data_ratio,
            vlen=64,
            preload_len=4
        )

        var = BatchVar(self, internal_name, input_var.shape, display_name=display_name)
        self._tensors[internal_name] = var
        return var

    def load(self, input_var: InputVar, name: Optional[str] = None) -> BatchVar:
        """
        从 HBM 加载之前 Store 过的数据到 VRAM

        Equivalent to load_batch, used to load previously stored intermediate results.

        Args:
            input_var: 输入变量（之前 store 返回的 InputVar）
            name: 结果名称

        Returns:
            BatchVar 代理对象
        """
        return self.load_batch(input_var, name=name)

    # ========================================================================
    # Store Operations
    # ========================================================================

    def store(self, tensor_var, name: Optional[str] = None,
              hbm_addr: Optional[int] = None) -> InputVar:
        """
        Write tensor from VRAM back to HBM

        Args:
            tensor_var: 要存储的变量（BatchVar 或 VRAMMatrixVar）
            name: 存储名称（之后可通过此名称 load）
            hbm_addr: 目标 HBM 地址（None = 自动分配）

        Returns:
            InputVar 代理对象（可以之后 load 回来）
        """
        if not isinstance(tensor_var, (BatchVar, VRAMMatrixVar)):
            raise TypeError(f"Store requires BatchVar or VRAMMatrixVar, got {type(tensor_var)}")

        display_name = name if name is not None else f"{tensor_var.display_name}_stored"
        internal_name = self._scoped_name(display_name)

        # 确定 HBM 地址
        if hbm_addr is None:
            h, w = tensor_var.shape
            size = h * w
            hbm_size = int(size * self._real_data_ratio)
            hbm_addr = self._next_hbm_addr
            self._next_hbm_addr = ((hbm_addr + hbm_size + 63) // 64) * 64
        else:
            h, w = tensor_var.shape
            hbm_size = int(h * w * self._real_data_ratio)

        # 生成 ISA（使用源 tensor 的 internal name）
        self._compiler.store_to_hbm(
            tensor_name=tensor_var.name,  # 这里用 internal name 查 symbol table
            hbm_addr=hbm_addr,
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
            tensor_var: 要释放的 tensor（必须是 BatchVar 或 VRAMMatrixVar）

        Example:
            y1 = linear(x1, w_sub, 0)
            prog.free_tensor(y1)  # 不再需要 y1，释放空间
            y2 = linear(x2, w_sub, 1)  # y2 可以重用 y1 的空间
        """
        if not isinstance(tensor_var, (BatchVar, VRAMMatrixVar)):
            raise TypeError(f"Can only free BatchVar or VRAMMatrixVar, got {type(tensor_var)}")
        
        # 使用 internal_name 释放 VRAM
        self._compiler.symbol_table.vram_allocator.free(tensor_var.name, strict=False)

    # ========================================================================
    # FP Variable (FPRAM)
    # ========================================================================

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

        address = self._compiler.allocate_fpram(internal_name, size)

        var = FPVar(self, internal_name, address, size, display_name=display_name)
        self._fp_vars[internal_name] = var
        return var

    def save_fpram_state(self) -> int:
        """Save FPRAM stack pointer for scoped allocation"""
        return self._compiler.save_fpram_state()

    def restore_fpram_state(self, snapshot: int):
        """Restore FPRAM stack pointer, freeing allocations after snapshot"""
        self._compiler.restore_fpram_state(snapshot)
        # Remove FPVars that were freed
        to_remove = [n for n, v in self._fp_vars.items() if v.address >= snapshot]
        for n in to_remove:
            del self._fp_vars[n]

    # ========================================================================
    # FPRAM Tile Operations
    # ========================================================================

    def tile_row_max(
        self,
        target_fpram_addr: int,
        source: VRAMMatrixVar,
        row_idx: int,
    ):
        """
        Tile Row Max: reduce a single row to max, store to FPRAM address.

        Args:
            target_fpram_addr: FPRAM address to write result
            source: VRAM tile (mlen x mlen)
            row_idx: which row to process (0 to mlen-1)

        Example:
            m = prog.fp_var("m", size=1)
            S = prog.alloc("S", 64, 64)
            for row in range(64):
                prog.tile_row_max(m.address, S, row)
        """
        source_info = self._compiler.symbol_table[source.name]
        source_vram_addr = source_info.vram_addr

        row_map = [(row_idx, target_fpram_addr)]

        self._compiler.tile_row_max_asm(
            source_vram_addr=source_vram_addr,
            row_map=row_map,
        )

    def tile_row_sum(
        self,
        target_fpram_addr: int,
        source: VRAMMatrixVar,
        row_idx: int,
    ):
        """
        Tile Row Sum: reduce a single row to sum, store to FPRAM address.

        Args:
            target_fpram_addr: FPRAM address to write result
            source: VRAM tile (mlen x mlen)
            row_idx: which row to process (0 to mlen-1)
        """
        source_vram_addr = self._compiler.symbol_table[source.name].vram_addr
        row_map = [(row_idx, target_fpram_addr)]
        self._compiler.tile_row_sum_asm(source_vram_addr, row_map)

    def tile_row_exp(
        self,
        source: VRAMMatrixVar,
        row_idx: int,
    ):
        """
        Tile Row Exp: in-place exp on a single row.

        For row i: source[i, :] = exp(source[i, :])
        """
        vram_addr = self._compiler.symbol_table[source.name].vram_addr
        self._compiler.tile_row_exp_asm(vram_addr, [row_idx])

    def tile_row_reci(
        self,
        source: VRAMMatrixVar,
        rows: Optional[List[int]] = None,
    ):
        """
        Tile Row Reciprocal: in-place 1/x on specified rows.

        For each row i: source[i, :] = 1.0 / source[i, :]
        """
        vram_addr = self._compiler.symbol_table[source.name].vram_addr
        if rows is None:
            rows = list(range(self._mlen))
        self._compiler.tile_row_reci_asm(vram_addr, rows)

    def tile_row_sub_fp(
        self,
        source: VRAMMatrixVar,
        fpram_addr: int,
        row_idx: int,
    ):
        """
        Tile Row Sub FP: subtract FPRAM scalar from a single row.

        For row i: source[i, :] = source[i, :] - FPRAM[fpram_addr]
        """
        vram_addr = self._compiler.symbol_table[source.name].vram_addr
        row_map = [(row_idx, fpram_addr)]
        self._compiler.tile_row_sub_fp_asm(vram_addr, row_map)

    def tile_row_mul_fp(
        self,
        source: VRAMMatrixVar,
        fpram_addr: int,
        row_idx: int,
    ):
        """
        Tile Row Mul FP: multiply a single row by FPRAM scalar.

        For row i: source[i, :] = source[i, :] * FPRAM[fpram_addr]
        """
        vram_addr = self._compiler.symbol_table[source.name].vram_addr
        row_map = [(row_idx, fpram_addr)]
        self._compiler.tile_row_mul_fp_asm(vram_addr, row_map)

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
        vram_addr = self._compiler.symbol_table[source.name].vram_addr
        if rows is None:
            rows = list(range(self._mlen))
        row_map = [(r, fp_var[r]) for r in rows]
        self._compiler.tile_row_add_fp_asm(vram_addr, row_map)

    def tile_row_add(
        self,
        dst: VRAMMatrixVar,
        src: VRAMMatrixVar,
        rows: Optional[List[int]] = None,
    ):
        """
        Tile Row Add: dst[i, :] += src[i, :] for specified rows.
        """
        dst_addr = self._compiler.symbol_table[dst.name].vram_addr
        src_addr = self._compiler.symbol_table[src.name].vram_addr
        if rows is None:
            rows = list(range(self._mlen))
        self._compiler.tile_row_add_asm(dst_addr, src_addr, rows)

    def tile_row_sub(
        self,
        dst: VRAMMatrixVar,
        src: VRAMMatrixVar,
        rows: Optional[List[int]] = None,
    ):
        """
        Tile Row Sub: dst[i, :] -= src[i, :] for specified rows.
        """
        dst_addr = self._compiler.symbol_table[dst.name].vram_addr
        src_addr = self._compiler.symbol_table[src.name].vram_addr
        if rows is None:
            rows = list(range(self._mlen))
        self._compiler.tile_row_sub_asm(dst_addr, src_addr, rows)

    def tile_row_mul(
        self,
        dst: VRAMMatrixVar,
        src: VRAMMatrixVar,
        rows: Optional[List[int]] = None,
    ):
        """
        Tile Row Mul: dst[i, :] *= src[i, :] for specified rows.
        """
        dst_addr = self._compiler.symbol_table[dst.name].vram_addr
        src_addr = self._compiler.symbol_table[src.name].vram_addr
        if rows is None:
            rows = list(range(self._mlen))
        self._compiler.tile_row_mul_asm(dst_addr, src_addr, rows)

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
        self._compiler.fpvar_reci_asm(src.address, dst.address, count)

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
        self._compiler.fpvar_max_asm(src1.address, src2.address, dst.address, count)

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
        self._compiler.fpvar_sub_asm(src1.address, src2.address, dst.address, count)

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
        self._compiler.fpvar_exp_asm(src.address, dst.address, count)

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
        self._compiler.fpvar_mul_asm(src1.address, src2.address, dst.address, count)

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
        self._compiler.fpvar_add_asm(src1.address, src2.address, dst.address, count)

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
        self._compiler.fpvar_copy_asm(src.address, dst.address, count)

    def tile_row_mul_fp_broadcast(
        self,
        source: VRAMMatrixVar,
        fpram_scalar_addr: int,
        row_idx: int,
    ):
        """
        Tile Row Mul FP Broadcast: multiply a single row by a FPRAM scalar.

        For row i: source[i, :] = source[i, :] * FPRAM[fpram_scalar_addr]

        Args:
            source: VRAM tile (mlen x mlen)
            fpram_scalar_addr: FPRAM address of the scalar
            row_idx: which row to process (0 to mlen-1)

        Example:
            scale_fp = prog.fp_var("scale", size=1)
            for row in range(64):
                prog.tile_row_mul_fp_broadcast(S, scale_fp.address, row)
        """
        vram_addr = self._compiler.symbol_table[source.name].vram_addr
        self._compiler.tile_row_mul_fp_broadcast_asm(vram_addr, fpram_scalar_addr, [row_idx])

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
        self._compiler.fpvar_fill_from_fpram_asm(dst.address, src_fpram_addr, count)

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
        vram_addr = self._compiler.symbol_table[matrix.name].vram_addr
        if rows is None:
            rows = list(range(matrix.shape[0]))
        self._compiler.vram_fill_zero_asm(vram_addr, rows)

    # ========================================================================
    # Sub-matrix Operations
    # ========================================================================

    def register_sub_matrix(self, input_var: InputVar,
                            name: Optional[str] = None) -> SubMatrixVar:
        """
        Register a large matrix for sub-block management

        Matrix is automatically divided into 64x64 sub-blocks, all addresses are pre-calculated at this time.

        Args:
            input_var: 输入变量
            name: 子矩阵名称（None = 使用输入名称 + "_sub"）

        Returns:
            SubMatrixVar 代理对象
        """
        if not isinstance(input_var, InputVar):
            raise TypeError(f"Expected InputVar, got {type(input_var)}")

        display_name = name if name is not None else f"{input_var.display_name}_sub"
        internal_name = self._scoped_name(display_name)

        h, w = input_var.shape

        self._compiler.register_sub_matrix(
            name=internal_name,
            hbm_addr=input_var.hbm_addr,
            h=h,
            w=w,
            real_data_ratio=self._real_data_ratio
        )

        var = SubMatrixVar(self, internal_name, input_var, input_var.shape,
                           display_name=display_name)
        self._sub_matrices[internal_name] = var
        self._tensors[internal_name] = var  # type: ignore
        return var

    def register_vram_sub_matrix(self, batch_var: BatchVar,
                                 name: Optional[str] = None) -> VRAMSubMatrixVar:
        """
        Register a Batch matrix in VRAM for sub-block management

        Args:
            batch_var: Batch 变量（必须已在 VRAM）
            name: 名称（None = 使用源名称 + "_vsub"）

        Returns:
            VRAMSubMatrixVar 代理对象
        """
        if not isinstance(batch_var, BatchVar):
            raise TypeError(f"Expected BatchVar, got {type(batch_var)}")

        display_name = name if name is not None else f"{batch_var.display_name}_vsub"
        internal_name = self._scoped_name(display_name)

        self._compiler.register_vram_sub_matrix(
            name=internal_name,
            source_tensor=batch_var.name  # 用 internal name 查 symbol table
        )

        var = VRAMSubMatrixVar(self, internal_name, batch_var.name, batch_var.shape,
                               display_name=display_name)
        self._vram_sub_matrices[internal_name] = var
        return var

    def reset_mram(self):
        """Reset MRAM allocator, freeing all loaded sub-blocks"""
        self._compiler.reset_mram()
        # Clear loaded records
        for sub in self._sub_matrices.values():
            sub.loaded_cols.clear()
            sub.loaded_rows.clear()

    # ========================================================================
    # 矩阵乘法已移除
    # 使用子矩阵投影 (sub_projection / sub_projection_T) 来实现所有矩阵乘法
    # ========================================================================

    # ========================================================================
    # 子块投影已移除，只使用 xxx_to 方法直接写入目标矩阵
    # ========================================================================

    # ========================================================================
    # VRAM 子块投影
    # ========================================================================

    def vram_sub_projection_to(
        self,
        vram_row: VRAMSubMatrixRowRef,
        mram_col: SubMatrixColRef,
        target: VRAMMatrixVar,
        target_row_idx: int,
        target_col_idx: int,
    ):
        """
        VRAM 子块乘法写入目标矩阵：
        target[row][col] = VRAM_A[i][:] @ MRAM_W[:][j]
        """
        self._compiler.vram_sub_projection_to(
            vram_mat_name=vram_row.vram_sub.name,
            vram_row_idx=vram_row.row_idx,
            mram_mat_name=mram_col.sub_matrix.name,
            mram_col_idx=mram_col.col_idx,
            target_matrix=target.name,
            target_row_idx=target_row_idx,
            target_col_idx=target_col_idx,
        )

    def vram_sub_projection_T_to(
        self,
        vram_row: VRAMSubMatrixRowRef,
        mram_row: SubMatrixRowRef,
        target: VRAMMatrixVar,
        target_row_idx: int,
        target_col_idx: int,
    ):
        """
        VRAM 子块转置乘法写入目标矩阵：
        target[row][col] = VRAM_A[i][:] @ MRAM_W[j][:]^T
        """
        self._compiler.vram_sub_projection_T_to(
            vram_mat_name=vram_row.vram_sub.name,
            vram_row_idx=vram_row.row_idx,
            mram_mat_name=mram_row.sub_matrix.name,
            mram_row_idx=mram_row.row_idx,
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

    # ========================================================================
    # Flash Attention Operations
    # ========================================================================

    def init_online_softmax(self, q_idx: int, o_matrix: VRAMMatrixVar):
        """Initialize Online Softmax state: m=-inf, l=0, O_row=0"""
        o_info = self._compiler.symbol_table[o_matrix.name]
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

    def compute_pv(self, s_block: VRAMMatrixVar, v_sub: SubMatrixVar,
                   k_idx: int, pv_matrix: VRAMMatrixVar):
        """Compute PV = P @ V[k_idx]"""
        pv_info = self._compiler.symbol_table[pv_matrix.name]
        head_dim = pv_info.shape[1]

        self._compiler.compute_pv(
            s_block_matrix=s_block.name,
            v_sub_matrix=v_sub.name,
            k_idx=k_idx,
            pv_matrix=pv_matrix.name,
            head_dim=head_dim,
        )

    def scale_o_row(self, o_matrix: VRAMMatrixVar, q_idx: int):
        """Scale current row block of O by m_res"""
        o_info = self._compiler.symbol_table[o_matrix.name]
        seq_len, head_dim = o_info.shape

        self._compiler.scale_o_row(
            o_matrix=o_matrix.name,
            q_idx=q_idx,
            seq_len=seq_len,
            head_dim=head_dim,
        )

    def final_scale_o(self, q_idx: int, o_matrix: VRAMMatrixVar):
        """Final scaling: O[q_idx] = O[q_idx] / l"""
        o_info = self._compiler.symbol_table[o_matrix.name]
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

            try:
                result = func(*args, **kwargs)

                # Auto-free: free locally allocated tensors that are not returned
                return_names = set()
                if isinstance(result, TensorVar):
                    return_names.add(result.internal_name)
                elif isinstance(result, (tuple, list)):
                    for r in result:
                        if isinstance(r, TensorVar):
                            return_names.add(r.internal_name)

                for name in set(self._tensors.keys()) - tensors_before:
                    if name not in return_names:
                        tensor = self._tensors[name]
                        if isinstance(tensor, (BatchVar, VRAMMatrixVar)):
                            self.free_tensor(tensor)
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
        return self._compiler.symbol_table

    # ========================================================================
    # 运算符分派（内部方法）
    # ========================================================================

    def _dispatch_matmul(self, left: TensorVar, right) -> TensorVar:
        """
        ⚠️ @ 运算符已不再支持
        
        请使用显式的 sub_projection_to 或 sub_projection_T_to 方法：
            # 错误：result = A @ W_sub.col(i)
            # 正确：
            Y = prog.alloc("Y", rows, cols)
            for row in range(num_row_blocks):
                for col in range(num_col_blocks):
                    A_sub.load_row(row)
                    W_sub.load_col(col)
                    prog.sub_projection_to(A_sub.row(row), W_sub.col(col), Y, row, col)
        """
        raise TypeError(
            f"@ operator is no longer supported. "
            f"Use explicit sub_projection_to() or sub_projection_T_to() methods:\n"
            f"  Y = prog.alloc('Y', rows, cols)\n"
            f"  prog.sub_projection_to(A_sub.row(r), W_sub.col(c), Y, r, c)"
        )

    def _dispatch_vram_sub_matmul(self, left: VRAMSubMatrixRowRef, right) -> Any:
        """
        分派 VRAM 子矩阵行的 @ 运算符

        支持的组合：
        - VRAMSubMatrixRowRef @ SubMatrixColRef  -> _DeferredSubProjection (projection)
        - VRAMSubMatrixRowRef @ TransposedSubMatrixRowRef -> _DeferredSubProjection (projection_T)

        返回 _DeferredSubProjection，可以被赋值到 VRAMMatrixVar 的子块位置。
        也可以直接作为独立的 vram_sub_projection 结果。
        """
        # VRAMSubRow @ SubMatrixCol -> deferred sub projection
        if isinstance(right, SubMatrixColRef):
            return _DeferredSubProjection(self, left, right, "projection")

        # VRAMSubRow @ SubMatrixRow.T -> deferred sub projection T
        if isinstance(right, TransposedSubMatrixRowRef):
            return _DeferredSubProjection(self, left, right.row_ref, "projection_T")

        raise TypeError(
            f"Unsupported VRAM sub matmul: VRAMSubRow @ {type(right).__name__}. "
            f"Supported: SubMatrixColRef, TransposedSubMatrixRowRef"
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


# ============================================================================
# 示例用法
# ============================================================================

if __name__ == "__main__":
    import math

    print("=" * 60)
    print("PLENAProgram Demo - SubMatrix Only")
    print("=" * 60)

    # ====== 示例 1: 子矩阵投影 ======
    print("\n--- Example 1: SubMatrix Projection ---")
    prog = PLENAProgram(mlen=64, blen=4)

    X = prog.input("X", shape=(64, 128))
    W = prog.input("W", shape=(128, 128))

    A = prog.load_batch(X)
    W_sub = prog.register_sub_matrix(W)

    # 加载列子块并投影
    W_sub.load_col(0)
    W_sub.load_col(1)
    C0 = A @ W_sub.col(0)   # sub_projection: A @ W[:][0]
    C1 = A @ W_sub.col(1)   # sub_projection: A @ W[:][1]

    prog.result(C0)
    print(f"C0 = {C0}")
    print(f"C1 = {C1}")
    print(f"ISA lines: {len(prog.compile().splitlines())}")
    prog.print_symbol_table()

    # ====== 示例 2: 子矩阵转置投影 ======
    print("\n--- Example 2: SubMatrix Transpose Projection ---")
    prog2 = PLENAProgram(mlen=64, blen=4)

    X2 = prog2.input("X", shape=(64, 128))
    W2 = prog2.input("W", shape=(64, 128))  # 注意：转置后是 (128, 64)

    A2 = prog2.load_batch(X2)
    W_sub2 = prog2.register_sub_matrix(W2)

    # 加载行子块并转置投影
    W_sub2.load_row(0)
    C_T = A2 @ W_sub2.row(0).T   # sub_projection_T: A @ W[0][:].T

    prog2.result(C_T)
    print(f"C_T = {C_T}")
    print(f"ISA lines: {len(prog2.compile().splitlines())}")
    prog2.print_symbol_table()

    # ====== 示例 3: 使用函数（带 local 矩阵）======
    print("\n--- Example 3: Functions with Local Matrices ---")
    prog3 = PLENAProgram(mlen=64, blen=4)

    @prog3.function
    def linear_with_submatrix(act, weight_sub, col_idx):
        """
        使用子矩阵的线性层
        
        每次调用内部的 local 变量都会自动加前缀：
        - 第 1 次调用: linear_with_submatrix_0/temp
        - 第 2 次调用: linear_with_submatrix_1/temp
        """
        temp = prog3.alloc("temp", act.shape[0], act.shape[1])  # local VRAM 矩阵
        result = act @ weight_sub.col(col_idx)
        return result

    X3 = prog3.input("X", shape=(64, 128))
    W3 = prog3.input("W", shape=(128, 128))

    A3 = prog3.load_batch(X3)
    W_sub3 = prog3.register_sub_matrix(W3)
    W_sub3.load_col(0)
    W_sub3.load_col(1)

    I = linear_with_submatrix(A3, W_sub3, 0)    # 第一次调用
    R = linear_with_submatrix(A3, W_sub3, 1)    # 第二次调用（不冲突）

    prog3.result(R)
    print(f"I = {I}")
    print(f"R = {R}")
    print(f"ISA lines: {len(prog3.compile().splitlines())}")
    print("\nSymbol Table (注意 local 矩阵的命名空间前缀):")
    prog3.print_symbol_table()

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)

