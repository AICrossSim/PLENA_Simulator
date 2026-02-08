"""
Sub Matrix Manager for PLENA Compiler
Sub-matrix manager: divides large matrices into 64x64 sub-blocks, supports block loading and computation

Key Concepts:
1. Large matrices (e.g., 256x256) are divided into multiple 64x64 sub-blocks
2. Sub-block indexing: a[row_idx][col_idx] or a[row_idx][:] represents entire row sub-blocks
3. Addresses are pre-calculated during compiler phase and used directly at runtime
4. Format differences:
   - HBM: [batch, hidden_size] - row-major contiguous storage
   - RAM: [batch, mlen, hidden/mlen] - column-block major storage
"""

from typing import Dict, List, Optional, Tuple, Literal, Union
from dataclasses import dataclass, field
import math


# ==============================================================================
# Constant Definitions
# ==============================================================================

MLEN = 64  # Minimum matrix block size
BLEN = 4   # Vector tile size
IMM2_BOUND = 2**18


# ==============================================================================
# Virtual Memory Manager
# ==============================================================================

@dataclass
class MemoryBlock:
    """Memory Block Information"""
    name: str          # Allocation name (e.g., "W[0][1]" or "activation_A")
    addr: int          # Starting address
    size: int          # Block size (number of elements)

    def __repr__(self) -> str:
        return f"MemBlock({self.name}, addr={self.addr}, size={self.size})"


class VirtualMemoryManager:
    """
    Virtual Memory Manager

    Core Design:
    - used_stack: Allocated and in-use memory blocks
    - free_stack: Freed and reusable memory blocks

    Workflow:
    1. allocate(name, size): Allocate memory
       - Prioritize best-fit search for reusable blocks in free_stack
       - If not found, use bump allocation from the end
    2. free(name): Free memory
       - Move block from used_stack to free_stack
       - Address can be reused by subsequent allocate calls
       - Throws exception if not found when strict=True, returns None when strict=False

    ⚠️ Adheres to CLAUDE.md hardware parameters:
    - VRAM/MRAM 存储格式: (batch_size, mlen, hidden_size / mlen)
    - mlen = 64, blen = 4
    - Alignment requirements depend on storage hierarchy
    """

    def __init__(
        self,
        total_size: int,
        alignment: int = MLEN,
        mem_name: str = "Memory"
    ):
        """
        Args:
            total_size: Total memory size (number of elements)
            alignment: 对齐大小（VRAM 用 MLEN=64, MRAM 用 MLEN*MLEN=4096）
            mem_name: Memory name, for debugging information (e.g., "VRAM" or "MRAM")
        """
        self.total_size = total_size
        self.alignment = alignment
        self.mem_name = mem_name
        self.next_bump = 0  # Bump allocation pointer

        # Two core stacks
        self.used_stack: List[MemoryBlock] = []
        self.free_stack: List[MemoryBlock] = []

    def _align(self, value: int) -> int:
        """Align value to alignment"""
        return ((value + self.alignment - 1) // self.alignment) * self.alignment

    def allocate(self, name: str, size: int) -> int:
        """
        Allocate memory

        Strategy:
        1. First find best-fit in free_stack (reusable block with least waste)
        2. If no suitable block in free_stack, use bump allocation

        Args:
            name: 分配名称（用于追踪和释放）
            size: 需要的大小（元素个数）

        Returns:
            分配的Starting address

        Raises:
            MemoryError: Insufficient memory
        """
        aligned_size = self._align(size)

        # === 策略 1: 从 free_stack 找 best-fit ===
        best_idx = None
        best_waste = float('inf')

        for i, block in enumerate(self.free_stack):
            if block.size >= aligned_size:
                waste = block.size - aligned_size
                if waste < best_waste:
                    best_waste = waste
                    best_idx = i

        if best_idx is not None:
            # Taken from free_stack
            reused_block = self.free_stack.pop(best_idx)

            # If block is larger than needed, split remaining part and return to free_stack
            if reused_block.size > aligned_size:
                remaining = MemoryBlock(
                    name="<fragment>",
                    addr=reused_block.addr + aligned_size,
                    size=reused_block.size - aligned_size
                )
                self.free_stack.append(remaining)

            # Create new used block
            new_block = MemoryBlock(
                name=name,
                addr=reused_block.addr,
                size=aligned_size
            )
            self.used_stack.append(new_block)
            return new_block.addr

        # === Strategy 2: Bump allocation ===
        aligned_addr = self._align(self.next_bump)

        if self.total_size > 0 and aligned_addr + aligned_size > self.total_size:
            raise MemoryError(
                f"{self.mem_name} overflow: need {aligned_size} at addr {aligned_addr}, "
                f"total_size={self.total_size}, "
                f"used={len(self.used_stack)} blocks, "
                f"free={len(self.free_stack)} blocks"
            )

        new_block = MemoryBlock(
            name=name,
            addr=aligned_addr,
            size=aligned_size
        )
        self.used_stack.append(new_block)
        self.next_bump = aligned_addr + aligned_size
        return aligned_addr

    def free(self, name: str, strict: bool = True) -> Optional[MemoryBlock]:
        """
        Free memory: move block from used_stack to free_stack

        Args:
            name: Name of allocation to free
            strict: Throws KeyError if not found when strict=True, returns None when strict=False

        Returns:
            Freed memory block, returns None if strict=False and not found
        """
        for i, block in enumerate(self.used_stack):
            if block.name == name:
                freed = self.used_stack.pop(i)
                self.free_stack.append(freed)
                return freed

        if strict:
            raise KeyError(
                f"{self.mem_name}: allocation '{name}' not found in used_stack. "
                f"Current used: {[b.name for b in self.used_stack]}"
            )
        return None

    def is_allocated(self, name: str) -> bool:
        """Check if a name is in used_stack"""
        return any(b.name == name for b in self.used_stack)

    def get_block(self, name: str) -> Optional[MemoryBlock]:
        """Get memory block with specified name from used_stack"""
        for block in self.used_stack:
            if block.name == name:
                return block
        return None

    def get_used_size(self) -> int:
        """Get total used size"""
        return sum(b.size for b in self.used_stack)

    def get_free_size(self) -> int:
        """Get total reusable size"""
        return sum(b.size for b in self.free_stack)

    def reset(self):
        """Reset manager"""
        self.next_bump = 0
        self.used_stack.clear()
        self.free_stack.clear()

    def print_status(self):
        """Print memory status"""
        print(f"=== {self.mem_name} Virtual Memory Status ===")
        print(f"Total size: {self.total_size}")
        print(f"Bump pointer: {self.next_bump}")
        print(f"Used blocks ({len(self.used_stack)}):")
        for b in self.used_stack:
            print(f"  {b}")
        print(f"Free blocks ({len(self.free_stack)}):")
        for b in self.free_stack:
            print(f"  {b}")
        total_used = self.get_used_size()
        total_free = self.get_free_size()
        if self.total_size > 0:
            available = self.total_size - self.next_bump + total_free
            print(f"Summary: used={total_used}, free={total_free}, "
                  f"bump={self.next_bump}, available={available}/{self.total_size}")
        else:
            print(f"Summary: used={total_used}, free={total_free}, "
                  f"bump={self.next_bump} (unlimited mode)")

    def __repr__(self) -> str:
        return (
            f"VirtualMemoryManager({self.mem_name}, "
            f"used={len(self.used_stack)}, free={len(self.free_stack)}, "
            f"bump={self.next_bump}/{self.total_size})"
        )


# ==============================================================================
# Sub-matrix Information
# ==============================================================================

@dataclass
class SubMatrixInfo:
    """Metadata for sub-matrices"""
    parent_name: str        # Parent matrix name
    row_idx: int           # Sub-block row index
    col_idx: int           # Sub-block column index
    shape: Tuple[int, int] # 子块形状 (通常是 64x64)
    
    # Pre-calculated addresses (computed during compiler phase, used directly at runtime)
    hbm_offset: int = 0     # Offset in HBM (in elements)
    mram_addr: Optional[int] = None   # Address in MRAM (if loaded)
    
    def __repr__(self) -> str:
        mram_str = f"{self.mram_addr}" if self.mram_addr is not None else "None"
        return (
            f"SubMatrix({self.parent_name}[{self.row_idx}][{self.col_idx}], "
            f"shape={self.shape}, hbm_off={self.hbm_offset}, mram={mram_str})"
        )


@dataclass 
class MatrixBlockLayout:
    """
    Block layout information for large matrices
    
    Storage format differences:
    - HBM: [rows, cols] 行主序连续存储，每行 cols 个元素
    - MRAM: [batch, mlen, hidden/mlen] 列块优先存储
    """
    name: str
    full_shape: Tuple[int, int]  # 完整矩阵形状 (rows, cols)
    block_size: int = MLEN       # 子块大小（默认 64）
    
    # 分块信息
    num_row_blocks: int = 0
    num_col_blocks: int = 0
    
    # HBM Address Information
    hbm_base_addr: int = 0
    hbm_size: int = 0  # Size after considering real_data_ratio
    
    # 子块映射：(row_idx, col_idx) -> SubMatrixInfo
    sub_blocks: Dict[Tuple[int, int], SubMatrixInfo] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize block information"""
        rows, cols = self.full_shape
        self.num_row_blocks = math.ceil(rows / self.block_size)
        self.num_col_blocks = math.ceil(cols / self.block_size)
        
        # Create information for all sub-blocks (pre-calculate addresses)
        for r in range(self.num_row_blocks):
            for c in range(self.num_col_blocks):
                # HBM offset calculation (row-major)
                # 子块 (r, c) 的起始位置 = r * block_size * cols + c * block_size
                hbm_offset = r * self.block_size * cols + c * self.block_size
                
                sub_info = SubMatrixInfo(
                    parent_name=self.name,
                    row_idx=r,
                    col_idx=c,
                    shape=(self.block_size, self.block_size),
                    hbm_offset=hbm_offset,
                    mram_addr=None
                )
                self.sub_blocks[(r, c)] = sub_info
    
    def get_sub_block(self, row_idx: int, col_idx: int) -> SubMatrixInfo:
        """Get specified sub-block"""
        if (row_idx, col_idx) not in self.sub_blocks:
            raise IndexError(f"Sub block [{row_idx}][{col_idx}] out of range")
        return self.sub_blocks[(row_idx, col_idx)]
    
    def get_row_blocks(self, row_idx: int) -> List[SubMatrixInfo]:
        """Get all sub-blocks in a row"""
        return [self.sub_blocks[(row_idx, c)] for c in range(self.num_col_blocks)]
    
    def get_col_blocks(self, col_idx: int) -> List[SubMatrixInfo]:
        """Get all sub-blocks in a column"""
        return [self.sub_blocks[(r, col_idx)] for r in range(self.num_row_blocks)]


# ==============================================================================
# VRAM Sub-matrix Information
# ==============================================================================

@dataclass
class VRAMSubMatrixInfo:
    """VRAM 中Metadata for sub-matrices"""
    parent_name: str        # Parent matrix name
    row_idx: int           # Sub-block row index (沿 batch 方向)
    col_idx: int           # Sub-block column index (沿 hidden 方向)
    shape: Tuple[int, int] # 子块形状 (通常是 mlen x mlen)
    
    # Pre-calculated VRAM address
    vram_addr: int = 0      # VRAM 地址
    
    def __repr__(self) -> str:
        return (
            f"VRAMSubMatrix({self.parent_name}[{self.row_idx}][{self.col_idx}], "
            f"shape={self.shape}, vram={self.vram_addr})"
        )


@dataclass 
class VRAMMatrixBlockLayout:
    """
    VRAM 中Block layout information for large matrices
    
    VRAM 存储格式：[batch, mlen, hidden/mlen] - 列块优先存储
    - Batch dimension contiguous
    - Then mlen dimension
    - Finally hidden/mlen column blocks
    
    Blocking method:
    - Row direction (batch dimension): divided into batch/mlen sub-blocks
    - Column direction (hidden dimension): divided into hidden/mlen sub-blocks
    - Each sub-block is mlen x mlen
    """
    name: str
    full_shape: Tuple[int, int]  # 完整矩阵形状 (batch, hidden_size)
    vram_base_addr: int          # VRAM 基地址
    block_size: int = MLEN       # 子块大小（默认 64）
    
    # 分块信息
    num_row_blocks: int = 0      # Number of blocks in batch dimension
    num_col_blocks: int = 0      # Number of blocks in hidden dimension
    
    # 子块映射：(row_idx, col_idx) -> VRAMSubMatrixInfo
    sub_blocks: Dict[Tuple[int, int], VRAMSubMatrixInfo] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize block information"""
        batch, hidden = self.full_shape
        self.num_row_blocks = math.ceil(batch / self.block_size)
        self.num_col_blocks = math.ceil(hidden / self.block_size)
        
        # Create information for all sub-blocks (pre-calculate VRAM addresses)
        # VRAM 格式: [batch, mlen, hidden/mlen]
        # 子块 (r, c) 对应 batch[r*mlen:(r+1)*mlen], hidden[c*mlen:(c+1)*mlen]
        # 
        # VRAM 中列块 c 的Starting address: vram_base + c * batch * mlen
        # 子块 (r, c) 在列块 c 中的偏移: r * mlen * mlen
        for r in range(self.num_row_blocks):
            for c in range(self.num_col_blocks):
                # VRAM address calculation (column-block major)
                # 列块 c 的基地址 = vram_base + c * batch * mlen
                # 行子块 r 在该列块中的偏移 = r * mlen * mlen
                col_block_base = self.vram_base_addr + c * batch * self.block_size
                row_offset = r * self.block_size * self.block_size
                vram_addr = col_block_base + row_offset
                
                sub_info = VRAMSubMatrixInfo(
                    parent_name=self.name,
                    row_idx=r,
                    col_idx=c,
                    shape=(self.block_size, self.block_size),
                    vram_addr=vram_addr
                )
                self.sub_blocks[(r, c)] = sub_info
    
    def get_sub_block(self, row_idx: int, col_idx: int) -> VRAMSubMatrixInfo:
        """Get specified sub-block"""
        if (row_idx, col_idx) not in self.sub_blocks:
            raise IndexError(f"VRAM sub block [{row_idx}][{col_idx}] out of range")
        return self.sub_blocks[(row_idx, col_idx)]
    
    def get_row_blocks(self, row_idx: int) -> List[VRAMSubMatrixInfo]:
        """Get all sub-blocks in a row (即 A[row_idx][:])"""
        return [self.sub_blocks[(row_idx, c)] for c in range(self.num_col_blocks)]
    
    def get_col_blocks(self, col_idx: int) -> List[VRAMSubMatrixInfo]:
        """Get all sub-blocks in a column"""
        return [self.sub_blocks[(r, col_idx)] for r in range(self.num_row_blocks)]
    
    def get_row_vram_addrs(self, row_idx: int) -> List[int]:
        """Get list of VRAM addresses for all sub-blocks in a row"""
        return [block.vram_addr for block in self.get_row_blocks(row_idx)]


# ==============================================================================
# MRAM Allocator
# ==============================================================================

class MRAMAllocator:
    """
    Matrix RAM address allocator (based on VirtualMemoryManager)

    Supports virtual memory management:
    - Prioritize reusing freed blocks during allocation
    - Move addresses from used_stack to free_stack during deallocation
    - Can invalidate and free space when matrix blocks are modified/overwritten

    ⚠️ MRAM 存储格式 (来自 CLAUDE.md):
    - 存储子块大小为 mlen x mlen (64x64 = 4096 元素)
    - 对齐到 mlen * mlen
    """

    def __init__(self, total_size: int = MLEN * MLEN * 4):
        """
        Args:
            total_size: Total MRAM size (default 16384, can hold 4 64x64 matrix blocks)
        """
        self.total_size = total_size
        self._vmm = VirtualMemoryManager(
            total_size=total_size,
            alignment=MLEN * MLEN,  # 对齐到一个子块大小
            mem_name="MRAM"
        )

    @property
    def next_free(self) -> int:
        return self._vmm.next_bump

    @property
    def used_stack(self) -> List[MemoryBlock]:
        return self._vmm.used_stack

    @property
    def free_stack(self) -> List[MemoryBlock]:
        return self._vmm.free_stack

    def allocate(self, name: str, size: int) -> int:
        """
        Allocate MRAM space (prioritize reusing freed blocks)

        Args:
            name: 分配名称（用于追踪）
            size: 需要的大小

        Returns:
            分配的地址
        """
        return self._vmm.allocate(name, size)

    def free(self, name: str, strict: bool = True) -> Optional[MemoryBlock]:
        """
        Free specified allocation: move from used_stack to free_stack

        Args:
            name: 分配名称
            strict: Throws KeyError if not found when strict=True, returns None when strict=False

        Returns:
            释放的内存块
        """
        return self._vmm.free(name, strict=strict)

    def is_allocated(self, name: str) -> bool:
        """Check if a name is allocated"""
        return self._vmm.is_allocated(name)

    def reset(self):
        """Reset allocator"""
        self._vmm.reset()

    def print_status(self):
        """Print memory status"""
        self._vmm.print_status()


class FPRAMAllocator:
    """Floating Point RAM Allocator"""
    
    def __init__(self, total_size: int = 256):
        """
        Args:
            total_size: Total FP RAM size (default 256)
        """
        self.total_size = total_size
        self.next_free = 0
        self.allocations: Dict[str, Tuple[int, int]] = {}
    
    def allocate(self, name: str, size: int) -> int:
        """Allocate FP RAM space"""
        if self.next_free + size > self.total_size:
            raise MemoryError(f"FPRAM overflow: need {size}, have {self.total_size - self.next_free}")
        
        addr = self.next_free
        self.next_free += size
        self.allocations[name] = (addr, size)
        return addr
    
    def reset(self):
        """Reset allocator"""
        self.next_free = 0
        self.allocations.clear()


# ==============================================================================
# Sub Matrix Manager
# ==============================================================================

class SubMatrixManager:
    """
    子矩阵管理器
    
    Core Functions:
    1. Register large matrices as blocked matrices
    2. Support sub-block indexing: matrix[row_idx][col_idx] or matrix[row_idx][:]
    3. Pre-calculate all addresses (during compiler phase)
    4. Generate ISA code for load_sub_matrix and sub_projection
    
    Key Constraints:
    - 最小块大小是 64x64 (MLEN)
    - Matrix must be loaded into MRAM before participating in computation
    - HBM and RAM have different storage formats, requiring conversion during loading
    """
    
    def __init__(self, mlen: int = MLEN, blen: int = BLEN):
        self.mlen = mlen
        self.blen = blen
        
        # Matrix layout table: name -> MatrixBlockLayout
        self.matrices: Dict[str, MatrixBlockLayout] = {}
        
        # Memory Allocators
        self.mram_allocator = MRAMAllocator()
        self.fpram_allocator = FPRAMAllocator()
        
        # Currently loaded sub-blocks in MRAM
        self.loaded_sub_blocks: Dict[str, SubMatrixInfo] = {}
        
        # Pre-calculated address cache
        self._address_cache: Dict[str, int] = {}
    
    def register_matrix(
        self,
        name: str,
        shape: Tuple[int, int],
        hbm_base_addr: int,
        real_data_ratio: float = 1.125
    ) -> MatrixBlockLayout:
        """
        Register a large matrix, automatically blocking it
        
        Args:
            name: 矩阵名称
            shape: 完整形状 (rows, cols)
            hbm_base_addr: HBM 基地址
            real_data_ratio: HBM 存储比例（MXFP 格式）
            
        Returns:
            MatrixBlockLayout 对象
        """
        rows, cols = shape
        
        # Verify if shape is a multiple of mlen
        if rows % self.mlen != 0:
            raise ValueError(f"Matrix rows ({rows}) must be multiple of mlen ({self.mlen})")
        if cols % self.mlen != 0:
            raise ValueError(f"Matrix cols ({cols}) must be multiple of mlen ({self.mlen})")
        
        # Calculate HBM size
        size = rows * cols
        hbm_size = int(size * real_data_ratio)
        
        # Create block layout
        layout = MatrixBlockLayout(
            name=name,
            full_shape=shape,
            block_size=self.mlen,
            hbm_base_addr=hbm_base_addr,
            hbm_size=hbm_size
        )
        
        self.matrices[name] = layout
        return layout
    
    def get_sub_block(
        self,
        name: str,
        row_idx: int,
        col_idx: int
    ) -> SubMatrixInfo:
        """
        Get sub-block information
        
        Args:
            name: 矩阵名称
            row_idx: 行索引
            col_idx: 列索引
            
        Returns:
            SubMatrixInfo 对象（包含预计算的地址）
        """
        if name not in self.matrices:
            raise KeyError(f"Matrix '{name}' not registered")
        return self.matrices[name].get_sub_block(row_idx, col_idx)
    
    def get_row_blocks(self, name: str, row_idx: int) -> List[SubMatrixInfo]:
        """Get all sub-blocks in a row：matrix[row_idx][:]"""
        if name not in self.matrices:
            raise KeyError(f"Matrix '{name}' not registered")
        return self.matrices[name].get_row_blocks(row_idx)
    
    def get_col_blocks(self, name: str, col_idx: int) -> List[SubMatrixInfo]:
        """Get all sub-blocks in a column：matrix[:][col_idx]"""
        if name not in self.matrices:
            raise KeyError(f"Matrix '{name}' not registered")
        return self.matrices[name].get_col_blocks(col_idx)
    
    # ==========================================================================
    # VRAM 子矩阵管理
    # ==========================================================================
    
    def register_vram_matrix(
        self,
        name: str,
        shape: Tuple[int, int],
        vram_base_addr: int
    ) -> VRAMMatrixBlockLayout:
        """
        Register a large matrix in VRAM, automatically blocking it
        
        Args:
            name: 矩阵名称
            shape: 完整形状 (batch, hidden_size)
            vram_base_addr: VRAM 基地址
            
        Returns:
            VRAMMatrixBlockLayout 对象
        """
        batch, hidden = shape
        
        # Verify if shape is a multiple of mlen
        if batch % self.mlen != 0:
            raise ValueError(f"VRAM matrix batch ({batch}) must be multiple of mlen ({self.mlen})")
        if hidden % self.mlen != 0:
            raise ValueError(f"VRAM matrix hidden ({hidden}) must be multiple of mlen ({self.mlen})")
        
        # Create block layout
        layout = VRAMMatrixBlockLayout(
            name=name,
            full_shape=shape,
            vram_base_addr=vram_base_addr,
            block_size=self.mlen
        )
        
        # Store in vram_matrices dictionary
        if not hasattr(self, 'vram_matrices'):
            self.vram_matrices: Dict[str, VRAMMatrixBlockLayout] = {}
        self.vram_matrices[name] = layout
        return layout
    
    def get_vram_sub_block(
        self,
        name: str,
        row_idx: int,
        col_idx: int
    ) -> VRAMSubMatrixInfo:
        """Get VRAM sub-block information"""
        if not hasattr(self, 'vram_matrices') or name not in self.vram_matrices:
            raise KeyError(f"VRAM matrix '{name}' not registered")
        return self.vram_matrices[name].get_sub_block(row_idx, col_idx)
    
    def get_vram_row_blocks(self, name: str, row_idx: int) -> List[VRAMSubMatrixInfo]:
        """Get all sub-blocks in a row of VRAM matrix: matrix[row_idx][:]"""
        if not hasattr(self, 'vram_matrices') or name not in self.vram_matrices:
            raise KeyError(f"VRAM matrix '{name}' not registered")
        return self.vram_matrices[name].get_row_blocks(row_idx)
    
    def get_vram_col_blocks(self, name: str, col_idx: int) -> List[VRAMSubMatrixInfo]:
        """Get all sub-blocks in a column of VRAM matrix：matrix[:][col_idx]"""
        if not hasattr(self, 'vram_matrices') or name not in self.vram_matrices:
            raise KeyError(f"VRAM matrix '{name}' not registered")
        return self.vram_matrices[name].get_col_blocks(col_idx)
    
    # ==========================================================================
    # Address Calculation (Core! Pre-calculated during compiler phase)
    # ==========================================================================
    
    def compute_hbm_offset(
        self,
        name: str,
        row_idx: int,
        col_idx: int
    ) -> int:
        """
        计算子块Offset in HBM (in elements)
        
        HBM 存储格式：[rows, cols] 行主序
        子块 (r, c) 的起始 = r * block_size * full_cols + c * block_size
        
        Args:
            name: 矩阵名称
            row_idx: 行索引
            col_idx: 列索引
            
        Returns:
            HBM 偏移（元素单位，不是字节！）
        """
        layout = self.matrices[name]
        full_cols = layout.full_shape[1]
        
        # Pre-calculated address, directly retrieved from sub_blocks
        sub_block = layout.get_sub_block(row_idx, col_idx)
        return sub_block.hbm_offset
    
    def compute_absolute_hbm_addr(
        self,
        name: str,
        row_idx: int,
        col_idx: int
    ) -> int:
        """
        Calculate absolute HBM address of sub-block (in elements)
        
        Returns:
            Absolute HBM address = base + offset
        """
        layout = self.matrices[name]
        offset = self.compute_hbm_offset(name, row_idx, col_idx)
        return layout.hbm_base_addr + offset
    
    # ==========================================================================
    # ISA Generation: Load Sub Matrix
    # ==========================================================================
    
    def load_sub_matrix_asm(
        self,
        name: str,
        row_idx: int,
        col_idx: int,
        mram_dest_addr: int,
        hbm_addr_reg: int = 1,
        gp_regs: List[int] = None,
    ) -> str:
        """
        Generate ISA code for loading sub-matrix from HBM to MRAM
        
        ⚠️ Important: Format Conversion!
        - HBM: [rows, cols] 行主序
        - MRAM: Directly load mlen x mlen blocks using H_PREFETCH_M
        
        Args:
            name: 矩阵名称
            row_idx: Sub-block row index
            col_idx: Sub-block column index
            mram_dest_addr: MRAM 目标地址
            hbm_addr_reg: HBM 地址寄存器
            gp_regs: 可用的 GP 寄存器列表
            
        Returns:
            ISA 代码
        """
        if gp_regs is None:
            gp_regs = [1, 2, 3]
        
        layout = self.matrices[name]
        sub_block = layout.get_sub_block(row_idx, col_idx)
        
        # Pre-calculated HBM offset
        hbm_offset = sub_block.hbm_offset
        
        # Update sub_block's MRAM address
        sub_block.mram_addr = mram_dest_addr
        
        # 生成 ISA
        lines = []
        lines.append(f"; Load SubMatrix {name}[{row_idx}][{col_idx}] -> MRAM[{mram_dest_addr}]")
        lines.append(f"; HBM offset: {hbm_offset} (precomputed)")
        
        # Set SCALE and STRIDE
        # SCALE = full matrix size (for boundary checks)
        # STRIDE = number of columns in full matrix (row stride for row-major)
        full_size = layout.full_shape[0] * layout.full_shape[1]
        full_cols = layout.full_shape[1]
        
        gp_scale = gp_regs[0]
        gp_stride = gp_regs[1]
        gp_mram = gp_regs[2]
        
        lines.append(f"S_ADDI_INT gp{gp_scale}, gp0, {full_size}")
        lines.append(f"C_SET_SCALE_REG gp{gp_scale}")
        lines.append(f"S_ADDI_INT gp{gp_stride}, gp0, {full_cols}")
        lines.append(f"C_SET_STRIDE_REG gp{gp_stride}")
        
        # Set MRAM target address and HBM offset
        lines.append(f"S_ADDI_INT gp{gp_mram}, gp0, {mram_dest_addr}")
        lines.append(f"S_ADDI_INT gp{gp_scale}, gp0, {hbm_offset}")
        
        # H_PREFETCH_M: Load from HBM to MRAM (parameters consistent with projection_asm)
        lines.append(f"H_PREFETCH_M gp{gp_mram}, gp{gp_scale}, a{hbm_addr_reg}, 1, 0")
        
        # Record load status
        block_key = f"{name}[{row_idx}][{col_idx}]"
        self.loaded_sub_blocks[block_key] = sub_block
        
        return "\n".join(lines) + "\n"
    
    def load_row_sub_matrices_asm(
        self,
        name: str,
        row_idx: int,
        mram_start_addr: int,
        hbm_addr_reg: int = 1,
        gp_regs: List[int] = None,
    ) -> str:
        """
        Generate ISA code for loading all sub-blocks in a row: matrix[row_idx][:]
        
        Args:
            name: 矩阵名称
            row_idx: Sub-block row index
            mram_start_addr: MRAM Starting address
            hbm_addr_reg: HBM 地址寄存器
            gp_regs: 可用的 GP 寄存器列表
            
        Returns:
            ISA 代码
        """
        if gp_regs is None:
            gp_regs = [1, 2, 3]
        
        layout = self.matrices[name]
        num_col_blocks = layout.num_col_blocks
        
        lines = []
        lines.append(f"; Load SubMatrix Row {name}[{row_idx}][:] -> MRAM[{mram_start_addr}]")
        
        # Set SCALE and STRIDE（只需设置一次）
        full_size = layout.full_shape[0] * layout.full_shape[1]
        full_cols = layout.full_shape[1]
        
        gp_scale = gp_regs[0]
        gp_stride = gp_regs[1]
        gp_mram = gp_regs[2]
        
        lines.append(f"S_ADDI_INT gp{gp_scale}, gp0, {full_size}")
        lines.append(f"C_SET_SCALE_REG gp{gp_scale}")
        lines.append(f"S_ADDI_INT gp{gp_stride}, gp0, {full_cols}")
        lines.append(f"C_SET_STRIDE_REG gp{gp_stride}")
        
        # Load each sub-block
        mram_addr = mram_start_addr
        block_size = self.mlen * self.mlen
        
        for col_idx in range(num_col_blocks):
            sub_block = layout.get_sub_block(row_idx, col_idx)
            hbm_offset = sub_block.hbm_offset
            
            # 更新 MRAM 地址
            sub_block.mram_addr = mram_addr
            
            lines.append(f"; SubBlock [{row_idx}][{col_idx}]: HBM offset = {hbm_offset}")
            lines.append(f"S_ADDI_INT gp{gp_mram}, gp0, {mram_addr}")
            lines.append(f"S_ADDI_INT gp{gp_scale}, gp0, {hbm_offset}")
            lines.append(f"H_PREFETCH_M gp{gp_mram}, gp{gp_scale}, a{hbm_addr_reg}, 1, 0")
            
            # 记录
            block_key = f"{name}[{row_idx}][{col_idx}]"
            self.loaded_sub_blocks[block_key] = sub_block
            
            mram_addr += block_size
        
        return "\n".join(lines) + "\n"
    
    def load_col_sub_matrices_asm(
        self,
        name: str,
        col_idx: int,
        mram_start_addr: int,
        hbm_addr_reg: int = 1,
        gp_regs: List[int] = None,
    ) -> str:
        """
        Generate ISA code for loading all sub-blocks in a column: matrix[:][col_idx]
        
        用于 sub_projection: A @ W[:, col_idx*mlen:(col_idx+1)*mlen]
        
        Args:
            name: 矩阵名称
            col_idx: Sub-block column index
            mram_start_addr: MRAM Starting address
            hbm_addr_reg: HBM 地址寄存器
            gp_regs: 可用的 GP 寄存器列表
            
        Returns:
            ISA 代码
        """
        if gp_regs is None:
            gp_regs = [1, 2, 3]
        
        layout = self.matrices[name]
        num_row_blocks = layout.num_row_blocks
        
        lines = []
        lines.append(f"; Load SubMatrix Col {name}[:][{col_idx}] -> MRAM[{mram_start_addr}]")
        
        # Set SCALE and STRIDE（只需设置一次）
        full_size = layout.full_shape[0] * layout.full_shape[1]
        full_cols = layout.full_shape[1]
        
        gp_scale = gp_regs[0]
        gp_stride = gp_regs[1]
        gp_mram = gp_regs[2]
        
        lines.append(f"S_ADDI_INT gp{gp_scale}, gp0, {full_size}")
        lines.append(f"C_SET_SCALE_REG gp{gp_scale}")
        lines.append(f"S_ADDI_INT gp{gp_stride}, gp0, {full_cols}")
        lines.append(f"C_SET_STRIDE_REG gp{gp_stride}")
        
        # Load each sub-block（遍历所有行的第 col_idx 列）
        mram_addr = mram_start_addr
        block_size = self.mlen * self.mlen
        
        for row_idx in range(num_row_blocks):
            sub_block = layout.get_sub_block(row_idx, col_idx)
            hbm_offset = sub_block.hbm_offset
            
            # 更新 MRAM 地址
            sub_block.mram_addr = mram_addr
            
            lines.append(f"; SubBlock [{row_idx}][{col_idx}]: HBM offset = {hbm_offset}")
            lines.append(f"S_ADDI_INT gp{gp_mram}, gp0, {mram_addr}")
            lines.append(f"S_ADDI_INT gp{gp_scale}, gp0, {hbm_offset}")
            lines.append(f"H_PREFETCH_M gp{gp_mram}, gp{gp_scale}, a{hbm_addr_reg}, 1, 0")
            
            # 记录
            block_key = f"{name}[{row_idx}][{col_idx}]"
            self.loaded_sub_blocks[block_key] = sub_block
            
            mram_addr += block_size
        
        return "\n".join(lines) + "\n"
    
    # ==========================================================================
    # ISA Generation: Sub Projection
    # ==========================================================================
    
    def sub_projection_asm(
        self,
        act_name: str,
        mat_name: str,
        mat_col_idx: int,
        result_vram_addr: int,
        act_vram_addr: int,
        batch: int,
        gp_regs: List[int] = None,
    ) -> str:
        """
        Generate ISA code for sub-block projection
        
        计算：result = activation @ matrix[:][col_idx]
        即：result = A @ W[:, col_idx*mlen:(col_idx+1)*mlen]
        
        Where:
        - activation: (batch, hidden_size) 在 VRAM
          - VRAM 存储格式: [batch, mlen, hidden/mlen] 列块优先
        - matrix[:][col_idx]: 一列子块（所有行的第 col_idx 列），每块 (mlen, mlen) 已在 MRAM
          - W[:, col_idx*mlen:(col_idx+1)*mlen] 形状是 (hidden_size, mlen)
        - result: (batch, mlen) 在 VRAM
        
        M_MM 指令：(blen, mlen) @ (mlen, blen) -> (blen, blen)
        
        ⚠️ 参考 projection_asm 的循环结构：
        - 外层：遍历输出的每 blen 列（mlen // blen 个）
        - 中层：遍历 batch 的每 blen 块
        - 内层：沿 hidden_size 方向累加
        
        Args:
            act_name: activation 名称（用于追踪）
            mat_name: 矩阵名称
            mat_col_idx: 矩阵列索引（取 W 的第 col_idx 列子块）
            result_vram_addr: 结果 VRAM 地址
            act_vram_addr: activation VRAM 地址
            batch: batch size
            gp_regs: 可用的 GP 寄存器
            
        Returns:
            ISA 代码
        """
        if gp_regs is None:
            gp_regs = [1, 2, 3, 4, 5, 6]
        
        layout = self.matrices[mat_name]
        # 获取列子块：W 的所有行的第 col_idx 列
        col_blocks = layout.get_col_blocks(mat_col_idx)
        num_hidden_blocks = len(col_blocks)  # hidden_size // mlen
        
        # 验证所有子块已加载到 MRAM
        for sub_block in col_blocks:
            if sub_block.mram_addr is None:
                raise RuntimeError(
                    f"SubBlock {mat_name}[{sub_block.row_idx}][{mat_col_idx}] not loaded to MRAM"
                )
        
        lines = []
        lines.append(f"; Sub Projection: {act_name} @ {mat_name}[:][{mat_col_idx}] -> result")
        lines.append(f"; 即: A @ W[:, {mat_col_idx}*mlen:{mat_col_idx+1}*mlen]")
        lines.append(f"; activation: VRAM[{act_vram_addr}], batch={batch}")
        lines.append(f"; result: VRAM[{result_vram_addr}]")
        lines.append(f"; M_MM: (blen, mlen) @ (mlen, blen) -> (blen, blen)")
        
        # 寄存器分配
        gp_act = gp_regs[0]
        gp_mat = gp_regs[1]
        gp_result = gp_regs[2]
        gp_mat_temp = gp_regs[3]
        
        tiles_per_mlen = self.mlen // self.blen  # mlen 中有几个 blen 块
        
        # ========================================================================
        # 核心循环（参考 projection_asm）
        # 
        # 输出形状: (batch, mlen)
        # M_MM 输出: (blen, blen)
        # 
        # 外层：遍历输出的每 blen 列（共 mlen // blen 个）
        # 中层：遍历 batch 的每 blen 块（共 batch // blen 个）
        # 内层：沿 hidden_size 累加（共 hidden_size // mlen 个块）
        # ========================================================================
        for output_col in range(tiles_per_mlen):
            lines.append(f"; Output column block {output_col} (columns {output_col*self.blen}:{(output_col+1)*self.blen})")
            
            for batch_block in range(batch // self.blen):
                lines.append(f";   Batch block {batch_block}")
                
                # 沿 hidden_size 方向累加
                for hidden_block, sub_block in enumerate(col_blocks):
                    # ============================================================
                    # MRAM 中矩阵块的地址
                    # sub_block 是 (mlen, mlen) 的块
                    # 需要取其中第 output_col 个 blen 列
                    # 地址偏移 = output_col * blen
                    # ============================================================
                    mat_mram_addr = sub_block.mram_addr + output_col * self.blen
                    
                    # ============================================================
                    # VRAM 中 activation 的地址
                    # 存储格式: [batch, mlen, hidden/mlen] 列块优先
                    # A[batch_block*blen:(batch_block+1)*blen, hidden_block*mlen:(hidden_block+1)*mlen]
                    # 地址 = base + hidden_block * batch * mlen + batch_block * blen * mlen
                    # ============================================================
                    act_addr = act_vram_addr + hidden_block * batch * self.mlen + batch_block * self.blen * self.mlen
                    
                    lines.append(f"S_ADDI_INT gp{gp_act}, gp0, {act_addr}")
                    lines.append(f"S_ADDI_INT gp{gp_mat}, gp0, {mat_mram_addr}")
                    
                    # M_MM: (blen, mlen) @ (mlen, blen) -> (blen, blen)
                    lines.append(f"M_MM 0, gp{gp_mat}, gp{gp_act}")
                
                # 写出结果 (blen, blen)
                # 结果地址: result[batch_block*blen:(batch_block+1)*blen, output_col*blen:(output_col+1)*blen]
                # 地址 = base + batch_block * blen * mlen + output_col * blen
                result_addr = result_vram_addr + batch_block * self.blen * self.mlen + output_col * self.blen
                lines.append(f"S_ADDI_INT gp{gp_result}, gp0, {result_addr}")
                lines.append(f"M_MM_WO gp{gp_result}, gp0, 0")
        
        return "\n".join(lines) + "\n"
    
    def sub_projection_T_asm(
        self,
        act_name: str,
        mat_name: str,
        mat_row_idx: int,
        result_vram_addr: int,
        act_vram_addr: int,
        batch: int,
        gp_regs: List[int] = None,
    ) -> str:
        """
        生成子块转置投影的 ISA 代码
        
        计算：result = activation @ matrix[row_idx][:]^T
        即：result = A @ W[row_idx*mlen:(row_idx+1)*mlen, :]^T
        
        Where:
        - activation: (batch, hidden_size) 在 VRAM
          - VRAM 存储格式: [batch, mlen, hidden/mlen] 列块优先
        - matrix[row_idx][:]: 一行子块（第 row_idx 行的所有列），每块 (mlen, mlen) 已在 MRAM
          - W[row_idx*mlen:(row_idx+1)*mlen, :] 形状是 (mlen, hidden_size)
          - W[row_idx][:]^T 形状是 (hidden_size, mlen)
        - result: (batch, mlen) 在 VRAM
        
        M_TMM 指令：(blen, mlen) @ (mlen, blen)^T -> (blen, blen)
        注意：M_TMM 参数顺序是 M_TMM 0, act_addr, weight_addr
        
        ⚠️ 参考 tmm_matmul_asm (projection_T_asm) 的循环结构：
        - 外层：遍历输出的每 blen 列（mlen // blen 个）
        - 中层：遍历 batch 的每 blen 块
        - 内层：沿 hidden_size 方向累加
        
        Args:
            act_name: activation 名称
            mat_name: 矩阵名称
            mat_row_idx: 矩阵行索引（取 W 的第 row_idx 行子块）
            result_vram_addr: 结果 VRAM 地址
            act_vram_addr: activation VRAM 地址
            batch: batch size
            gp_regs: 可用的 GP 寄存器
            
        Returns:
            ISA 代码
        """
        if gp_regs is None:
            gp_regs = [1, 2, 3, 4, 5, 6]
        
        layout = self.matrices[mat_name]
        # 获取行子块：W 的第 row_idx 行的所有列
        row_blocks = layout.get_row_blocks(mat_row_idx)
        num_hidden_blocks = len(row_blocks)  # hidden_size // mlen
        
        # 验证所有子块已加载到 MRAM
        for sub_block in row_blocks:
            if sub_block.mram_addr is None:
                raise RuntimeError(
                    f"SubBlock {mat_name}[{mat_row_idx}][{sub_block.col_idx}] not loaded to MRAM"
                )
        
        lines = []
        lines.append(f"; Sub Projection T: {act_name} @ {mat_name}[{mat_row_idx}][:]^T -> result")
        lines.append(f"; 即: A @ W[{mat_row_idx}*mlen:{mat_row_idx+1}*mlen, :]^T")
        lines.append(f"; activation: VRAM[{act_vram_addr}], batch={batch}")
        lines.append(f"; result: VRAM[{result_vram_addr}]")
        lines.append(f"; M_TMM: (blen, mlen) @ (mlen, blen)^T -> (blen, blen)")
        
        # 寄存器分配
        gp_act = gp_regs[0]
        gp_mat = gp_regs[1]
        gp_result = gp_regs[2]
        gp_mat_temp = gp_regs[3]
        
        tiles_per_mlen = self.mlen // self.blen  # mlen 中有几个 blen 块
        
        # ========================================================================
        # 核心循环（参考 tmm_matmul_asm）
        # 
        # 输出形状: (batch, mlen)
        # M_TMM 输出: (blen, blen)
        # 
        # 外层：遍历输出的每 blen 列（共 mlen // blen 个）
        # 中层：遍历 batch 的每 blen 块（共 batch // blen 个）
        # 内层：沿 hidden_size 累加（共 hidden_size // mlen 个块）
        #
        # 关键区别于 M_MM：
        # - M_TMM 的 weight 偏移是 out_col * blen * mlen（行偏移，因为转置）
        # - M_TMM 参数顺序是 (0, act_addr, weight_addr)
        # ========================================================================
        for output_col in range(tiles_per_mlen):
            lines.append(f"; Output column block {output_col} (columns {output_col*self.blen}:{(output_col+1)*self.blen})")
            
            for batch_block in range(batch // self.blen):
                lines.append(f";   Batch block {batch_block}")
                
                # 沿 hidden_size 方向累加
                for hidden_block, sub_block in enumerate(row_blocks):
                    # ============================================================
                    # MRAM 中矩阵块的地址
                    # sub_block 是 (mlen, mlen) 的块
                    # 对于 M_TMM，需要取其中第 output_col 个 blen 行（转置后变成列）
                    # 偏移 = output_col * blen * mlen（与 tmm_matmul_asm 一致）
                    # ============================================================
                    mat_mram_addr = sub_block.mram_addr + output_col * self.blen * self.mlen
                    
                    # ============================================================
                    # VRAM 中 activation 的地址
                    # 存储格式: [batch, mlen, hidden/mlen] 列块优先
                    # A[batch_block*blen:(batch_block+1)*blen, hidden_block*mlen:(hidden_block+1)*mlen]
                    # 地址 = base + hidden_block * batch * mlen + batch_block * blen * mlen
                    # ============================================================
                    act_addr = act_vram_addr + hidden_block * batch * self.mlen + batch_block * self.blen * self.mlen
                    
                    lines.append(f"S_ADDI_INT gp{gp_act}, gp0, {act_addr}")
                    lines.append(f"S_ADDI_INT gp{gp_mat}, gp0, {mat_mram_addr}")
                    
                    # M_TMM: (blen, mlen) @ (mlen, blen)^T -> (blen, blen)
                    # 注意参数顺序：M_TMM 0, act_addr, weight_addr
                    lines.append(f"M_TMM 0, gp{gp_act}, gp{gp_mat}")
                
                # 写出结果 (blen, blen)
                # 结果地址: result[batch_block*blen:(batch_block+1)*blen, output_col*blen:(output_col+1)*blen]
                # 地址 = base + batch_block * blen * mlen + output_col * blen
                result_addr = result_vram_addr + batch_block * self.blen * self.mlen + output_col * self.blen
                lines.append(f"S_ADDI_INT gp{gp_result}, gp0, {result_addr}")
                lines.append(f"M_MM_WO gp{gp_result}, gp0, 0")
        
        return "\n".join(lines) + "\n"
    
    # ==========================================================================
    # ISA 生成：VRAM 子块 @ MRAM SubMatrix
    # ==========================================================================
    
    def vram_sub_projection_asm(
        self,
        vram_mat_name: str,
        vram_row_idx: int,
        mram_mat_name: str,
        mram_col_idx: int,
        result_vram_addr: int,
        gp_regs: List[int] = None,
    ) -> str:
        """
        生成 VRAM 子块与 MRAM SubMatrix 乘法的 ISA 代码
        
        计算：result = VRAM_A[row_idx][:] @ MRAM_W[:][col_idx]
        
        Where:
        - VRAM_A[row_idx][:]: 一行子块（mlen, hidden_size）
          - 实际是多个 (mlen, mlen) 子块，分布在不同列块
        - MRAM_W[:][col_idx]: 一列子块（hidden_size, mlen）
          - 已加载到 MRAM
        - result: (mlen, mlen) 在 VRAM
        
        M_MM 指令：(blen, mlen) @ (mlen, blen) -> (blen, blen)
        
        循环结构（参考 projection_asm）：
        - 外层：遍历输出的每 blen 列（共 mlen // blen 个）
        - 中层：遍历输出的每 blen 行（共 mlen // blen 个）
        - 内层：沿 hidden_size 方向累加
        
        Args:
            vram_mat_name: VRAM 矩阵名称
            vram_row_idx: VRAM 矩阵行索引
            mram_mat_name: MRAM 矩阵名称
            mram_col_idx: MRAM 矩阵列索引（取 W 的第 col_idx 列子块）
            result_vram_addr: 结果 VRAM 地址
            gp_regs: 可用的 GP 寄存器
            
        Returns:
            ISA 代码
        """
        if gp_regs is None:
            gp_regs = [1, 2, 3, 4, 5, 6]
        
        # 获取 VRAM 矩阵布局
        if not hasattr(self, 'vram_matrices') or vram_mat_name not in self.vram_matrices:
            raise KeyError(f"VRAM matrix '{vram_mat_name}' not registered")
        vram_layout = self.vram_matrices[vram_mat_name]
        
        # 获取 MRAM 矩阵布局
        mram_layout = self.matrices[mram_mat_name]
        
        # VRAM_A[row_idx][:]: 获取 VRAM 矩阵第 row_idx 行的所有子块
        vram_row_blocks = vram_layout.get_row_blocks(vram_row_idx)
        
        # MRAM_W[:][col_idx]: 获取 MRAM 矩阵第 col_idx 列的所有子块
        mram_col_blocks = mram_layout.get_col_blocks(mram_col_idx)
        
        # 验证维度匹配
        if len(vram_row_blocks) != len(mram_col_blocks):
            raise ValueError(
                f"Dimension mismatch: VRAM has {len(vram_row_blocks)} blocks, "
                f"MRAM has {len(mram_col_blocks)} blocks"
            )
        
        num_hidden_blocks = len(vram_row_blocks)
        
        # 验证 MRAM 子块已加载
        for sub_block in mram_col_blocks:
            if sub_block.mram_addr is None:
                raise RuntimeError(
                    f"SubBlock {mram_mat_name}[{sub_block.row_idx}][{mram_col_idx}] not loaded to MRAM"
                )
        
        lines = []
        lines.append(f"; VRAM Sub Projection: {vram_mat_name}[{vram_row_idx}][:] @ {mram_mat_name}[:][{mram_col_idx}]")
        lines.append(f"; VRAM A[row_idx][:]: ({self.mlen}, hidden) 分布在 {num_hidden_blocks} 个列块")
        lines.append(f"; MRAM W[:][col_idx]: (hidden, {self.mlen}) 共 {num_hidden_blocks} 个子块")
        lines.append(f"; Result: ({self.mlen}, {self.mlen}) at VRAM[{result_vram_addr}]")
        
        # 寄存器分配
        gp_act = gp_regs[0]
        gp_mat = gp_regs[1]
        gp_result = gp_regs[2]
        
        tiles_per_mlen = self.mlen // self.blen
        
        # ========================================================================
        # 核心循环
        # 
        # 输出形状: (mlen, mlen)
        # M_MM 输出: (blen, blen)
        # 
        # 外层：遍历输出的每 blen 列（共 mlen // blen 个）
        # 中层：遍历输出的每 blen 行（共 mlen // blen 个）
        # 内层：沿 hidden_size 累加
        #
        # VRAM 存储格式: [batch, mlen, hidden/mlen] 列块优先
        # 对于 A[row_idx][:]，它实际上是:
        #   A[row_idx*mlen:(row_idx+1)*mlen, :] = mlen 行，hidden_size 列
        # 在 VRAM 中，第 h 个列块的地址是:
        #   vram_base + h * full_batch * mlen + row_idx * mlen * mlen
        # 
        # 结果存储格式：假设也是 [mlen, mlen] 按列块优先
        # 但由于结果只有 mlen x mlen，可以简化存储
        # ========================================================================
        full_batch = vram_layout.full_shape[0]
        
        for output_col in range(tiles_per_mlen):
            lines.append(f"; Output column block {output_col}")
            
            for output_row in range(tiles_per_mlen):
                lines.append(f";   Output row block {output_row}")
                
                # 沿 hidden_size 方向累加
                for hidden_block in range(num_hidden_blocks):
                    vram_block = vram_row_blocks[hidden_block]
                    mram_block = mram_col_blocks[hidden_block]
                    
                    # VRAM activation 地址
                    # A[row_idx*mlen + output_row*blen : row_idx*mlen + (output_row+1)*blen, 
                    #   hidden_block*mlen : (hidden_block+1)*mlen]
                    # 地址 = vram_block.vram_addr + output_row * blen * mlen
                    act_addr = vram_block.vram_addr + output_row * self.blen * self.mlen
                    
                    # MRAM weight 地址
                    # W[hidden_block*mlen:(hidden_block+1)*mlen, col_idx*mlen + output_col*blen]
                    # 偏移 = output_col * blen（取第 output_col 组 blen 列）
                    mat_mram_addr = mram_block.mram_addr + output_col * self.blen
                    
                    lines.append(f"S_ADDI_INT gp{gp_act}, gp0, {act_addr}")
                    lines.append(f"S_ADDI_INT gp{gp_mat}, gp0, {mat_mram_addr}")
                    # M_MM 参数顺序: 0, mat_addr (MRAM), act_addr (VRAM)
                    lines.append(f"M_MM 0, gp{gp_mat}, gp{gp_act}")
                
                # 写出结果 (blen, blen)
                # 结果地址计算：假设结果按行主序存储
                # result[output_row*blen:(output_row+1)*blen, output_col*blen:(output_col+1)*blen]
                # 地址 = base + output_row * blen * mlen + output_col * blen
                result_addr = result_vram_addr + output_row * self.blen * self.mlen + output_col * self.blen
                lines.append(f"S_ADDI_INT gp{gp_result}, gp0, {result_addr}")
                lines.append(f"M_MM_WO gp{gp_result}, gp0, 0")
        
        return "\n".join(lines) + "\n"
    
    def vram_sub_projection_T_asm(
        self,
        vram_mat_name: str,
        vram_row_idx: int,
        mram_mat_name: str,
        mram_row_idx: int,
        result_vram_addr: int,
        gp_regs: List[int] = None,
    ) -> str:
        """
        生成 VRAM 子块与 MRAM SubMatrix 转置乘法的 ISA 代码
        
        计算：result = VRAM_A[row_idx][:] @ MRAM_W[row_idx][:]^T
        
        Where:
        - VRAM_A[row_idx][:]: 一行子块（mlen, hidden_size）
        - MRAM_W[row_idx][:]: 一行子块（mlen, hidden_size）转置后是（hidden_size, mlen）
        - result: (mlen, mlen) 在 VRAM
        
        M_TMM 指令：(blen, mlen) @ (mlen, blen)^T -> (blen, blen)
        
        Args:
            vram_mat_name: VRAM 矩阵名称
            vram_row_idx: VRAM 矩阵行索引
            mram_mat_name: MRAM 矩阵名称
            mram_row_idx: MRAM 矩阵行索引（取 W 的第 row_idx 行子块并转置）
            result_vram_addr: 结果 VRAM 地址
            gp_regs: 可用的 GP 寄存器
            
        Returns:
            ISA 代码
        """
        if gp_regs is None:
            gp_regs = [1, 2, 3, 4, 5, 6]
        
        # 获取 VRAM 矩阵布局
        if not hasattr(self, 'vram_matrices') or vram_mat_name not in self.vram_matrices:
            raise KeyError(f"VRAM matrix '{vram_mat_name}' not registered")
        vram_layout = self.vram_matrices[vram_mat_name]
        
        # 获取 MRAM 矩阵布局
        mram_layout = self.matrices[mram_mat_name]
        
        # VRAM_A[row_idx][:]: 获取 VRAM 矩阵第 row_idx 行的所有子块
        vram_row_blocks = vram_layout.get_row_blocks(vram_row_idx)
        
        # MRAM_W[row_idx][:]: 获取 MRAM 矩阵第 row_idx 行的所有子块
        mram_row_blocks = mram_layout.get_row_blocks(mram_row_idx)
        
        # 验证维度匹配
        if len(vram_row_blocks) != len(mram_row_blocks):
            raise ValueError(
                f"Dimension mismatch: VRAM has {len(vram_row_blocks)} blocks, "
                f"MRAM has {len(mram_row_blocks)} blocks"
            )
        
        num_hidden_blocks = len(vram_row_blocks)
        
        # 验证 MRAM 子块已加载
        for sub_block in mram_row_blocks:
            if sub_block.mram_addr is None:
                raise RuntimeError(
                    f"SubBlock {mram_mat_name}[{mram_row_idx}][{sub_block.col_idx}] not loaded to MRAM"
                )
        
        lines = []
        lines.append(f"; VRAM Sub Projection T: {vram_mat_name}[{vram_row_idx}][:] @ {mram_mat_name}[{mram_row_idx}][:]^T")
        lines.append(f"; VRAM A[row_idx][:]: ({self.mlen}, hidden)")
        lines.append(f"; MRAM W[row_idx][:]^T: (hidden, {self.mlen})")
        lines.append(f"; Result: ({self.mlen}, {self.mlen}) at VRAM[{result_vram_addr}]")
        
        # 寄存器分配
        gp_act = gp_regs[0]
        gp_mat = gp_regs[1]
        gp_result = gp_regs[2]
        
        tiles_per_mlen = self.mlen // self.blen
        full_batch = vram_layout.full_shape[0]
        
        # ========================================================================
        # 核心循环（参考 tmm_matmul_asm）
        # 
        # M_TMM: (blen, mlen) @ (mlen, blen)^T -> (blen, blen)
        # ========================================================================
        for output_col in range(tiles_per_mlen):
            lines.append(f"; Output column block {output_col}")
            
            for output_row in range(tiles_per_mlen):
                lines.append(f";   Output row block {output_row}")
                
                for hidden_block in range(num_hidden_blocks):
                    vram_block = vram_row_blocks[hidden_block]
                    mram_block = mram_row_blocks[hidden_block]
                    
                    # VRAM activation 地址
                    act_addr = vram_block.vram_addr + output_row * self.blen * self.mlen
                    
                    # MRAM weight 地址（转置，所以取行方向的偏移）
                    # 偏移 = output_col * blen * mlen
                    mat_mram_addr = mram_block.mram_addr + output_col * self.blen * self.mlen
                    
                    lines.append(f"S_ADDI_INT gp{gp_act}, gp0, {act_addr}")
                    lines.append(f"S_ADDI_INT gp{gp_mat}, gp0, {mat_mram_addr}")
                    lines.append(f"M_TMM 0, gp{gp_act}, gp{gp_mat}")
                
                # 写出结果
                result_addr = result_vram_addr + output_row * self.blen * self.mlen + output_col * self.blen
                lines.append(f"S_ADDI_INT gp{gp_result}, gp0, {result_addr}")
                lines.append(f"M_MM_WO gp{gp_result}, gp0, 0")
        
        return "\n".join(lines) + "\n"
    
    # ==========================================================================
    # 高级接口：完整的子块计算
    # ==========================================================================
    
    def compute_sub_matmul(
        self,
        a_name: str,
        a_row_idx: Union[int, slice],
        b_name: str,
        b_col_idx: Union[int, slice],
        result_name: str,
        transpose_b: bool = False,
    ) -> Tuple[str, int]:
        """
        计算子矩阵乘法并返回结果信息
        
        示例：
        - c = a[1][:] x b[:] -> a 的第 1 行子块 乘以 b 的所有行
        - c = a[1][:] x b[:]^T -> 转置版本
        
        Args:
            a_name: 矩阵 A 名称
            a_row_idx: A 的行索引（int 或 slice）
            b_name: 矩阵 B 名称
            b_col_idx: B 的列索引（int 或 slice）
            result_name: 结果名称（用于追踪）
            transpose_b: 是否转置 B
            
        Returns:
            (ISA 代码, 结果大小)
        """
        # TODO: 实现完整的子矩阵乘法编排
        pass
    
    # ==========================================================================
    # 格式转换：HBM <-> RAM
    # ==========================================================================
    
    def load_activation_with_format_convert_asm(
        self,
        name: str,
        hbm_base_addr: int,
        batch: int,
        hidden_size: int,
        vram_dest_addr: int,
        hbm_addr_reg: int = 0,
        gp_regs: List[int] = None,
    ) -> str:
        """
        从 HBM 加载 activation 到 VRAM，同时进行格式转换
        
        ⚠️ 格式差异：
        - HBM: [batch, hidden_size] 行主序连续存储
        - VRAM: [batch, mlen, hidden/mlen] 列块优先存储
        
        转换逻辑：
        HBM 中 element[b, h] 的地址 = hbm_base + b * hidden_size + h
        VRAM 中 element[b, h] 的地址 = vram_base + (h // mlen) * batch * mlen + b * mlen + (h % mlen)
        
        由于 H_PREFETCH_V 按 mlen 为单位加载，需要按列块分批加载
        
        Args:
            name: activation 名称
            hbm_base_addr: HBM 基地址
            batch: batch size
            hidden_size: hidden dimension
            vram_dest_addr: VRAM 目标地址
            hbm_addr_reg: HBM 地址寄存器
            gp_regs: 可用的 GP 寄存器
            
        Returns:
            ISA 代码
        """
        if gp_regs is None:
            gp_regs = [1, 2, 3, 4, 5]
        
        lines = []
        lines.append(f"; Load Activation with Format Convert: {name}")
        lines.append(f"; HBM[{hbm_base_addr}]: [batch={batch}, hidden={hidden_size}] row-major")
        lines.append(f"; VRAM[{vram_dest_addr}]: [batch, mlen, hidden/mlen] column-block major")
        
        # 寄存器分配
        gp_hbm_offset = gp_regs[0]
        gp_stride = gp_regs[1]
        gp_vram = gp_regs[2]
        gp_outer = gp_regs[3]
        gp_inner = gp_regs[4]
        
        num_col_blocks = hidden_size // self.mlen
        preload_len = 4  # 每次加载 4 行
        
        # 设置 SCALE（总大小）
        total_size = batch * hidden_size
        lines.append(f"S_ADDI_INT gp{gp_hbm_offset}, gp0, {total_size}")
        lines.append(f"C_SET_SCALE_REG gp{gp_hbm_offset}")
        
        # 设置 STRIDE（HBM 中每行的跨度）
        lines.append(f"S_ADDI_INT gp{gp_stride}, gp0, {hidden_size}")
        lines.append(f"C_SET_STRIDE_REG gp{gp_stride}")
        
        # 按列块加载
        for col_block in range(num_col_blocks):
            lines.append(f"; Column block {col_block}")
            
            # HBM 偏移：跳到第 col_block 个 mlen 列
            hbm_offset = col_block * self.mlen
            
            # VRAM 地址：列块优先
            vram_addr = vram_dest_addr + col_block * batch * self.mlen
            
            lines.append(f"S_ADDI_INT gp{gp_hbm_offset}, gp0, {hbm_offset}")
            lines.append(f"S_ADDI_INT gp{gp_vram}, gp0, {vram_addr}")
            
            # 按 batch 分批加载
            for batch_block in range(math.ceil(batch / preload_len)):
                actual_batch_offset = batch_block * preload_len * hidden_size
                actual_vram_offset = batch_block * preload_len * self.mlen
                
                lines.append(f"S_ADDI_INT gp{gp_hbm_offset}, gp0, {hbm_offset + actual_batch_offset}")
                lines.append(f"S_ADDI_INT gp{gp_vram}, gp0, {vram_addr + actual_vram_offset}")
                lines.append(f"H_PREFETCH_V gp{gp_vram}, gp{gp_hbm_offset}, a{hbm_addr_reg}, 1, 0")
        
        return "\n".join(lines) + "\n"
    
    def store_activation_with_format_convert_asm(
        self,
        name: str,
        vram_src_addr: int,
        batch: int,
        hidden_size: int,
        hbm_dest_addr: int,
        hbm_addr_reg: int = 0,
        gp_regs: List[int] = None,
    ) -> str:
        """
        从 VRAM 存储 activation 到 HBM，同时进行格式转换
        
        ⚠️ 格式差异：
        - VRAM: [batch, mlen, hidden/mlen] 列块优先存储
        - HBM: [batch, hidden_size] 行主序连续存储
        
        Args:
            name: activation 名称
            vram_src_addr: VRAM 源地址
            batch: batch size
            hidden_size: hidden dimension
            hbm_dest_addr: HBM 目标地址
            hbm_addr_reg: HBM 地址寄存器
            gp_regs: 可用的 GP 寄存器
            
        Returns:
            ISA 代码
        """
        if gp_regs is None:
            gp_regs = [1, 2, 3, 4, 5]
        
        lines = []
        lines.append(f"; Store Activation with Format Convert: {name}")
        lines.append(f"; VRAM[{vram_src_addr}]: [batch, mlen, hidden/mlen] column-block major")
        lines.append(f"; HBM[{hbm_dest_addr}]: [batch={batch}, hidden={hidden_size}] row-major")
        
        # 寄存器分配
        gp_hbm_offset = gp_regs[0]
        gp_stride = gp_regs[1]
        gp_vram = gp_regs[2]
        gp_outer = gp_regs[3]
        gp_inner = gp_regs[4]
        
        num_col_blocks = hidden_size // self.mlen
        store_amount = 4  # 每次存储 4 行
        
        # 设置 STRIDE（HBM 中每行的跨度）
        lines.append(f"S_ADDI_INT gp{gp_stride}, gp0, {hidden_size}")
        lines.append(f"C_SET_STRIDE_REG gp{gp_stride}")
        
        # 按列块存储
        for col_block in range(num_col_blocks):
            lines.append(f"; Column block {col_block}")
            
            # HBM 偏移：跳到第 col_block 个 mlen 列
            hbm_offset = col_block * self.mlen
            
            # VRAM 地址：列块优先
            vram_addr = vram_src_addr + col_block * batch * self.mlen
            
            # 按 batch 分批存储
            for batch_block in range(math.ceil(batch / store_amount)):
                actual_batch_offset = batch_block * store_amount * hidden_size
                actual_vram_offset = batch_block * store_amount * self.mlen
                
                lines.append(f"S_ADDI_INT gp{gp_hbm_offset}, gp0, {hbm_offset + actual_batch_offset}")
                lines.append(f"S_ADDI_INT gp{gp_vram}, gp0, {vram_addr + actual_vram_offset}")
                lines.append(f"H_STORE_V gp{gp_vram}, gp{gp_hbm_offset}, a{hbm_addr_reg}, 0")
        
        return "\n".join(lines) + "\n"
    
    # ==========================================================================
    # 预计算地址表生成
    # ==========================================================================
    
    def generate_address_table(self, name: str) -> Dict[str, int]:
        """
        生成矩阵的完整地址表（用于调试和验证）
        
        Args:
            name: 矩阵名称
            
        Returns:
            地址表字典
        """
        if name not in self.matrices:
            raise KeyError(f"Matrix '{name}' not registered")
        
        layout = self.matrices[name]
        addr_table = {}
        
        for (r, c), sub_block in layout.sub_blocks.items():
            key = f"{name}[{r}][{c}]"
            addr_table[f"{key}_hbm_offset"] = sub_block.hbm_offset
            addr_table[f"{key}_hbm_abs"] = layout.hbm_base_addr + sub_block.hbm_offset
            if sub_block.mram_addr is not None:
                addr_table[f"{key}_mram"] = sub_block.mram_addr
        
        return addr_table
    
    def print_address_table(self, name: str):
        """打印矩阵的地址表"""
        addr_table = self.generate_address_table(name)
        print(f"Address Table for {name}:")
        for key, addr in sorted(addr_table.items()):
            print(f"  {key}: {addr}")
    
    # ==========================================================================
    # 辅助方法
    # ==========================================================================
    
    def get_loaded_block_addr(self, name: str, row_idx: int, col_idx: int) -> int:
        """获取已加载子块的 MRAM 地址"""
        block_key = f"{name}[{row_idx}][{col_idx}]"
        if block_key not in self.loaded_sub_blocks:
            raise KeyError(f"SubBlock {block_key} not loaded")
        return self.loaded_sub_blocks[block_key].mram_addr
    
    def is_block_loaded(self, name: str, row_idx: int, col_idx: int) -> bool:
        """检查子块是否已加载到 MRAM"""
        block_key = f"{name}[{row_idx}][{col_idx}]"
        return block_key in self.loaded_sub_blocks
    
    def reset(self):
        """Reset manager状态"""
        self.mram_allocator.reset()
        self.fpram_allocator.reset()
        self.loaded_sub_blocks.clear()
        self._address_cache.clear()
    
    def print_layout(self, name: str):
        """打印矩阵的分块布局"""
        if name not in self.matrices:
            print(f"Matrix '{name}' not registered")
            return
        
        layout = self.matrices[name]
        print(f"Matrix: {name}")
        print(f"  Full shape: {layout.full_shape}")
        print(f"  Block size: {layout.block_size}")
        print(f"  Blocks: {layout.num_row_blocks} x {layout.num_col_blocks}")
        print(f"  HBM base: {layout.hbm_base_addr}")
        print(f"  Sub blocks:")
        for (r, c), sub in layout.sub_blocks.items():
            loaded = "LOADED" if sub.mram_addr is not None else ""
            print(f"    [{r}][{c}]: hbm_off={sub.hbm_offset}, mram={sub.mram_addr} {loaded}")


# ==============================================================================
# 示例用法
# ==============================================================================

if __name__ == "__main__":
    # 创建管理器
    manager = SubMatrixManager()
    
    # 注册一个 256x256 的矩阵
    manager.register_matrix("W", shape=(256, 256), hbm_base_addr=0)
    
    # 打印布局
    print("=" * 60)
    print("Matrix Layout")
    print("=" * 60)
    manager.print_layout("W")
    
    # Get sub-block information（地址已预计算）
    print("\n" + "=" * 60)
    print("Sub Block Info (Precomputed Addresses)")
    print("=" * 60)
    for r in range(4):
        for c in range(4):
            sub = manager.get_sub_block("W", r, c)
            print(f"  W[{r}][{c}]: hbm_offset = {sub.hbm_offset}")
    
    # 生成加载子块的 ISA
    print("\n" + "=" * 60)
    print("Load Sub Matrix ISA")
    print("=" * 60)
    isa = manager.load_sub_matrix_asm("W", row_idx=1, col_idx=0, mram_dest_addr=0)
    print(isa)
    
    # 生成加载一行子块的 ISA
    print("\n" + "=" * 60)
    print("Load Row Sub Matrices ISA")
    print("=" * 60)
    isa = manager.load_row_sub_matrices_asm("W", row_idx=1, mram_start_addr=0)
    print(isa)
    
    # 打印更新后的布局（显示加载状态）
    print("\n" + "=" * 60)
    print("Updated Layout (After Loading)")
    print("=" * 60)
    manager.print_layout("W")

