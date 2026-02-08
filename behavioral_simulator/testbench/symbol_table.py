"""
Symbol Table for PLENA Compiler
Used to record tensor metadata: name, type, shape, address, etc.
"""

from typing import Dict, Optional, Tuple, Literal, List
from dataclasses import dataclass
from sub_matrix_manager import VirtualMemoryManager, MemoryBlock, MLEN


@dataclass
class TensorInfo:
    """Tensor Metadata"""
    kind: Literal["Batch", "Matrix"]  # Type: Activation Batch vs Weight Matrix
    dtype: str = "fp16"  # Data type: fp16 / mx / u8
    shape: Tuple[int, int] = (0, 0)  # 形状：(h, w)
    hbm_addr: int = 0  # HBM base address (address already considers real_data_ratio)
    vram_addr: Optional[int] = None  # VRAM base address (if any, otherwise None)
    size: int = 0  # 逻辑元素总数（h * w），不考虑 scalar
    hbm_size: int = 0  # Actual HBM storage size (considering real_data_ratio for scalar storage)

    def __repr__(self) -> str:
        vram_str = f"{self.vram_addr}" if self.vram_addr is not None else "None"
        return (
            f"TensorInfo(kind={self.kind}, dtype={self.dtype}, "
            f"shape={self.shape}, hbm_addr={self.hbm_addr}, "
            f"vram_addr={vram_str}, size={self.size}, hbm_size={self.hbm_size})"
        )


class VRAMAllocator:
    """
    VRAM Address Allocator (based on VirtualMemoryManager)

    支持虚拟内存管理：
    - 分配时优先重用已释放的块
    - 释放时将地址从 used_stack 移到 free_stack
    - 当 batch/向量被修改时可 invalidate 释放空间

    ⚠️ VRAM 存储格式 (来自 CLAUDE.md):
    - 格式: (batch_size, mlen, hidden_size / mlen)
    - 对齐到 MLEN = 64
    """

    def __init__(self, alignment: int = MLEN, total_size: int = 0):
        """
        Args:
            alignment: Alignment size (PLENA's MLEN = 64)
            total_size: Total VRAM size (0 means unlimited, grows upwards)
        """
        self.alignment = alignment
        self._vmm = VirtualMemoryManager(
            total_size=total_size,
            alignment=alignment,
            mem_name="VRAM"
        )

    @property
    def next_free(self) -> int:
        return self._vmm.next_bump

    @next_free.setter
    def next_free(self, value: int):
        self._vmm.next_bump = value

    @property
    def used_stack(self) -> List[MemoryBlock]:
        return self._vmm.used_stack

    @property
    def free_stack(self) -> List[MemoryBlock]:
        return self._vmm.free_stack

    def allocate(self, size: int, name: str = "") -> int:
        """
        Allocate VRAM address (prioritize reusing freed blocks)

        Args:
            size: 需要的大小（元素个数）
            name: 分配名称（必须提供，用于后续 free）

        Returns:
            分配的 VRAM base address（对齐到 alignment）

        Raises:
            ValueError: 未提供 name
        """
        if not name:
            raise ValueError(
                "VRAMAllocator.allocate() 必须提供 name 参数，"
                "否则无法 free。请传入 tensor 名称。"
            )
        return self._vmm.allocate(name, size)

    def free(self, name: str, strict: bool = True) -> Optional[MemoryBlock]:
        """
        释放 VRAM 分配：从 used_stack 移到 free_stack

        Args:
            name: 分配名称
            strict: True 时找不到会抛 KeyError，False 时返回 None

        Returns:
            释放的内存块
        """
        return self._vmm.free(name, strict=strict)

    def is_allocated(self, name: str) -> bool:
        """检查某个名称是否已分配"""
        return self._vmm.is_allocated(name)

    def get_next_free(self) -> int:
        """Get next available address (without allocating)"""
        return self._vmm.next_bump

    def reset(self):
        """重置分配器"""
        self._vmm.reset()

    def print_status(self):
        """打印内存状态"""
        self._vmm.print_status()

    def __repr__(self) -> str:
        return (
            f"VRAMAllocator(next_free={self.next_free}, alignment={self.alignment}, "
            f"used={len(self.used_stack)}, free={len(self.free_stack)})"
        )


class SymbolTable:
    """Symbol Table: Records metadata for all tensors"""
    
    def __init__(self):
        """Initialize an empty symbol table"""
        self.table: Dict[str, TensorInfo] = {}
        self.vram_allocator = VRAMAllocator()
    
    def add_batch(
        self,
        name: str,
        hbm_addr: int,
        h: int,
        w: int,
        dtype: str = "fp16",
        real_data_ratio: float = 1.125
    ) -> TensorInfo:
        """
        Add Batch tensor (allocates VRAM, generates Load ISA)
        
        Batch can be loaded independently: from HBM → VRAM
        Generates preload_act_asm and other ISA instructions
        
        ⚠️ 重要：HBM 存储考虑 scalar（MXFP 格式）
        - HBM 中每 8 个元素有 1 个 scalar (scale factor)
        - real_data_ratio = (8*8 + 8) / (8 * 8) = 1.125
        - HBM 实际大小 = size * real_data_ratio
        
        Args:
            name: tensor 名字
            hbm_addr: HBM base address (address already considers real_data_ratio)
            h: 高度（batch size）
            w: 宽度（hidden size）
            dtype: 数据类型
            real_data_ratio: HBM 存储比例（默认 1.125，即 MXFP 格式）
            
        Returns:
            Created TensorInfo (includes automatically allocated vram_addr and calculated hbm_size)
        """
        size = h * w  # 逻辑元素总数
        hbm_size = int(size * real_data_ratio)  # HBM 实际存储大小（考虑 scalar）
        vram_addr = self.vram_allocator.allocate(size, name=name)
        
        info = TensorInfo(
            kind="Batch",
            dtype=dtype,
            shape=(h, w),
            hbm_addr=hbm_addr,
            vram_addr=vram_addr,
            size=size,
            hbm_size=hbm_size
        )
        
        self.table[name] = info
        return info
    
    def add_matrix(
        self,
        name: str,
        hbm_addr: int,
        h: int,
        w: int,
        dtype: str = "fp16",
        real_data_ratio: float = 1.125
    ) -> TensorInfo:
        """
        Declare Matrix tensor (only records metadata, no ISA generation, no VRAM allocation)
        
        ⚠️ 重要：Matrix cannot be loaded independently!
        - Matrix is only prefetched from HBM via H_PREFETCH_M during computation (e.g., projection)
        - This function is only used to register Matrix metadata in the symbol table
        - Does not generate any ISA instructions, does not allocate VRAM
        
        ⚠️ 重要：HBM 存储考虑 scalar（MXFP 格式）
        - HBM 中每 8 个元素有 1 个 scalar (scale factor)
        - real_data_ratio = (8*8 + 8) / (8 * 8) = 1.125
        - HBM 实际大小 = size * real_data_ratio
        
        Args:
            name: tensor 名字
            hbm_addr: HBM base address (address already considers real_data_ratio)
            h: 高度（in_features）
            w: 宽度（out_features）
            dtype: 数据类型
            real_data_ratio: HBM 存储比例（默认 1.125，即 MXFP 格式）
            
        Returns:
            Created TensorInfo (vram_addr is always None, includes calculated hbm_size)
        """
        size = h * w  # 逻辑元素总数
        hbm_size = int(size * real_data_ratio)  # HBM 实际存储大小（考虑 scalar）
        
        info = TensorInfo(
            kind="Matrix",
            dtype=dtype,
            shape=(h, w),
            hbm_addr=hbm_addr,
            vram_addr=None,  # Matrix 不在 VRAM，只在 HBM
            size=size,
            hbm_size=hbm_size
        )
        
        self.table[name] = info
        return info
    
    def get(self, name: str, default: Optional[TensorInfo] = None) -> Optional[TensorInfo]:
        """
        Query tensor information (similar to dictionary's get method)
        
        Args:
            name: tensor 名字
            default: Default value to return if not found (defaults to None)
            
        Returns:
            TensorInfo 或 default（如果不存在）
        """
        return self.table.get(name, default)
    
    def __getitem__(self, name: str) -> TensorInfo:
        """支持 table["A"] 语法"""
        if name not in self.table:
            raise KeyError(f"Tensor '{name}' not found in symbol table")
        return self.table[name]
    
    def __contains__(self, name: str) -> bool:
        """Supports 'A' in table syntax"""
        return name in self.table
    
    def __repr__(self) -> str:
        """打印符号表"""
        lines = ["SymbolTable:"]
        lines.append(f"  VRAM Allocator: {self.vram_allocator}")
        lines.append("  Tensors:")
        for name, info in self.table.items():
            lines.append(f"    {name}: {info}")
        return "\n".join(lines)
    
    def print_table(self):
        """Print symbol table (formatted output)"""
        print("=" * 60)
        print("Symbol Table")
        print("=" * 60)
        print(f"VRAM Allocator: next_free={self.vram_allocator.next_free}")
        print()
        print(f"{'Name':<10} {'Kind':<8} {'Shape':<15} {'HBM Addr':<10} {'HBM Size':<10} {'VRAM Addr':<12} {'Size':<8}")
        print("-" * 75)
        for name, info in self.table.items():
            vram_str = f"{info.vram_addr}" if info.vram_addr is not None else "None (HBM only)"
            shape_str = f"({info.shape[0]}, {info.shape[1]})"
            print(f"{name:<10} {info.kind:<8} {shape_str:<15} {info.hbm_addr:<10} {info.hbm_size:<10} {vram_str:<12} {info.size:<8}")
        print("=" * 75)
    
    def validate_matrix_usage(self, name: str) -> bool:
        """
        Verify if Matrix usage is legal
        
        Matrix cannot be loaded independently, only used in computation operations
        This method is used to check if Matrix is used correctly
        
        Args:
            name: tensor 名字
            
        Returns:
            True 如果是 Matrix 且使用合法
        """
        if name not in self.table:
            raise KeyError(f"Tensor '{name}' not found in symbol table")
        
        info = self.table[name]
        if info.kind == "Matrix":
            # Matrix should be prefetched via H_PREFETCH_M during computation
            # This only checks that it is indeed a Matrix
            return True
        return False


# 示例用法
if __name__ == "__main__":
    # 创建符号表
    table = SymbolTable()
    
    # Load_Batch: 从 HBM addr=0 加载 (8, 128) 的 Batch
    table.add_batch("A", hbm_addr=0, h=8, w=128)
    
    # Declare Matrix: only registered in symbol table, no ISA generation, no VRAM allocation
    # Matrix will be prefetched from HBM via H_PREFETCH_M during projection
    table.add_matrix("B", hbm_addr=1024, h=128, w=256)
    
    # Load another Batch
    table.add_batch("C", hbm_addr=2048, h=8, w=256)
    
    # 打印符号表
    table.print_table()
    
    # 查询
    print("\n查询 table['A']:")
    print(table["A"])
    
    print("\n查询 table['B']:")
    print(table["B"])
    
    # 检查类型
    print(f"\nA 是 Batch: {table['A'].kind == 'Batch'}")
    print(f"B 是 Matrix: {table['B'].kind == 'Matrix'}")
    print(f"A 在 VRAM: {table['A'].vram_addr is not None}")
    print(f"B 在 VRAM: {table['B'].vram_addr is not None}")
