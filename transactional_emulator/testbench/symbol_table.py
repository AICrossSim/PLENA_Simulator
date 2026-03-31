"""
Symbol Table for PLENA Compiler
Used to record tensor metadata: name, type, shape, address, etc.
"""

from typing import Literal
from dataclasses import dataclass
from sub_matrix_manager import VirtualMemoryManager, MemoryBlock, MLEN


@dataclass
class TensorInfo:
    """Tensor Metadata"""

    kind: Literal["Batch", "Matrix"]  # Type: Activation Batch vs Weight Matrix
    dtype: str = "fp16"  # Data type: fp16 / mx / u8
    shape: tuple[int, int] = (0, 0)  # Shape: (h, w)
    hbm_addr: int = 0  # HBM base address (address already considers real_data_ratio)
    vram_addr: int | None = None  # VRAM base address (if any, otherwise None)
    size: int = 0  # Total logical element count (h * w), excluding scalar
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

    Supports virtual memory management:
    - Prioritizes reusing freed blocks during allocation
    - Moves address from used_stack to free_stack upon deallocation
    - Can invalidate and free space when a batch/vector is modified

    ⚠️ VRAM storage format (from CLAUDE.md):
    - Format: (batch_size, mlen, hidden_size / mlen)
    - Aligned to MLEN = 64
    """

    def __init__(self, alignment: int = MLEN, total_size: int = 0):
        """
        Args:
            alignment: Alignment size (PLENA's MLEN = 64)
            total_size: Total VRAM size (0 means unlimited, grows upwards)
        """
        self.alignment = alignment
        self._vmm = VirtualMemoryManager(total_size=total_size, alignment=alignment, mem_name="VRAM")

    @property
    def next_free(self) -> int:
        return self._vmm.next_bump

    @next_free.setter
    def next_free(self, value: int):
        self._vmm.next_bump = value

    @property
    def used_stack(self) -> list[MemoryBlock]:
        return self._vmm.used_stack

    @property
    def free_stack(self) -> list[MemoryBlock]:
        return self._vmm.free_stack

    def allocate(self, size: int, name: str = "") -> int:
        """
        Allocate VRAM address (prioritize reusing freed blocks)

        Args:
            size: Required size (number of elements)
            name: Allocation name (must be provided for subsequent free)

        Returns:
            Allocated VRAM base address (aligned to alignment)

        Raises:
            ValueError: name not provided
        """
        if not name:
            raise ValueError(
                "VRAMAllocator.allocate() must provide the name parameter, "
                "otherwise free is not possible. Please pass the tensor name."
            )
        return self._vmm.allocate(name, size)

    def free(self, name: str, strict: bool = True) -> MemoryBlock | None:
        """
        Free a VRAM allocation: move from used_stack to free_stack

        Args:
            name: Allocation name
            strict: If True, raises KeyError when not found; if False, returns None

        Returns:
            The freed memory block
        """
        return self._vmm.free(name, strict=strict)

    def is_allocated(self, name: str) -> bool:
        """Check whether a given name has been allocated"""
        return self._vmm.is_allocated(name)

    def get_next_free(self) -> int:
        """Get next available address (without allocating)"""
        return self._vmm.next_bump

    def reset(self):
        """Reset the allocator"""
        self._vmm.reset()

    def print_status(self):
        """Print memory status"""
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
        self.table: dict[str, TensorInfo] = {}
        self.vram_allocator = VRAMAllocator()

    def add_batch(
        self, name: str, hbm_addr: int, h: int, w: int, dtype: str = "fp16", real_data_ratio: float = 1.125
    ) -> TensorInfo:
        """
        Add Batch tensor (allocates VRAM, generates Load ISA)

        Batch can be loaded independently: from HBM → VRAM
        Generates preload_act_asm and other ISA instructions

        ⚠️ Important: HBM storage accounts for scalar (MXFP format)
        - Each 8 elements in HBM has 1 scalar (scale factor)
        - real_data_ratio = (8*8 + 8) / (8 * 8) = 1.125
        - Actual HBM size = size * real_data_ratio

        Args:
            name: Tensor name
            hbm_addr: HBM base address (address already considers real_data_ratio)
            h: Height (batch size)
            w: Width (hidden size)
            dtype: Data type
            real_data_ratio: HBM storage ratio (default 1.125, i.e. MXFP format)

        Returns:
            Created TensorInfo (includes automatically allocated vram_addr and calculated hbm_size)
        """
        size = h * w  # Total logical element count
        hbm_size = int(size * real_data_ratio)  # Actual HBM storage size (accounting for scalar)
        vram_addr = self.vram_allocator.allocate(size, name=name)

        info = TensorInfo(
            kind="Batch",
            dtype=dtype,
            shape=(h, w),
            hbm_addr=hbm_addr,
            vram_addr=vram_addr,
            size=size,
            hbm_size=hbm_size,
        )

        self.table[name] = info
        return info

    def add_matrix(
        self, name: str, hbm_addr: int, h: int, w: int, dtype: str = "fp16", real_data_ratio: float = 1.125
    ) -> TensorInfo:
        """
        Declare Matrix tensor (only records metadata, no ISA generation, no VRAM allocation)

        ⚠️ Important: Matrix cannot be loaded independently!
        - Matrix is only prefetched from HBM via H_PREFETCH_M during computation (e.g., projection)
        - This function is only used to register Matrix metadata in the symbol table
        - Does not generate any ISA instructions, does not allocate VRAM

        ⚠️ Important: HBM storage accounts for scalar (MXFP format)
        - Each 8 elements in HBM has 1 scalar (scale factor)
        - real_data_ratio = (8*8 + 8) / (8 * 8) = 1.125
        - Actual HBM size = size * real_data_ratio

        Args:
            name: Tensor name
            hbm_addr: HBM base address (address already considers real_data_ratio)
            h: Height (in_features)
            w: Width (out_features)
            dtype: Data type
            real_data_ratio: HBM storage ratio (default 1.125, i.e. MXFP format)

        Returns:
            Created TensorInfo (vram_addr is always None, includes calculated hbm_size)
        """
        size = h * w  # Total logical element count
        hbm_size = int(size * real_data_ratio)  # Actual HBM storage size (accounting for scalar)

        info = TensorInfo(
            kind="Matrix",
            dtype=dtype,
            shape=(h, w),
            hbm_addr=hbm_addr,
            vram_addr=None,  # Matrix is not in VRAM, only in HBM
            size=size,
            hbm_size=hbm_size,
        )

        self.table[name] = info
        return info

    def get(self, name: str, default: TensorInfo | None = None) -> TensorInfo | None:
        """
        Query tensor information (similar to dictionary's get method)

        Args:
            name: Tensor name
            default: Default value to return if not found (defaults to None)

        Returns:
            TensorInfo or default (if not found)
        """
        return self.table.get(name, default)

    def __getitem__(self, name: str) -> TensorInfo:
        """Supports table["A"] syntax"""
        if name not in self.table:
            raise KeyError(f"Tensor '{name}' not found in symbol table")
        return self.table[name]

    def __contains__(self, name: str) -> bool:
        """Supports 'A' in table syntax"""
        return name in self.table

    def __repr__(self) -> str:
        """Print the symbol table"""
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
            print(
                f"{name:<10} {info.kind:<8} {shape_str:<15} {info.hbm_addr:<10} {info.hbm_size:<10} {vram_str:<12} {info.size:<8}"
            )
        print("=" * 75)

    def validate_matrix_usage(self, name: str) -> bool:
        """
        Verify if Matrix usage is legal

        Matrix cannot be loaded independently, only used in computation operations
        This method is used to check if Matrix is used correctly

        Args:
            name: Tensor name

        Returns:
            True if it is a Matrix and usage is valid
        """
        if name not in self.table:
            raise KeyError(f"Tensor '{name}' not found in symbol table")

        info = self.table[name]
        if info.kind == "Matrix":
            # Matrix should be prefetched via H_PREFETCH_M during computation
            # This only checks that it is indeed a Matrix
            return True
        return False


# Example usage
if __name__ == "__main__":
    # Create symbol table
    table = SymbolTable()

    # Load_Batch: load Batch of shape (8, 128) from HBM addr=0
    table.add_batch("A", hbm_addr=0, h=8, w=128)

    # Declare Matrix: only registered in symbol table, no ISA generation, no VRAM allocation
    # Matrix will be prefetched from HBM via H_PREFETCH_M during projection
    table.add_matrix("B", hbm_addr=1024, h=128, w=256)

    # Load another Batch
    table.add_batch("C", hbm_addr=2048, h=8, w=256)

    # Print symbol table
    table.print_table()

    # Query
    print("\nQuery table['A']:")
    print(table["A"])

    print("\nQuery table['B']:")
    print(table["B"])

    # Check types
    print(f"\nA is Batch: {table['A'].kind == 'Batch'}")
    print(f"B is Matrix: {table['B'].kind == 'Matrix'}")
    print(f"A is in VRAM: {table['A'].vram_addr is not None}")
    print(f"B is in VRAM: {table['B'].vram_addr is not None}")
