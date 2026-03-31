"""
Developer Compiler for PLENA
Implements compilation from high-level IR to ISA, Phase 1: Load_Batch
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from typing import List, Optional, Callable, Dict, Any, Tuple, Union
from sub_matrix_manager import SubMatrixManager, MatrixBlockLayout, SubMatrixInfo, VRAMMatrixBlockLayout, MLEN, BLEN
from compiler.asm_templates import (
    preload_act_asm,
    reset_reg_asm,
    preload_addr_reg_asm,
    projection_asm,
    store_act_asm,
    rms_norm_asm,
    layer_norm_asm,
    rope_asm,
)

class RegisterAllocator:
    """Register Allocator: Manages address registers and GP registers"""
    
    def __init__(self, start_gp: int = 1, start_addr: int = 0, start_fp: int = 1):
        """
        Args:
            start_gp: GP 寄存器起始编号（gp0 通常保留为常量 0）
            start_addr: Address 寄存器起始编号
            start_fp: FP 寄存器起始编号（f0 保留为常量 0，写入 f0 是 no-op）
        """
        # 硬件 OPERAND_WIDTH = 4 位，所以只有 gp0-gp15 可用（16个寄存器）
        # gp0 通常保留为常量 0，所以可用 gp1-gp15（15个）
        self.gp_registers = list(range(start_gp, 16))  # gp1-gp15 可用
        self.addr_registers = list(range(start_addr, 8))  # a0-a7 可用
        # f0 保留为常量 0（写入 f0 对 V_RED_MAX/V_RED_SUM 是 no-op）
        # 总共 8 个 FP 寄存器，其中可用的是 f1-f7（默认 7 个）
        self.fp_registers = list(range(start_fp, 8))  # f1-f7 可用
        self.used_gp = []
        self.used_addr = []
        self.used_fp = []
    
    def allocate_gp(self, count: int = 1) -> List[int]:
        """
        Allocate GP registers
        
        Args:
            count: 需要分配的寄存器数量
            
        Returns:
            分配的寄存器编号列表
        """
        if len(self.gp_registers) < count:
            raise RuntimeError(f"Not enough GP registers available. Need {count}, have {len(self.gp_registers)}")
        
        allocated = self.gp_registers[:count]
        self.gp_registers = self.gp_registers[count:]
        self.used_gp.extend(allocated)
        return allocated
    
    def allocate_addr(self, count: int = 1) -> List[int]:
        """
        Allocate Address registers
        
        Args:
            count: 需要分配的寄存器数量
            
        Returns:
            分配的寄存器编号列表
        """
        if len(self.addr_registers) < count:
            raise RuntimeError(f"Not enough address registers available. Need {count}, have {len(self.addr_registers)}")
        
        allocated = self.addr_registers[:count]
        self.addr_registers = self.addr_registers[count:]
        self.used_addr.extend(allocated)
        return allocated
    
    def free_gp(self, registers: List[int]):
        """Free GP registers"""
        for reg in registers:
            if reg in self.used_gp:
                self.used_gp.remove(reg)
                self.gp_registers.append(reg)
        self.gp_registers.sort()
    
    def free_addr(self, registers: List[int]):
        """Free Address registers"""
        for reg in registers:
            if reg in self.used_addr:
                self.used_addr.remove(reg)
                self.addr_registers.append(reg)
        self.addr_registers.sort()

    def allocate_fp(self, count: int = 1) -> List[int]:
        """
        Allocate FP registers (f0-f7)

        Args:
            count: 需要分配的寄存器数量

        Returns:
            分配的寄存器编号列表
        """
        if len(self.fp_registers) < count:
            raise RuntimeError(f"Not enough FP registers available. Need {count}, have {len(self.fp_registers)}")

        # 倒序分配：优先使用高编号寄存器，避免与历史前向硬编码冲突
        allocated = list(reversed(self.fp_registers[-count:]))
        self.fp_registers = self.fp_registers[:-count]
        self.used_fp.extend(allocated)
        return allocated

    def free_fp(self, registers: List[int]):
        """Free FP registers"""
        for reg in registers:
            if reg in self.used_fp:
                self.used_fp.remove(reg)
                self.fp_registers.append(reg)
        # 保持列表有序，allocate_fp 通过尾部切片继续实现倒序分配
        self.fp_registers.sort()


class DeveloperCompiler:
    """
    Developer Compiler: Compiles high-level IR to ISA
    
    DeveloperCompiler 是权限最大的管理器，包含：
    - symbol_table: 符号表
    - register_allocator: 寄存器分配器
    - interrupt: InterruptManager 内部类实例
    """
    
    _ONLINE_SOFTMAX_FPSRAM_BASE = 10

    # === Inner Class: InterruptManager ===
    class InterruptManager:
        """
        Interrupt Manager - Only manages execution timing
        
        具体逻辑写在 DeveloperCompiler 的方法里：
        - _handle_k_start()
        - _handle_k_prefetch_done()
        - _handle_s_tile_done()
        - _handle_k_end()
        
        使用方法：
        ```python
        compiler = DeveloperCompiler()
        compiler.interrupt.setup(batch=64, out_features=64, hidden_size=64)
        compiler.interrupt.enable()
        # 具体逻辑在 compiler 的 _handle_xxx 方法里
        ```
        """
        
        def __init__(self, compiler: "DeveloperCompiler"):
            self.compiler = compiler
            self.enabled = False
            
            # === Internal Counters ===
            self._k_count = 0
            self._tile_count = 0
            
            # === 当前操作信息（由 flash_attention 设置）===
            self.current_matrix: str = ""  # 当前矩阵名称
            self.current_activation: str = ""  # 当前向量集名称
            self._mlen = 64
            self._blen = 4
            self._batch = 64  # batch size
            
            # === Q-K 循环专用状态（由 _flash_attn_qk_loop_asm 设置）===
            self._q_block_idx = 0
            self._k_block_idx = 0
            self._s_tile_address = 0
        
        # === 属性访问（从 sub_matrix_manager 自动获取）===
        
        @property
        def sub_matrix_manager(self):
            return self.compiler.sub_matrix_manager
        
        @property
        def k_count(self) -> int:
            return self._k_count
        
        @property
        def tile_count(self) -> int:
            return self._tile_count
        
        @property
        def batch(self) -> int:
            """由 projection_T_asm 设置"""
            return self._batch
        
        @property
        def out_features(self) -> int:
            """从 sub_matrix_manager 自动获取"""
            if self.current_matrix and self.current_matrix in self.sub_matrix_manager:
                info = self.sub_matrix_manager[self.current_matrix]
                return info.shape[0]  # Matrix 的 shape[0] 是 out_features
            return self._mlen
        
        @property
        def hidden_size(self) -> int:
            """从 sub_matrix_manager 自动获取"""
            if self.current_matrix and self.current_matrix in self.sub_matrix_manager:
                info = self.sub_matrix_manager[self.current_matrix]
                return info.shape[1]  # Matrix 的 shape[1] 是 hidden_size
            return self._mlen
        
        @property
        def k_block(self) -> int:
            """当前 K block 索引（Q-K 循环中的内层索引）"""
            return self._k_block_idx
        
        @property
        def q_block(self) -> int:
            """当前 Q block 索引（Q-K 循环中的外层索引）"""
            return self._q_block_idx
        
        @property
        def s_tile_address(self) -> int:
            """当前 S tile 在 VRAM 中的地址"""
            return self._s_tile_address
        
        @property
        def mlen(self) -> int:
            return self._mlen
        
        @property
        def blen(self) -> int:
            return self._blen
        
        # === Configuration Methods ===
        
        def reset(self):
            """Reset counters (does not clear current_matrix)"""
            self._k_count = 0
            self._tile_count = 0
            self._q_block_idx = 0
            self._k_block_idx = 0
            self._s_tile_address = 0
        
        def enable(self):
            self.enabled = True
        
        def disable(self):
            self.enabled = False
        
        # === Trigger 方法（由 projection_T_asm 调用）===
        # 只管理时机，调用 compiler 里的方法
        
        def trigger_k_start(self) -> str:
            if not self.enabled:
                return ""
            return self.compiler._handle_k_start()
        
        def trigger_k_prefetch_done(self) -> str:
            if not self.enabled:
                return ""
            result = self.compiler._handle_k_prefetch_done()
            self._k_count += 1
            return result
        
        def trigger_s_tile_done(self) -> str:
            if not self.enabled:
                return ""
            result = self.compiler._handle_s_tile_done()
            self._tile_count += 1
            return result
        
        def trigger_k_end(self) -> str:
            if not self.enabled:
                return ""
            return self.compiler._handle_k_end()
    
    # === DeveloperCompiler 主类 ===
    
    def __init__(self, mlen: int = 64, blen: int = 4):
        """初始化编译器"""
        self.mlen = mlen
        self.blen = blen
        self.sub_matrix_manager = SubMatrixManager(mlen=mlen, blen=blen)
        self.register_allocator = RegisterAllocator()
        self.generated_code = ""  # 累积生成的 ISA 代码
        self.interrupt = self.InterruptManager(self)

    # === Interrupt 处理方法（占位符）===
    
    def _handle_k_start(self) -> str:
        """K block 开始时的处理"""
        return ""
    
    def _handle_k_prefetch_done(self) -> str:
        """K 预取完成后的处理"""
        return ""
    
    def _handle_s_tile_done(self) -> str:
        """S tile 完成时的处理"""
        return ""
    
    def _handle_k_end(self) -> str:
        """K block 结束时的处理"""
        return ""
        
    # =========================================================================
    # Flash Attention Implementation
    # =========================================================================
    
    def _online_softmax_asm(
        self,
        mlen: int,
        s_address: int,
        m_start_address: int,
        scale: float = 1.0,
    ) -> str:
        """
        Online Softmax Computation

        对 S 矩阵的每一行执行：
        1. m_curr = max(S[row], m_old)
        2. m_res = exp(m_old - m_curr)  # 用于后续 O 更新
        3. S'[row] = S[row] - m_curr
        4. P[row] = exp(S'[row])
        5. l_new = l_old * m_res + sum(P[row])

        FP SRAM 布局（从 m_start_address 开始）:
        - [0, mlen): m_old / m_curr
        - [mlen, 2*mlen): m_res = exp(m_old - m_curr)
        - [2*mlen, 3*mlen): l_old / l_new

        Args:
            mlen: 行数（也是每行的列数）
            s_address: S 矩阵在 VRAM 中的起始地址
            m_start_address: FP SRAM 中存储 m/l 值的起始地址
            scale: QK^T 的缩放因子 (1/sqrt(d))
        """
        # 通过 allocator Allocate GP registers
        gp_regs = self.register_allocator.allocate_gp(4)
        gp_s = gp_regs[0]          # S 行地址
        gp_m_addr = gp_regs[1]     # m_old 地址
        gp_m_res_addr = gp_regs[2] # m_res 地址
        gp_l_addr = gp_regs[3]     # l_old 地址

        # Fixed FP register allocation for online softmax pipeline.
        # These registers are shared across _online_softmax_asm, _scale_o_asm,
        # and _final_scaling_asm — they MUST remain consistent across all three.
        # WARNING: Do not use f1-f6 in any code that calls these methods.
        fp_m_old = 1      # f1: m_old value
        fp_m_res = 2      # f2: exp(m_old - m_curr)
        fp_l_old = 3      # f3: l_old value
        fp_sum_p = 4      # f4: sum(P)
        fp_scale = 5      # f5: scale factor
        fp_row_max = 6    # f6: current row max (temporary)
        
        lines = []
        lines.append("; === Online Softmax ===")
        
        # Set address registers
        lines.append(f"S_ADDI_INT gp{gp_s}, gp0, {s_address}")
        lines.append(f"S_ADDI_INT gp{gp_m_addr}, gp0, {m_start_address}")
        lines.append(f"S_ADDI_INT gp{gp_m_res_addr}, gp{gp_m_addr}, {mlen}")
        lines.append(f"S_ADDI_INT gp{gp_l_addr}, gp{gp_m_res_addr}, {mlen}")
        
        # 设置 scale factor（如果需要）
        if scale != 1.0:
            # 假设 scale 已经预存到 FP SRAM 地址 1
            lines.append(f"S_LD_FP f{fp_scale}, gp0, 1")
        
        for row in range(mlen):
            lines.append(f"; Row {row}")
            
            # 1. 加载 m_old
            lines.append(f"S_LD_FP f{fp_m_old}, gp{gp_m_addr}, {row}")
            
            # 保存 m_old 到 m_res（临时）
            lines.append(f"S_ADD_FP f{fp_m_res}, f{fp_m_old}, f0")
            
            # 2. 对当前行应用 scale（如果需要）
            if scale != 1.0:
                lines.append(f"V_MUL_VF gp{gp_s}, gp{gp_s}, f{fp_scale}, 0")
            
            # 3. 计算 row_max = max(S[row])
            lines.append(f"V_RED_MAX f{fp_row_max}, gp{gp_s}, 0")
            
            # 4. m_curr = max(row_max, m_old) - 关键！Online Softmax 需要和历史 max 比较
            lines.append(f"S_MAX_FP f{fp_m_old}, f{fp_row_max}, f{fp_m_old}")
            
            # 5. m_res = exp(m_old_saved - m_curr)
            lines.append(f"S_SUB_FP f{fp_m_res}, f{fp_m_res}, f{fp_m_old}")
            lines.append(f"S_EXP_FP f{fp_m_res}, f{fp_m_res}, 0")
            
            # 6. 存储 m_res 和 m_curr
            lines.append(f"S_ST_FP f{fp_m_res}, gp{gp_m_res_addr}, {row}")
            lines.append(f"S_ST_FP f{fp_m_old}, gp{gp_m_addr}, {row}")
            
            # 7. S' = S - m_curr
            lines.append(f"V_SUB_VF gp{gp_s}, gp{gp_s}, f{fp_m_old}, 0, 0")
            
            # 8. P = exp(S')
            lines.append(f"V_EXP_V gp{gp_s}, gp{gp_s}, 0, 0")
            
            # 9. 加载 l_old
            lines.append(f"S_LD_FP f{fp_l_old}, gp{gp_l_addr}, {row}")
            
            # 10. sum_p = sum(P)
            lines.append(f"S_ADD_FP f{fp_sum_p}, f0, f0")
            lines.append(f"V_RED_SUM f{fp_sum_p}, gp{gp_s}, 0, 0")
            
            # 11. l_new = l_old * m_res + sum_p
            lines.append(f"S_MUL_FP f{fp_l_old}, f{fp_l_old}, f{fp_m_res}")
            lines.append(f"S_ADD_FP f{fp_l_old}, f{fp_l_old}, f{fp_sum_p}")
            
            # 12. 存储 l_new
            lines.append(f"S_ST_FP f{fp_l_old}, gp{gp_l_addr}, {row}")
            
            # Next row
            lines.append(f"S_ADDI_INT gp{gp_s}, gp{gp_s}, {mlen}")

        self.register_allocator.free_gp(gp_regs)
        return "\n".join(lines) + "\n"

    def _pv_multiply_asm(
        self,
        mlen: int,
        blen: int,
        head_dim: int,
        p_address: int,
        v_hbm_offset_reg: int,
        v_hbm_offset: int,
        pv_address: int,
    ) -> str:
        """
        计算 PV = P @ V
        
        P: (mlen, mlen) 在 VRAM，softmax 结果
        V: (mlen, head_dim) 在 HBM，需要 prefetch 到 MSRAM
        PV: (mlen, head_dim) 在 VRAM
        
        支持 head_dim >= mlen（分块处理 V 的列）
        
        使用 M_MM：PV = P @ V
        
        M_MM 操作：
        - VRAM 提供 (blen, mlen) 的 P tile
        - MSRAM 提供 (mlen, blen) 的 V tile
        - 一次 M_MM 完成 K=mlen 的完整乘法，输出 (blen, blen)
        
        循环结构（当 head_dim > mlen 时）：
        - 最外层：遍历 V 的列块（每 mlen 列一个块）
        - 中层：遍历当前块内的 V 列（每 blen 列一个 tile）
        - 内层：遍历 P 的行（每 blen 行一个 tile）
        """
        assert head_dim % mlen == 0, f"head_dim ({head_dim}) must be multiple of mlen ({mlen})"

        gp_regs = self.register_allocator.allocate_gp(5)
        gp_p = gp_regs[0]       # P 的 VRAM 地址
        gp_v = gp_regs[1]       # V 在 MSRAM 的偏移
        gp_pv = gp_regs[2]      # PV 结果的 VRAM 地址
        gp_hbm = gp_regs[3]     # HBM 偏移临时
        gp_stride = gp_regs[4]  # M_MM_WO 的 stride（= head_dim）

        # V 的列块数（每块 mlen 列）
        num_v_col_blocks = head_dim // mlen
        # NOTE: For head_dim > mlen, V is split into column blocks of mlen width.
        # Each H_PREFETCH_M loads one (mlen, mlen) tile. The stride=1 is correct
        # because V column blocks are stored contiguously in HBM for each tile.
        
        lines = []
        lines.append("; === PV Multiply (P @ V) using M_MM ===")
        lines.append(f"; P: ({mlen}, {mlen}) @ V: ({mlen}, {head_dim}) -> PV: ({mlen}, {head_dim})")
        lines.append(f"; M_MM: (blen, mlen) @ (mlen, blen) -> (blen, blen), K=mlen in one shot")
        lines.append(f"; V 分为 {num_v_col_blocks} 个列块，每块 {mlen} 列")
        lines.append(f"; 存储格式: (batch, mlen, hidden/mlen) 列块优先")
        
        # 注意：STRIDE 已在 flash_attention 主函数中设置为 mlen
        # 这里不需要重新设置，避免覆盖
        
        # M_MM_WO 需要 stride 寄存器（不能用 gp0，因为 gp0=0 时 stride 会变成 1）
        # 存储格式是 (batch, mlen, hidden/mlen)，即列块优先
        # 同一列块内的连续行是相邻存储的，所以 stride = 1
        lines.append(f"S_ADDI_INT gp{gp_stride}, gp0, 1")  # stride = 1 (列块优先存储)
        
        # 最外层：遍历 V 的列块
        for v_col_block in range(num_v_col_blocks):
            lines.append(f"; --- V column block {v_col_block} (columns {v_col_block * mlen} to {(v_col_block + 1) * mlen - 1}) ---")
            
            # Prefetch 当前 V 列块 (mlen, mlen) 到 MSRAM
            # V 在 HBM 中是行优先存储: V[row, col] at offset row * head_dim + col
            # 对于 V[k_idx][:][:, v_col_block*mlen:(v_col_block+1)*mlen]:
            # HBM 偏移 = v_hbm_offset + v_col_block * mlen（列偏移，元素单位）
            v_block_hbm_offset = v_hbm_offset + v_col_block * mlen
            lines.append(f"S_ADDI_INT gp{gp_v}, gp0, 0")
            lines.append(f"S_ADDI_INT gp{gp_hbm}, gp0, {v_block_hbm_offset}")
            lines.append(f"H_PREFETCH_M gp{gp_v}, gp{gp_hbm}, a{v_hbm_offset_reg}, 1, 1")
            
            # 中层：遍历当前块内的 V 列（输出 PV 的列）
            # mat_offset 必须 < mlen 且是 blen 的倍数
            for v_col in range(mlen // blen):
                lines.append(f"; V column {v_col_block * mlen + v_col * blen}")
                
                # V 在 MSRAM 中的列偏移
                v_msram_offset = v_col * blen  # 这个值 < mlen，满足约束
                lines.append(f"S_ADDI_INT gp{gp_v}, gp0, {v_msram_offset}")
                
                # 内层：遍历 P 的行（输出 PV 的行）
                for p_row in range(mlen // blen):
                    # P 的起始地址：第 p_row 个 blen 行
                    p_row_addr = p_address + p_row * blen * mlen
                    lines.append(f"S_ADDI_INT gp{gp_p}, gp0, {p_row_addr}")
                    
                    # M_MM: (blen, mlen) @ (mlen, blen) -> (blen, blen)
                    lines.append(f"M_MM 0, gp{gp_v}, gp{gp_p}")
                    
                    # 写出结果
                    # 存储格式是 (batch, mlen, hidden/mlen)，即列块优先
                    # PV[row, col] 地址 = base + col_block * mlen * mlen + row * mlen + col_in_block
                    # 这里 row = p_row * blen, col_in_block = v_col * blen
                    pv_offset = v_col_block * mlen * mlen + p_row * blen * mlen + v_col * blen
                    lines.append(f"S_ADDI_INT gp{gp_pv}, gp0, {pv_address + pv_offset}")
                    lines.append(f"M_MM_WO gp{gp_pv}, gp{gp_stride}, 0")

        self.register_allocator.free_gp(gp_regs)
        return "\n".join(lines) + "\n"

    def _scale_o_asm(
        self,
        mlen: int,
        head_dim: int,
        seq_len: int,
        m_res_address: int,
        o_address: int,
        row_offset: int = 0,
    ) -> str:
        """
        对 O 进行 m_res 缩放：O[row] = O[row] * m_res[row]
        
        Args:
            mlen: 当前 Q block 的行数
            head_dim: head dimension
            seq_len: 整个输出的行数（用于计算列块偏移）
            m_res_address: FP SRAM 中 m_res 的地址
            o_address: O 在 VRAM 的起始地址
            row_offset: 当前 Q block 在整个 O 中的行偏移
        """
        assert head_dim % mlen == 0, f"head_dim ({head_dim}) must be multiple of mlen ({mlen})"

        gp_regs = self.register_allocator.allocate_gp(2)
        gp_m_res = gp_regs[0]
        gp_o = gp_regs[1]
        fp_m_res = 1  # FP register

        num_col_blocks = head_dim // mlen

        lines = []
        lines.append("; === Scale O by m_res ===")
        lines.append(f"; head_dim = {head_dim}, 每行分 {num_col_blocks} 个块处理")
        lines.append(f"; seq_len = {seq_len}, row_offset = {row_offset}")

        lines.append(f"S_ADDI_INT gp{gp_m_res}, gp0, {m_res_address}")

        for row in range(mlen):
            lines.append(f"S_LD_FP f{fp_m_res}, gp{gp_m_res}, {row}")
            actual_row = row_offset + row

            for col_block in range(num_col_blocks):
                o_addr = o_address + col_block * seq_len * mlen + actual_row * mlen
                lines.append(f"S_ADDI_INT gp{gp_o}, gp0, {o_addr}")
                lines.append(f"V_MUL_VF gp{gp_o}, gp{gp_o}, f{fp_m_res}, 0")

        self.register_allocator.free_gp(gp_regs)
        return "\n".join(lines) + "\n"
    
    def _add_pv_to_o_asm(
        self,
        mlen: int,
        head_dim: int,
        seq_len: int,
        pv_address: int,
        o_address: int,
        row_offset: int = 0,
    ) -> str:
        """
        将 PV 累加到 O：O[row] = O[row] + PV[row]
        
        Args:
            mlen: 当前 Q block 的行数
            head_dim: head dimension
            seq_len: 整个输出的行数（用于计算列块偏移）
            pv_address: PV 在 VRAM 的地址
            o_address: O 在 VRAM 的起始地址
            row_offset: 当前 Q block 在整个 O 中的行偏移
        """
        assert head_dim % mlen == 0, f"head_dim ({head_dim}) must be multiple of mlen ({mlen})"

        gp_regs = self.register_allocator.allocate_gp(2)
        gp_o = gp_regs[0]
        gp_pv = gp_regs[1]

        num_col_blocks = head_dim // mlen

        lines = []
        lines.append("; === Add PV to O ===")
        lines.append(f"; head_dim = {head_dim}, 每行分 {num_col_blocks} 个块处理")
        lines.append(f"; seq_len = {seq_len}, row_offset = {row_offset}")

        for row in range(mlen):
            actual_row = row_offset + row

            for col_block in range(num_col_blocks):
                o_addr = o_address + col_block * seq_len * mlen + actual_row * mlen
                pv_addr = pv_address + col_block * mlen * mlen + row * mlen

                lines.append(f"S_ADDI_INT gp{gp_o}, gp0, {o_addr}")
                lines.append(f"S_ADDI_INT gp{gp_pv}, gp0, {pv_addr}")
                lines.append(f"V_ADD_VV gp{gp_o}, gp{gp_o}, gp{gp_pv}, 0")

        self.register_allocator.free_gp(gp_regs)
        return "\n".join(lines) + "\n"
    
    def _final_scaling_asm(
        self,
        mlen: int,
        head_dim: int,
        seq_len: int,
        l_address: int,
        o_address: int,
        row_offset: int = 0,
    ) -> str:
        """
        最终缩放：O = O / l
        
        对每一行：O[row] = O[row] / l[row]
        
        支持 head_dim >= mlen：
        - V_MUL_VF 一次处理 mlen 个元素
        - 当 head_dim > mlen 时，每行需要处理 head_dim // mlen 次
        
        Args:
            mlen: 当前 Q block 的行数
            head_dim: head dimension（可以是 mlen 的倍数）
            seq_len: 整个输出的行数（用于计算列块偏移）
            l_address: FP SRAM 中 l 的地址
            o_address: O 在 VRAM 的起始地址（整个 O 矩阵）
            row_offset: 当前 Q block 在整个 O 中的行偏移
        """
        assert head_dim % mlen == 0, f"head_dim ({head_dim}) must be multiple of mlen ({mlen})"

        gp_regs = self.register_allocator.allocate_gp(2)
        gp_l = gp_regs[0]
        gp_o = gp_regs[1]
        fp_l = 1  # FP register

        # 每行需要处理的 mlen 块数
        num_col_blocks = head_dim // mlen

        lines = []
        lines.append("; === Final Scaling O = O / l ===")
        lines.append(f"; head_dim = {head_dim}, 每行分 {num_col_blocks} 个块处理")
        lines.append(f"; 存储格式: (seq_len, mlen, head_dim/mlen) 列块优先")
        lines.append(f"; seq_len = {seq_len}, row_offset = {row_offset}")

        lines.append(f"S_ADDI_INT gp{gp_l}, gp0, {l_address}")

        for row in range(mlen):
            lines.append(f"S_LD_FP f{fp_l}, gp{gp_l}, {row}")
            lines.append(f"S_RECI_FP f{fp_l}, f{fp_l}, 0")
            actual_row = row_offset + row

            for col_block in range(num_col_blocks):
                o_addr = o_address + col_block * seq_len * mlen + actual_row * mlen
                lines.append(f"S_ADDI_INT gp{gp_o}, gp0, {o_addr}")
                lines.append(f"V_MUL_VF gp{gp_o}, gp{gp_o}, f{fp_l}, 0")

        self.register_allocator.free_gp(gp_regs)
        return "\n".join(lines) + "\n"
    
    def _reset_fpsram_asm(
        self,
        start_address: int,
        count: int,
        value_address: int,  # FP SRAM 地址，0=零，2=-inf
    ) -> str:
        """
        Reset a region of FP SRAM

        Args:
            start_address: 起始地址
            count: 要重置的元素数量
            value_address: 重置值所在的 FP SRAM 地址（0=零，2=-inf）
        """
        gp_regs = self.register_allocator.allocate_gp(1)
        gp_addr = gp_regs[0]

        lines = []
        lines.append(f"; Reset FP SRAM [{start_address}, {start_address + count})")

        lines.append(f"S_ADDI_INT gp{gp_addr}, gp0, {start_address}")
        # Use f1 for FP scalar - FP registers don't go through GP allocator
        lines.append(f"S_LD_FP f1, gp0, {value_address}")

        for i in range(count):
            lines.append(f"S_ST_FP f1, gp{gp_addr}, {i}")

        self.register_allocator.free_gp(gp_regs)
        return "\n".join(lines) + "\n"
    
    def _reset_vram_asm(
        self,
        start_address: int,
        rows: int,
        cols: int,
        total_rows: int,
        mlen: int = 64,
        row_offset: int = 0,
    ) -> str:
        """
        Reset a region of VRAM to zero

        支持 cols >= mlen：
        - V_MUL_VF 一次处理 mlen 个元素
        - 当 cols > mlen 时，每行需要处理 cols // mlen 次

        Args:
            start_address: 整个矩阵的起始地址
            rows: 当前要重置的行数
            cols: 每行的元素数（必须是 mlen 的倍数）
            total_rows: 整个矩阵的总行数（用于计算列块偏移）
            mlen: 向量长度（默认 64）
            row_offset: 当前要重置的行在整个矩阵中的行偏移
        """
        gp_regs = self.register_allocator.allocate_gp(1)
        gp_addr = gp_regs[0]

        # 每行需要处理的 mlen 块数
        num_col_blocks = (cols + mlen - 1) // mlen

        lines = []
        lines.append(f"; Reset VRAM rows [{row_offset}, {row_offset + rows}) of matrix at {start_address}")
        lines.append(f"; {rows} rows x {cols} cols, {num_col_blocks} blocks per row")
        lines.append(f"; 存储格式: (total_rows, mlen, cols/mlen) 列块优先")
        lines.append(f"; total_rows = {total_rows}, row_offset = {row_offset}")

        for row in range(rows):
            actual_row = row_offset + row
            for col_block in range(num_col_blocks):
                addr = start_address + col_block * total_rows * mlen + actual_row * mlen
                lines.append(f"S_ADDI_INT gp{gp_addr}, gp0, {addr}")
                lines.append(f"V_MUL_VF gp{gp_addr}, gp{gp_addr}, f0, 0")

        self.register_allocator.free_gp(gp_regs)
        return "\n".join(lines) + "\n"
    
    
    
    def load_batch(
        self,
        hbm_object_name: str,
        vram_object_name: str,
        vlen: int = 64,
        preload_len: int = 4,
    ) -> str:
        """
        Load_Batch: Load Batch tensor from HBM to VRAM
        
        ⚠️ Important: HBM storage considers scalar (MXFP format)
        - HBM 中每 8 个元素有 1 个 scalar (scale factor)
        - real_data_ratio = (8*8 + 8) / (8 * 8) = 1.125
        - HBM 实际大小 = size * real_data_ratio
        - VRAM 只存 vector 数据，不考虑 scalar
        
        编译器在这一行里要做的事（顺序非常重要）：
        1. VRAM 地址分配（compiler 内部）
        2. 写 symbol table（包含 HBM 实际大小计算）
        3. Generate ISA：HBM → VRAM
        
        Args:
            hbm_object_name: HBM object 名字（必须已通过 add_hbm_object 预注册）
            vram_object_name: 目标 VRAM object 名字（load 结果写入该对象）
            vlen: vector length (MLEN = 64)
            preload_len: preload length (blen = 4)
            
        Returns:
            生成的 ISA 代码片段
        """
        # (1) VRAM 地址分配（compiler 内部）
        # 从 sub_matrix_manager 的 vram_allocator 自动分配
        # 注意：VRAM 只需要逻辑大小（没有 scalar）
        hbm_layout = self.sub_matrix_manager.get_hbm_layout(hbm_object_name)
        h, w = hbm_layout.full_shape
        hbm_addr = hbm_layout.hbm_base_addr
        size = h * w
        vram_base = self.sub_matrix_manager.vram_allocator.allocate(size, name=vram_object_name)
        self.sub_matrix_manager.add_vram_object(
            name=vram_object_name,
            shape=(h, w),
            vram_addr=vram_base,
            dtype="fp16",
            kind="Batch",
            allocate_if_none=False,
            strict=False,
        )
        
        # (2) 写 metadata table（已在上面完成）
        # 从这一刻起，A 的地址不允许再手算，只能查表
        
        # (3) Generate ISA：HBM → VRAM
        
        # 3.1 Set HBM address register
        # 分配一个 address register 用于存储 HBM base address
        addr_reg = self.register_allocator.allocate_addr(1)[0]
        gp_regs_for_addr = self.register_allocator.allocate_gp(1)
        
        isa_code = f"; Load_Batch {hbm_object_name} -> {vram_object_name}\n"
        isa_code += f"; HBM[{hbm_addr}] → VRAM[{vram_base}], shape=({h}, {w})\n"
        
        isa_code += preload_addr_reg_asm(
            addr_reg_to_set=[addr_reg],
            available_registers=gp_regs_for_addr,
            addr_reg_val=[hbm_addr]
        )
        
        # 3.2 Reset GP registers（为 preload_act_asm 准备）
        # preload_act_asm 需要 5 个 GP registers: [a_actual, stride, result, outer_loop, inner_loop]
        gp_regs_for_preload = self.register_allocator.allocate_gp(5)
        isa_code += reset_reg_asm(
            alive_registers=gp_regs_for_preload
        )
        
        # 3.3 生成 preload_act_asm
        isa_code += preload_act_asm(
            vlen=vlen,
            preload_len=preload_len,
            batch=h,
            hidden_size=w,
            alive_registers=gp_regs_for_preload,
            act_vram_offset=vram_base,  # 从 symbol table 查，不是硬编码 0
            activation_offset_reg=addr_reg,  # 指向步骤 3.1 设置的 address register
            stride_size=w  # 通常是 hidden_size
        )
        
        # 释放所有使用的寄存器，确保下次可以使用相同的寄存器
        self.register_allocator.free_gp(gp_regs_for_addr)
        self.register_allocator.free_gp(gp_regs_for_preload)
        self.register_allocator.free_addr([addr_reg])
        
        # 累积生成的代码
        self.generated_code += isa_code
        
        return isa_code
    
    def store_to_hbm(
        self,
        tensor_name: str,
        hbm_addr: Optional[int] = None,
        hbm_object_name: Optional[str] = None,
        hbm_addr_reg: Optional[int] = None,
        vlen: int = 64,
        precision: int = 0,  # 0 = Activation, 1 = KeyValue
        store_amount: int = 4  # HBM_V_Writeback_Amount
    ) -> str:
        """
        Write tensor from VRAM back to HBM
        
        这个函数用于将计算出来的中间结果（如 K）从 VRAM 写回 HBM，
        以便后续操作（如 QK^T）可以从 HBM 读取。
        
        使用 store_act_asm 生成存储代码，支持任意大小的 tensor。
        
        Args:
            tensor_name: tensor 名字（必须在 symbol table 中，且是 Batch 类型）
            hbm_addr: HBM 目标地址。如果为 None，使用 tensor 的原始 HBM 地址（如果有）
            hbm_object_name: 若提供，则自动以该名字注册 HBM object
            hbm_addr_reg: HBM 地址寄存器索引（a0-a7）。如果为 None，自动分配
            vlen: Vector length（默认 64）
            precision: 数据精度（0 = Activation, 1 = KeyValue）
            store_amount: 每次 H_STORE_V 存储的行数（默认 4，对应 HBM_V_Writeback_Amount）
            
        Returns:
            生成的 ISA 代码片段
        """
        # (1) 从 symbol table 查信息
        if tensor_name not in self.sub_matrix_manager:
            raise KeyError(f"Tensor '{tensor_name}' not found in symbol table")
        
        tensor_info = self.sub_matrix_manager[tensor_name]
        
        # 类型检查：允许 Batch 和 VRAMMatrix，因为它们在 VRAM 中的存储格式相同
        if tensor_info.kind not in ("Batch", "VRAMMatrix"):
            raise ValueError(f"Tensor '{tensor_name}' must be Batch or VRAMMatrix to store from VRAM, got {tensor_info.kind}")
        
        if tensor_info.vram_addr is None:
            raise ValueError(f"Tensor '{tensor_name}' has no VRAM address to store")
        
        # (2) 确定 HBM 地址
        if hbm_addr is None:
            # 如果没有指定，使用 tensor 的原始 HBM 地址（如果有）
            if tensor_info.hbm_addr >= 0:
                hbm_addr = tensor_info.hbm_addr
            else:
                raise ValueError(f"Tensor '{tensor_name}' has no HBM address. Please specify hbm_addr.")
        
        # (3) Generate ISA
        batch_size = tensor_info.shape[0]
        hidden_size = tensor_info.shape[1]
        
        isa_code = f"; Store {tensor_name} from VRAM to HBM\n"
        isa_code += f"; VRAM[{tensor_info.vram_addr}] -> HBM[{hbm_addr}], shape=({batch_size}, {hidden_size})\n"
        
        # (4) 分配寄存器
        # store_act_asm 需要 5 个 GP 寄存器
        gp_regs = self.register_allocator.allocate_gp(5)

        # 分配 HBM 地址寄存器
        if hbm_addr_reg is None:
            addr_regs = self.register_allocator.allocate_addr(1)
            hbm_addr_reg = addr_regs[0]
            need_free_addr = True
        else:
            addr_regs = []
            need_free_addr = False

        try:
            # (5) Set HBM address register
            gp_regs_for_addr = self.register_allocator.allocate_gp(2)
            isa_code += preload_addr_reg_asm(
                addr_reg_to_set=[hbm_addr_reg],
                available_registers=gp_regs_for_addr,
                addr_reg_val=[hbm_addr]
            )
            self.register_allocator.free_gp(gp_regs_for_addr)

            # (6) 使用 store_act_asm 生成存储代码
            isa_code += store_act_asm(
                vlen=vlen,
                batch=batch_size,
                hidden_size=hidden_size,
                alive_registers=gp_regs,
                act_vram_offset=tensor_info.vram_addr,
                hbm_addr_reg=hbm_addr_reg,
                stride_size=hidden_size,
                store_amount=store_amount,
            )

            # (7) 更新 symbol table 中的 HBM 地址（如果需要）
            if tensor_info.hbm_addr < 0 or tensor_info.hbm_addr != hbm_addr:
                tensor_info.hbm_addr = hbm_addr
                # 计算 HBM size（考虑 real_data_ratio）
                size = batch_size * hidden_size
                real_data_ratio = 1.125  # 默认值，可以从参数传入
                tensor_info.hbm_size = int(size * real_data_ratio)
        finally:
            # (8) 释放寄存器
            self.register_allocator.free_gp(gp_regs)
            if need_free_addr:
                self.register_allocator.free_addr(addr_regs)

        # (9) 可选：自动注册 HBM object
        if hbm_object_name is not None:
            self.add_hbm_object(
                name=hbm_object_name,
                hbm_addr=hbm_addr,
                shape=(batch_size, hidden_size),
            )
        
        # 累积生成的代码
        self.generated_code += isa_code
        
        return isa_code

    def normalize(
        self,
        tensor_name: str,
        mode: str = "rms",
        eps_offset: int = 1,
        reci_hid_offset: int = 2,
        vlen: Optional[int] = None,
        scratchpad_vram_addr: Optional[int] = None,
    ) -> str:
        """
        Normalize a VRAM tensor in-place.

        Supports:
        - mode="rms":   RMSNorm
        - mode="layer": LayerNorm

        Args:
            tensor_name: Tensor name in symbol table (must have VRAM address)
            mode: "rms" or "layer"
            eps_offset: FPRAM address of epsilon
            reci_hid_offset: FPRAM address of 1/hidden_dim
            vlen: vector length (default: self.mlen)
            scratchpad_vram_addr: scratchpad VRAM address (default: auto-allocate temporary space)
        """
        if tensor_name not in self.sub_matrix_manager:
            raise KeyError(f"Tensor '{tensor_name}' not found in symbol table")

        tensor_info = self.sub_matrix_manager[tensor_name]
        if tensor_info.vram_addr is None:
            raise ValueError(f"Tensor '{tensor_name}' has no VRAM address")

        batch_size, hidden_dim = tensor_info.shape
        if vlen is None:
            vlen = self.mlen
        if hidden_dim % vlen != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by vlen ({vlen}) for normalization_asm"
            )

        mode = mode.lower()
        if mode not in ("rms", "layer"):
            raise ValueError(f"Unsupported normalization mode: {mode}. Expected 'rms' or 'layer'.")

        gp_regs = self.register_allocator.allocate_gp(4)

        temp_scratchpad_name = None
        if scratchpad_vram_addr is None:
            temp_scratchpad_name = f"__norm_scratch__{tensor_name}__{len(self.generated_code)}"
            scratchpad_vram_addr = self.sub_matrix_manager.vram_allocator.allocate(vlen, name=temp_scratchpad_name)

        try:
            isa_code = f"; Normalize ({mode}) {tensor_name}, shape=({batch_size}, {hidden_dim})\n"
            if mode == "rms":
                isa_code += rms_norm_asm(
                    _eps_offset=eps_offset,
                    reci_hid_offset=reci_hid_offset,
                    alive_registers=gp_regs,
                    activation_base_address=tensor_info.vram_addr,
                    scratchpad_base_address=scratchpad_vram_addr,
                    vlen=vlen,
                    batch_size=batch_size,
                    hidden_dim=hidden_dim,
                )
            else:
                isa_code += layer_norm_asm(
                    _eps_offset=eps_offset,
                    reci_hid_offset=reci_hid_offset,
                    alive_registers=gp_regs,
                    activation_base_address=tensor_info.vram_addr,
                    scratchpad_base_address=scratchpad_vram_addr,
                    vlen=vlen,
                    batch_size=batch_size,
                    hidden_dim=hidden_dim,
                )

            self.generated_code += isa_code
            return isa_code
        finally:
            # Always release allocated GP registers used by normalization template.
            self.register_allocator.free_gp(gp_regs)
            if temp_scratchpad_name is not None:
                self.sub_matrix_manager.vram_allocator.free(temp_scratchpad_name, strict=False)

    def rope(
        self,
        x_name: str,
        x_rot_name: str,
        cos_name: str,
        sin_name: str,
    ) -> str:
        """Apply RoPE in-place: x = x * cos + rotate_half(x) * sin

        All four tensors must already be in VRAM with the same shape (seq_len, head_dim).
        x_rot must be preloaded by the caller as rotate_half(x).
        """
        x_info    = self.sub_matrix_manager[x_name]
        xrot_info = self.sub_matrix_manager[x_rot_name]
        cos_info  = self.sub_matrix_manager[cos_name]
        sin_info  = self.sub_matrix_manager[sin_name]

        if x_info.vram_addr is None:
            raise ValueError(f"Tensor '{x_name}' has no VRAM address")

        seq_len, head_dim = x_info.shape
        vlen = self.mlen

        if head_dim % vlen != 0:
            raise ValueError(
                f"head_dim ({head_dim}) must be divisible by vlen ({vlen}) for rope"
            )

        gp_regs = self.register_allocator.allocate_gp(5)

        scratch_name = f"__rope_scratch__{x_name}__{len(self.generated_code)}"
        scratch_addr = self.sub_matrix_manager.vram_allocator.allocate(vlen, name=scratch_name)

        try:
            isa_code = rope_asm(
                alive_registers=gp_regs,
                x_base_address=x_info.vram_addr,
                x_rot_base_address=xrot_info.vram_addr,
                cos_base_address=cos_info.vram_addr,
                sin_base_address=sin_info.vram_addr,
                scratchpad_base_address=scratch_addr,
                vlen=vlen,
                seq_len=seq_len,
                head_dim=head_dim,
            )
            self.generated_code += isa_code
            return isa_code
        finally:
            self.register_allocator.free_gp(gp_regs)
            self.sub_matrix_manager.vram_allocator.free(scratch_name, strict=False)

    def get_code(self) -> str:
        """Get all accumulated generated ISA code"""
        return self.generated_code
    
    def reset(self):
        """Reset compiler state (clear code, but retain symbol table)"""
        self.generated_code = ""
        self.register_allocator = RegisterAllocator()
        self.sub_matrix_manager.reset()
    
    def print_symbol_table(self):
        """Print symbol table"""
        self.sub_matrix_manager.print_table()

    def get_symbol_table(self):
        """Get managed object table view."""
        return self.sub_matrix_manager

    def get_tensor_info(self, name: str):
        """Get unified tensor/object info by name."""
        return self.sub_matrix_manager[name]

    def add_hbm_object(
        self,
        name: str,
        hbm_addr: int,
        shape: Tuple[int, int],
        real_data_ratio: float = 1.125,
    ):
        """Register an HBM object and build its HBM layout."""
        return self.sub_matrix_manager.add_hbm_object(
            name=name,
            shape=shape,
            hbm_addr=hbm_addr,
            real_data_ratio=real_data_ratio,
        )

    def free_hbm_object(self, name: str, strict: bool = False):
        """Free an HBM object by name."""
        return self.sub_matrix_manager.free_hbm_object(name, strict=strict)

    def get_vram_addr(self, name: str) -> int:
        """Get VRAM base address of an object."""
        info = self.get_tensor_info(name)
        if info.vram_addr is None:
            raise ValueError(f"Object '{name}' has no VRAM address")
        return info.vram_addr

    def get_vram_tile_addr(
        self,
        name: str,
        tile_row_idx: int = 0,
        tile_col_idx: int = 0,
    ) -> int:
        """
        Get VRAM address of a specific tile (sub-block) in a VRAM matrix.

        Args:
            name: matrix name
            tile_row_idx: tile row index (0-based)
            tile_col_idx: tile col index (0-based)
        """
        self._ensure_vram_matrix_layout(name)
        sub = self.sub_matrix_manager.get_vram_sub_block(name, tile_row_idx, tile_col_idx)
        return sub.vram_addr

    def ensure_hbm_sub_matrix(
        self,
        name: str,
        hbm_addr: int,
        shape: Tuple[int, int],
        real_data_ratio: float = 1.125,
    ):
        """Ensure HBM matrix layout exists."""
        if name in self.sub_matrix_manager.hbm_matrices:
            return
        self.add_hbm_object(
            name=name,
            hbm_addr=hbm_addr,
            shape=shape,
            real_data_ratio=real_data_ratio,
        )

    def ensure_vram_matrix_layout(self, name: str, shape: Tuple[int, int]):
        """Ensure VRAM matrix layout exists for an already allocated VRAM object."""
        if name in self.sub_matrix_manager.vram_matrices:
            return
        vram_addr = self.get_vram_addr(name)
        self.sub_matrix_manager.add_vram_object(
            name=name,
            shape=shape,
            vram_addr=vram_addr,
            allocate_if_none=False,
        )

    def free_vram_object(self, name: str, strict: bool = False):
        """Free a VRAM object by name."""
        return self.sub_matrix_manager.free_vram_object(name, strict=strict)

    # =========================================================================
    # FP Register & FPRAM Management
    # =========================================================================

    def allocate_fp_reg(self, count: int = 1) -> List[int]:
        """Allocate FP registers (f0-f7)"""
        return self.register_allocator.allocate_fp(count)

    def free_fp_reg(self, registers: List[int]):
        """Free FP registers"""
        self.register_allocator.free_fp(registers)

    def allocate_fpram(self, name: str, size: int) -> int:
        """Allocate FPRAM space, returns address"""
        info = self.sub_matrix_manager.add_fpram_object(name=name, size=size)
        if info.fpram_addr is None:
            raise RuntimeError(f"Failed to allocate FPRAM for '{name}'")
        return info.fpram_addr

    def free_fpram(self, name: str, strict: bool = True):
        """Free FPRAM object by name."""
        return self.sub_matrix_manager.free_fpram_object(name, strict=strict)

    def save_fpram_state(self) -> int:
        """Save FPRAM allocator snapshot"""
        return self.sub_matrix_manager.fpram_allocator.save_state()

    def restore_fpram_state(self, snapshot: int):
        """Restore FPRAM allocator snapshot"""
        self.sub_matrix_manager.fpram_allocator.restore_state(snapshot)

    def list_fpram_allocations(self) -> List[str]:
        """List currently allocated FPRAM object names."""
        return list(self.sub_matrix_manager.fpram_allocator.allocations.keys())

    def get_fpram_addr(self, name: str) -> int:
        """Get FPRAM base address from object name."""
        return self.sub_matrix_manager.get_fpram_layout(name).fpram_addr

    def get_fpram_size(self, name: str) -> int:
        """Get FPRAM allocation size from object name."""
        return self.sub_matrix_manager.get_fpram_layout(name).size

    # =========================================================================
    # FPVar ISA helpers (address-based)
    # =========================================================================

    def fpvar_copy_asm(self, src_addr: int, dst_addr: int, count: int) -> str:
        if count <= 0:
            isa_code = f"; FPVar Copy skipped: count={count}\n"
            self.generated_code += isa_code
            return isa_code
        gp = self.register_allocator.allocate_gp(3)
        gp_src, gp_dst, gp_loop = gp
        lines = [f"; FPVar Copy: FPRAM[{dst_addr}:{dst_addr+count}] = FPRAM[{src_addr}:{src_addr+count}]"]
        lines.append(f"S_ADDI_INT gp{gp_src}, gp0, {src_addr}")
        lines.append(f"S_ADDI_INT gp{gp_dst}, gp0, {dst_addr}")
        lines.append(f"C_LOOP_START gp{gp_loop}, {count}")
        lines.append(f"S_LD_FP f1, gp{gp_src}, 0")
        lines.append(f"S_ST_FP f1, gp{gp_dst}, 0")
        lines.append(f"S_ADDI_INT gp{gp_src}, gp{gp_src}, 1")
        lines.append(f"S_ADDI_INT gp{gp_dst}, gp{gp_dst}, 1")
        lines.append(f"C_LOOP_END gp{gp_loop}")
        self.register_allocator.free_gp(gp)
        isa_code = "\n".join(lines) + "\n"
        self.generated_code += isa_code
        return isa_code

    def fpvar_fill_from_fpram_asm(self, dst_addr: int, src_fpram_addr: int, count: int) -> str:
        if count <= 0:
            isa_code = f"; FPVar Fill skipped: count={count}\n"
            self.generated_code += isa_code
            return isa_code
        gp = self.register_allocator.allocate_gp(3)
        gp_src, gp_dst, gp_loop = gp
        lines = [f"; FPVar Fill: FPRAM[{dst_addr}:{dst_addr+count}] = FPRAM[{src_fpram_addr}]"]
        lines.append(f"S_ADDI_INT gp{gp_src}, gp0, {src_fpram_addr}")
        lines.append(f"S_ADDI_INT gp{gp_dst}, gp0, {dst_addr}")
        lines.append(f"S_LD_FP f1, gp{gp_src}, 0")
        lines.append(f"C_LOOP_START gp{gp_loop}, {count}")
        lines.append(f"S_ST_FP f1, gp{gp_dst}, 0")
        lines.append(f"S_ADDI_INT gp{gp_dst}, gp{gp_dst}, 1")
        lines.append(f"C_LOOP_END gp{gp_loop}")
        self.register_allocator.free_gp(gp)
        isa_code = "\n".join(lines) + "\n"
        self.generated_code += isa_code
        return isa_code

    def fpvar_reci_asm(self, src_addr: int, dst_addr: int, count: int) -> str:
        if count <= 0:
            isa_code = f"; FPVar Reci skipped: count={count}\n"
            self.generated_code += isa_code
            return isa_code
        gp = self.register_allocator.allocate_gp(3)
        gp_src, gp_dst, gp_loop = gp
        lines = [f"; FPVar Reci: dst = 1/src, count={count}"]
        lines.append(f"S_ADDI_INT gp{gp_src}, gp0, {src_addr}")
        lines.append(f"S_ADDI_INT gp{gp_dst}, gp0, {dst_addr}")
        lines.append(f"C_LOOP_START gp{gp_loop}, {count}")
        lines.append(f"S_LD_FP f1, gp{gp_src}, 0")
        lines.append("S_RECI_FP f1, f1, 0")
        lines.append(f"S_ST_FP f1, gp{gp_dst}, 0")
        lines.append(f"S_ADDI_INT gp{gp_src}, gp{gp_src}, 1")
        lines.append(f"S_ADDI_INT gp{gp_dst}, gp{gp_dst}, 1")
        lines.append(f"C_LOOP_END gp{gp_loop}")
        self.register_allocator.free_gp(gp)
        isa_code = "\n".join(lines) + "\n"
        self.generated_code += isa_code
        return isa_code

    def fpvar_exp_asm(self, src_addr: int, dst_addr: int, count: int) -> str:
        if count <= 0:
            isa_code = f"; FPVar Exp skipped: count={count}\n"
            self.generated_code += isa_code
            return isa_code
        gp = self.register_allocator.allocate_gp(3)
        gp_src, gp_dst, gp_loop = gp
        lines = [f"; FPVar Exp: dst = exp(src), count={count}"]
        lines.append(f"S_ADDI_INT gp{gp_src}, gp0, {src_addr}")
        lines.append(f"S_ADDI_INT gp{gp_dst}, gp0, {dst_addr}")
        lines.append(f"C_LOOP_START gp{gp_loop}, {count}")
        lines.append(f"S_LD_FP f1, gp{gp_src}, 0")
        lines.append("S_EXP_FP f1, f1, 0")
        lines.append(f"S_ST_FP f1, gp{gp_dst}, 0")
        lines.append(f"S_ADDI_INT gp{gp_src}, gp{gp_src}, 1")
        lines.append(f"S_ADDI_INT gp{gp_dst}, gp{gp_dst}, 1")
        lines.append(f"C_LOOP_END gp{gp_loop}")
        self.register_allocator.free_gp(gp)
        isa_code = "\n".join(lines) + "\n"
        self.generated_code += isa_code
        return isa_code

    def fpvar_add_asm(self, src1_addr: int, src2_addr: int, dst_addr: int, count: int) -> str:
        if count <= 0:
            isa_code = f"; FPVar Add skipped: count={count}\n"
            self.generated_code += isa_code
            return isa_code
        gp = self.register_allocator.allocate_gp(4)
        gp_a, gp_b, gp_dst, gp_loop = gp
        lines = [f"; FPVar Add: dst = src1 + src2, count={count}"]
        lines.append(f"S_ADDI_INT gp{gp_a}, gp0, {src1_addr}")
        lines.append(f"S_ADDI_INT gp{gp_b}, gp0, {src2_addr}")
        lines.append(f"S_ADDI_INT gp{gp_dst}, gp0, {dst_addr}")
        lines.append(f"C_LOOP_START gp{gp_loop}, {count}")
        lines.append(f"S_LD_FP f1, gp{gp_a}, 0")
        lines.append(f"S_LD_FP f2, gp{gp_b}, 0")
        lines.append("S_ADD_FP f1, f1, f2")
        lines.append(f"S_ST_FP f1, gp{gp_dst}, 0")
        lines.append(f"S_ADDI_INT gp{gp_a}, gp{gp_a}, 1")
        lines.append(f"S_ADDI_INT gp{gp_b}, gp{gp_b}, 1")
        lines.append(f"S_ADDI_INT gp{gp_dst}, gp{gp_dst}, 1")
        lines.append(f"C_LOOP_END gp{gp_loop}")
        self.register_allocator.free_gp(gp)
        isa_code = "\n".join(lines) + "\n"
        self.generated_code += isa_code
        return isa_code

    def fpvar_sub_asm(self, src1_addr: int, src2_addr: int, dst_addr: int, count: int) -> str:
        if count <= 0:
            isa_code = f"; FPVar Sub skipped: count={count}\n"
            self.generated_code += isa_code
            return isa_code
        gp = self.register_allocator.allocate_gp(4)
        gp_a, gp_b, gp_dst, gp_loop = gp
        lines = [f"; FPVar Sub: dst = src1 - src2, count={count}"]
        lines.append(f"S_ADDI_INT gp{gp_a}, gp0, {src1_addr}")
        lines.append(f"S_ADDI_INT gp{gp_b}, gp0, {src2_addr}")
        lines.append(f"S_ADDI_INT gp{gp_dst}, gp0, {dst_addr}")
        lines.append(f"C_LOOP_START gp{gp_loop}, {count}")
        lines.append(f"S_LD_FP f1, gp{gp_a}, 0")
        lines.append(f"S_LD_FP f2, gp{gp_b}, 0")
        lines.append("S_SUB_FP f1, f1, f2")
        lines.append(f"S_ST_FP f1, gp{gp_dst}, 0")
        lines.append(f"S_ADDI_INT gp{gp_a}, gp{gp_a}, 1")
        lines.append(f"S_ADDI_INT gp{gp_b}, gp{gp_b}, 1")
        lines.append(f"S_ADDI_INT gp{gp_dst}, gp{gp_dst}, 1")
        lines.append(f"C_LOOP_END gp{gp_loop}")
        self.register_allocator.free_gp(gp)
        isa_code = "\n".join(lines) + "\n"
        self.generated_code += isa_code
        return isa_code

    def fpvar_mul_asm(self, src1_addr: int, src2_addr: int, dst_addr: int, count: int) -> str:
        if count <= 0:
            isa_code = f"; FPVar Mul skipped: count={count}\n"
            self.generated_code += isa_code
            return isa_code
        gp = self.register_allocator.allocate_gp(4)
        gp_a, gp_b, gp_dst, gp_loop = gp
        lines = [f"; FPVar Mul: dst = src1 * src2, count={count}"]
        lines.append(f"S_ADDI_INT gp{gp_a}, gp0, {src1_addr}")
        lines.append(f"S_ADDI_INT gp{gp_b}, gp0, {src2_addr}")
        lines.append(f"S_ADDI_INT gp{gp_dst}, gp0, {dst_addr}")
        lines.append(f"C_LOOP_START gp{gp_loop}, {count}")
        lines.append(f"S_LD_FP f1, gp{gp_a}, 0")
        lines.append(f"S_LD_FP f2, gp{gp_b}, 0")
        lines.append("S_MUL_FP f1, f1, f2")
        lines.append(f"S_ST_FP f1, gp{gp_dst}, 0")
        lines.append(f"S_ADDI_INT gp{gp_a}, gp{gp_a}, 1")
        lines.append(f"S_ADDI_INT gp{gp_b}, gp{gp_b}, 1")
        lines.append(f"S_ADDI_INT gp{gp_dst}, gp{gp_dst}, 1")
        lines.append(f"C_LOOP_END gp{gp_loop}")
        self.register_allocator.free_gp(gp)
        isa_code = "\n".join(lines) + "\n"
        self.generated_code += isa_code
        return isa_code

    def fpvar_max_asm(self, src1_addr: int, src2_addr: int, dst_addr: int, count: int) -> str:
        if count <= 0:
            isa_code = f"; FPVar Max skipped: count={count}\n"
            self.generated_code += isa_code
            return isa_code
        gp = self.register_allocator.allocate_gp(4)
        gp_a, gp_b, gp_dst, gp_loop = gp
        lines = [f"; FPVar Max: dst = max(src1, src2), count={count}"]
        lines.append(f"S_ADDI_INT gp{gp_a}, gp0, {src1_addr}")
        lines.append(f"S_ADDI_INT gp{gp_b}, gp0, {src2_addr}")
        lines.append(f"S_ADDI_INT gp{gp_dst}, gp0, {dst_addr}")
        lines.append(f"C_LOOP_START gp{gp_loop}, {count}")
        lines.append(f"S_LD_FP f1, gp{gp_a}, 0")
        lines.append(f"S_LD_FP f2, gp{gp_b}, 0")
        lines.append("S_MAX_FP f1, f1, f2")
        lines.append(f"S_ST_FP f1, gp{gp_dst}, 0")
        lines.append(f"S_ADDI_INT gp{gp_a}, gp{gp_a}, 1")
        lines.append(f"S_ADDI_INT gp{gp_b}, gp{gp_b}, 1")
        lines.append(f"S_ADDI_INT gp{gp_dst}, gp{gp_dst}, 1")
        lines.append(f"C_LOOP_END gp{gp_loop}")
        self.register_allocator.free_gp(gp)
        isa_code = "\n".join(lines) + "\n"
        self.generated_code += isa_code
        return isa_code

    def fpvar_sum_asm(self, src_addr: int, dst_addr: int, count: int) -> str:
        if count <= 0:
            isa_code = f"; FPVar Sum skipped: count={count}\n"
            self.generated_code += isa_code
            return isa_code
        gp = self.register_allocator.allocate_gp(3)
        gp_src, gp_dst, gp_loop = gp
        lines = [f"; FPVar Sum: FPRAM[{dst_addr}] = sum(FPRAM[{src_addr}:{src_addr+count}])"]
        lines.append(f"S_ADDI_INT gp{gp_src}, gp0, {src_addr}")
        lines.append(f"S_ADDI_INT gp{gp_dst}, gp0, {dst_addr}")
        lines.append("S_ADD_FP f1, f0, f0")
        lines.append(f"C_LOOP_START gp{gp_loop}, {count}")
        lines.append(f"S_LD_FP f2, gp{gp_src}, 0")
        lines.append("S_ADD_FP f1, f1, f2")
        lines.append(f"S_ADDI_INT gp{gp_src}, gp{gp_src}, 1")
        lines.append(f"C_LOOP_END gp{gp_loop}")
        lines.append(f"S_ST_FP f1, gp{gp_dst}, 0")
        self.register_allocator.free_gp(gp)
        isa_code = "\n".join(lines) + "\n"
        self.generated_code += isa_code
        return isa_code

    def fpvar_shift_asm(
        self,
        src_addr: int,
        dst_addr: int,
        count: int,
        shift: int,
        fill_fpram_addr: int = 0,
    ) -> str:
        """
        Shift FPVar into dst.
        - shift > 0: right shift (leading positions filled)
        - shift < 0: left shift (trailing positions filled)
        """
        gp = self.register_allocator.allocate_gp(3)
        gp_src, gp_dst, gp_fill = gp
        lines = [
            f"; FPVar Shift: dst=shift(src, shift={shift}), count={count}, fill=FPRAM[{fill_fpram_addr}]"
        ]
        lines.append(f"S_ADDI_INT gp{gp_src}, gp0, {src_addr}")
        lines.append(f"S_ADDI_INT gp{gp_dst}, gp0, {dst_addr}")
        lines.append(f"S_ADDI_INT gp{gp_fill}, gp0, {fill_fpram_addr}")
        lines.append(f"S_LD_FP f3, gp{gp_fill}, 0")

        for i in range(count):
            src_idx = i - shift
            if 0 <= src_idx < count:
                lines.append(f"S_LD_FP f1, gp{gp_src}, {src_idx}")
                lines.append(f"S_ST_FP f1, gp{gp_dst}, {i}")
            else:
                lines.append(f"S_ST_FP f3, gp{gp_dst}, {i}")

        self.register_allocator.free_gp(gp)
        isa_code = "\n".join(lines) + "\n"
        self.generated_code += isa_code
        return isa_code

    # =========================================================================
    # FPVar helpers (name-based)
    # =========================================================================

    def fpram_copy(self, src_name: str, dst_name: str, count: Optional[int] = None) -> str:
        if count is None:
            count = min(self.get_fpram_size(src_name), self.get_fpram_size(dst_name))
        return self.fpvar_copy_asm(self.get_fpram_addr(src_name), self.get_fpram_addr(dst_name), count)

    def fpram_reci(self, src_name: str, dst_name: str, count: Optional[int] = None) -> str:
        if count is None:
            count = min(self.get_fpram_size(src_name), self.get_fpram_size(dst_name))
        return self.fpvar_reci_asm(self.get_fpram_addr(src_name), self.get_fpram_addr(dst_name), count)

    def fpram_exp(self, src_name: str, dst_name: str, count: Optional[int] = None) -> str:
        if count is None:
            count = min(self.get_fpram_size(src_name), self.get_fpram_size(dst_name))
        return self.fpvar_exp_asm(self.get_fpram_addr(src_name), self.get_fpram_addr(dst_name), count)

    def fpram_add(self, src1_name: str, src2_name: str, dst_name: str, count: Optional[int] = None) -> str:
        if count is None:
            count = min(self.get_fpram_size(src1_name), self.get_fpram_size(src2_name), self.get_fpram_size(dst_name))
        return self.fpvar_add_asm(self.get_fpram_addr(src1_name), self.get_fpram_addr(src2_name), self.get_fpram_addr(dst_name), count)

    def fpram_sub(self, src1_name: str, src2_name: str, dst_name: str, count: Optional[int] = None) -> str:
        if count is None:
            count = min(self.get_fpram_size(src1_name), self.get_fpram_size(src2_name), self.get_fpram_size(dst_name))
        return self.fpvar_sub_asm(self.get_fpram_addr(src1_name), self.get_fpram_addr(src2_name), self.get_fpram_addr(dst_name), count)

    def fpram_mul(self, src1_name: str, src2_name: str, dst_name: str, count: Optional[int] = None) -> str:
        if count is None:
            count = min(self.get_fpram_size(src1_name), self.get_fpram_size(src2_name), self.get_fpram_size(dst_name))
        return self.fpvar_mul_asm(self.get_fpram_addr(src1_name), self.get_fpram_addr(src2_name), self.get_fpram_addr(dst_name), count)

    def fpram_max(self, src1_name: str, src2_name: str, dst_name: str, count: Optional[int] = None) -> str:
        if count is None:
            count = min(self.get_fpram_size(src1_name), self.get_fpram_size(src2_name), self.get_fpram_size(dst_name))
        return self.fpvar_max_asm(self.get_fpram_addr(src1_name), self.get_fpram_addr(src2_name), self.get_fpram_addr(dst_name), count)

    def fpram_sum(self, src_name: str, dst_name: str, count: Optional[int] = None) -> str:
        if count is None:
            count = self.get_fpram_size(src_name)
        return self.fpvar_sum_asm(
            self.get_fpram_addr(src_name),
            self.get_fpram_addr(dst_name),
            count,
        )

    def fpram_shift(
        self,
        src_name: str,
        dst_name: str,
        shift: int,
        count: Optional[int] = None,
        fill_fpram_name: Optional[str] = None,
    ) -> str:
        if count is None:
            count = min(self.get_fpram_size(src_name), self.get_fpram_size(dst_name))
        fill_addr = 0 if fill_fpram_name is None else self.get_fpram_addr(fill_fpram_name)
        return self.fpvar_shift_asm(
            src_addr=self.get_fpram_addr(src_name),
            dst_addr=self.get_fpram_addr(dst_name),
            count=count,
            shift=shift,
            fill_fpram_addr=fill_addr,
        )

    def fpram_fill_from_fpram(self, dst_name: str, src_fpram_addr: int, count: Optional[int] = None) -> str:
        if count is None:
            count = self.get_fpram_size(dst_name)
        return self.fpvar_fill_from_fpram_asm(
            dst_addr=self.get_fpram_addr(dst_name),
            src_fpram_addr=src_fpram_addr,
            count=count,
        )

    # =========================================================================
    # Tile-row helpers (name-based)
    # =========================================================================

    def tile_row_max(
        self,
        source_matrix: str,
        row_map: List[Tuple[int, int]],
        tile_row_idx: int = 0,
        tile_col_idx: int = 0,
    ) -> str:
        source_addr = self.get_vram_tile_addr(source_matrix, tile_row_idx, tile_col_idx)
        return self.tile_row_max_asm(source_addr, row_map)

    def tile_row_sum(
        self,
        source_matrix: str,
        row_map: List[Tuple[int, int]],
        tile_row_idx: int = 0,
        tile_col_idx: int = 0,
    ) -> str:
        source_addr = self.get_vram_tile_addr(source_matrix, tile_row_idx, tile_col_idx)
        return self.tile_row_sum_asm(source_addr, row_map)

    def tile_row_exp(
        self,
        matrix_name: str,
        rows: List[int],
        tile_row_idx: int = 0,
        tile_col_idx: int = 0,
    ) -> str:
        matrix_addr = self.get_vram_tile_addr(matrix_name, tile_row_idx, tile_col_idx)
        return self.tile_row_exp_asm(matrix_addr, rows)

    def tile_row_reci(
        self,
        matrix_name: str,
        rows: List[int],
        tile_row_idx: int = 0,
        tile_col_idx: int = 0,
    ) -> str:
        matrix_addr = self.get_vram_tile_addr(matrix_name, tile_row_idx, tile_col_idx)
        return self.tile_row_reci_asm(matrix_addr, rows)

    def tile_row_sub_fp(
        self,
        matrix_name: str,
        row_map: List[Tuple[int, int]],
        tile_row_idx: int = 0,
        tile_col_idx: int = 0,
    ) -> str:
        matrix_addr = self.get_vram_tile_addr(matrix_name, tile_row_idx, tile_col_idx)
        return self.tile_row_sub_fp_asm(matrix_addr, row_map)

    def tile_row_mul_fp(
        self,
        matrix_name: str,
        row_map: List[Tuple[int, int]],
        tile_row_idx: int = 0,
        tile_col_idx: int = 0,
    ) -> str:
        matrix_addr = self.get_vram_tile_addr(matrix_name, tile_row_idx, tile_col_idx)
        return self.tile_row_mul_fp_asm(matrix_addr, row_map)

    def tile_row_add_fp(
        self,
        matrix_name: str,
        row_map: List[Tuple[int, int]],
        tile_row_idx: int = 0,
        tile_col_idx: int = 0,
    ) -> str:
        matrix_addr = self.get_vram_tile_addr(matrix_name, tile_row_idx, tile_col_idx)
        return self.tile_row_add_fp_asm(matrix_addr, row_map)

    def tile_row_add(
        self,
        dst_matrix: str,
        src_matrix: str,
        rows: List[int],
        dst_tile_row_idx: int = 0,
        dst_tile_col_idx: int = 0,
        src_tile_row_idx: int = 0,
        src_tile_col_idx: int = 0,
    ) -> str:
        dst_addr = self.get_vram_tile_addr(dst_matrix, dst_tile_row_idx, dst_tile_col_idx)
        src_addr = self.get_vram_tile_addr(src_matrix, src_tile_row_idx, src_tile_col_idx)
        return self.tile_row_add_asm(dst_addr, src_addr, rows)

    def tile_row_sub(
        self,
        dst_matrix: str,
        src_matrix: str,
        rows: List[int],
        dst_tile_row_idx: int = 0,
        dst_tile_col_idx: int = 0,
        src_tile_row_idx: int = 0,
        src_tile_col_idx: int = 0,
    ) -> str:
        dst_addr = self.get_vram_tile_addr(dst_matrix, dst_tile_row_idx, dst_tile_col_idx)
        src_addr = self.get_vram_tile_addr(src_matrix, src_tile_row_idx, src_tile_col_idx)
        return self.tile_row_sub_asm(dst_addr, src_addr, rows)

    def tile_row_mul(
        self,
        dst_matrix: str,
        src_matrix: str,
        rows: List[int],
        dst_tile_row_idx: int = 0,
        dst_tile_col_idx: int = 0,
        src_tile_row_idx: int = 0,
        src_tile_col_idx: int = 0,
    ) -> str:
        dst_addr = self.get_vram_tile_addr(dst_matrix, dst_tile_row_idx, dst_tile_col_idx)
        src_addr = self.get_vram_tile_addr(src_matrix, src_tile_row_idx, src_tile_col_idx)
        return self.tile_row_mul_asm(dst_addr, src_addr, rows)

    def tile_row_mul_fp_broadcast(
        self,
        matrix_name: str,
        fpram_scalar_addr: int,
        rows: List[int],
        tile_row_idx: int = 0,
        tile_col_idx: int = 0,
    ) -> str:
        matrix_addr = self.get_vram_tile_addr(matrix_name, tile_row_idx, tile_col_idx)
        return self.tile_row_mul_fp_broadcast_asm(matrix_addr, fpram_scalar_addr, rows)

    def vram_fill_zero(
        self,
        matrix_name: str,
        rows: List[int],
        tile_row_idx: int = 0,
        tile_col_idx: int = 0,
    ) -> str:
        matrix_addr = self.get_vram_tile_addr(matrix_name, tile_row_idx, tile_col_idx)
        return self.vram_fill_zero_asm(matrix_addr, rows)

    # =========================================================================
    # Tile-row ISA helpers (address-based)
    # =========================================================================

    def _arith_progression(self, values: List[int]) -> Optional[Tuple[int, int, int]]:
        """Return (start, count, step) if values form an arithmetic progression."""
        if not values:
            return None
        if len(values) == 1:
            return (values[0], 1, 0)
        step = values[1] - values[0]
        for i in range(2, len(values)):
            if values[i] - values[i - 1] != step:
                return None
        if step == 0:
            return None  # Constant sequence (step=0, count>1) would cause infinite HW loop
        return (values[0], len(values), step)

    def tile_row_max_asm(self, source_vram_addr: int, row_map: List[Tuple[int, int]]) -> str:
        gp = self.register_allocator.allocate_gp(3)
        gp_src, gp_dst, gp_loop = gp
        lines = [f"; Tile Row Max from VRAM[{source_vram_addr}]"]
        rows = [r for r, _ in row_map]
        fp_addrs = [a for _, a in row_map]
        row_prog = self._arith_progression(rows)
        fp_prog = self._arith_progression(fp_addrs)
        if row_prog is not None and fp_prog is not None:
            row_start, row_count, row_step = row_prog
            fp_start, _, fp_step = fp_prog
            lines.append(f"S_ADDI_INT gp{gp_src}, gp0, {source_vram_addr + row_start * self.mlen}")
            lines.append(f"S_ADDI_INT gp{gp_dst}, gp0, {fp_start}")
            lines.append(f"C_LOOP_START gp{gp_loop}, {row_count}")
            lines.append(f"V_RED_MAX f1, gp{gp_src}, 0")
            lines.append(f"S_ST_FP f1, gp{gp_dst}, 0")
            lines.append(f"S_ADDI_INT gp{gp_src}, gp{gp_src}, {row_step * self.mlen}")
            lines.append(f"S_ADDI_INT gp{gp_dst}, gp{gp_dst}, {fp_step}")
            lines.append(f"C_LOOP_END gp{gp_loop}")
        else:
            for row_idx, fpram_addr in row_map:
                row_addr = source_vram_addr + row_idx * self.mlen
                lines.append(f"S_ADDI_INT gp{gp_src}, gp0, {row_addr}")
                lines.append(f"V_RED_MAX f1, gp{gp_src}, 0")
                lines.append(f"S_ADDI_INT gp{gp_dst}, gp0, {fpram_addr}")
                lines.append(f"S_ST_FP f1, gp{gp_dst}, 0")
        self.register_allocator.free_gp(gp)
        isa_code = "\n".join(lines) + "\n"
        self.generated_code += isa_code
        return isa_code

    def tile_row_sum_asm(self, source_vram_addr: int, row_map: List[Tuple[int, int]]) -> str:
        gp = self.register_allocator.allocate_gp(3)
        gp_src, gp_dst, gp_loop = gp
        lines = [f"; Tile Row Sum from VRAM[{source_vram_addr}]"]
        rows = [r for r, _ in row_map]
        fp_addrs = [a for _, a in row_map]
        row_prog = self._arith_progression(rows)
        fp_prog = self._arith_progression(fp_addrs)
        if row_prog is not None and fp_prog is not None:
            row_start, row_count, row_step = row_prog
            fp_start, _, fp_step = fp_prog
            lines.append(f"S_ADDI_INT gp{gp_src}, gp0, {source_vram_addr + row_start * self.mlen}")
            lines.append(f"S_ADDI_INT gp{gp_dst}, gp0, {fp_start}")
            lines.append(f"C_LOOP_START gp{gp_loop}, {row_count}")
            lines.append("S_ADD_FP f1, f0, f0")
            lines.append(f"V_RED_SUM f1, gp{gp_src}, 0, 0")
            lines.append(f"S_ST_FP f1, gp{gp_dst}, 0")
            lines.append(f"S_ADDI_INT gp{gp_src}, gp{gp_src}, {row_step * self.mlen}")
            lines.append(f"S_ADDI_INT gp{gp_dst}, gp{gp_dst}, {fp_step}")
            lines.append(f"C_LOOP_END gp{gp_loop}")
        else:
            for row_idx, fpram_addr in row_map:
                row_addr = source_vram_addr + row_idx * self.mlen
                lines.append(f"S_ADDI_INT gp{gp_src}, gp0, {row_addr}")
                lines.append("S_ADD_FP f1, f0, f0")
                lines.append(f"V_RED_SUM f1, gp{gp_src}, 0, 0")
                lines.append(f"S_ADDI_INT gp{gp_dst}, gp0, {fpram_addr}")
                lines.append(f"S_ST_FP f1, gp{gp_dst}, 0")
        self.register_allocator.free_gp(gp)
        isa_code = "\n".join(lines) + "\n"
        self.generated_code += isa_code
        return isa_code

    def tile_row_exp_asm(self, vram_addr: int, rows: List[int]) -> str:
        gp = self.register_allocator.allocate_gp(2)
        gp_src, gp_loop = gp
        lines = [f"; Tile Row Exp on VRAM[{vram_addr}]"]
        prog = self._arith_progression(rows)
        if prog is not None:
            row_start, row_count, row_step = prog
            lines.append(f"S_ADDI_INT gp{gp_src}, gp0, {vram_addr + row_start * self.mlen}")
            lines.append(f"C_LOOP_START gp{gp_loop}, {row_count}")
            lines.append(f"V_EXP_V gp{gp_src}, gp{gp_src}, 0")
            lines.append(f"S_ADDI_INT gp{gp_src}, gp{gp_src}, {row_step * self.mlen}")
            lines.append(f"C_LOOP_END gp{gp_loop}")
        else:
            for row_idx in rows:
                row_addr = vram_addr + row_idx * self.mlen
                lines.append(f"S_ADDI_INT gp{gp_src}, gp0, {row_addr}")
                lines.append(f"V_EXP_V gp{gp_src}, gp{gp_src}, 0")
        self.register_allocator.free_gp(gp)
        isa_code = "\n".join(lines) + "\n"
        self.generated_code += isa_code
        return isa_code

    def tile_row_reci_asm(self, vram_addr: int, rows: List[int]) -> str:
        gp = self.register_allocator.allocate_gp(2)
        gp_src, gp_loop = gp
        lines = [f"; Tile Row Reciprocal on VRAM[{vram_addr}]"]
        prog = self._arith_progression(rows)
        if prog is not None:
            row_start, row_count, row_step = prog
            lines.append(f"S_ADDI_INT gp{gp_src}, gp0, {vram_addr + row_start * self.mlen}")
            lines.append(f"C_LOOP_START gp{gp_loop}, {row_count}")
            lines.append(f"V_RECI_V gp{gp_src}, gp{gp_src}, 0")
            lines.append(f"S_ADDI_INT gp{gp_src}, gp{gp_src}, {row_step * self.mlen}")
            lines.append(f"C_LOOP_END gp{gp_loop}")
        else:
            for row_idx in rows:
                row_addr = vram_addr + row_idx * self.mlen
                lines.append(f"S_ADDI_INT gp{gp_src}, gp0, {row_addr}")
                lines.append(f"V_RECI_V gp{gp_src}, gp{gp_src}, 0")
        self.register_allocator.free_gp(gp)
        isa_code = "\n".join(lines) + "\n"
        self.generated_code += isa_code
        return isa_code

    def tile_row_sub_fp_asm(self, vram_addr: int, row_map: List[Tuple[int, int]]) -> str:
        gp = self.register_allocator.allocate_gp(3)
        gp_src, gp_fp, gp_loop = gp
        lines = [f"; Tile Row Sub FP on VRAM[{vram_addr}]"]
        rows = [r for r, _ in row_map]
        fp_addrs = [a for _, a in row_map]
        row_prog = self._arith_progression(rows)
        fp_prog = self._arith_progression(fp_addrs)
        if row_prog is not None and fp_prog is not None:
            row_start, row_count, row_step = row_prog
            fp_start, _, fp_step = fp_prog
            lines.append(f"S_ADDI_INT gp{gp_src}, gp0, {vram_addr + row_start * self.mlen}")
            lines.append(f"S_ADDI_INT gp{gp_fp}, gp0, {fp_start}")
            lines.append(f"C_LOOP_START gp{gp_loop}, {row_count}")
            lines.append(f"S_LD_FP f1, gp{gp_fp}, 0")
            lines.append(f"V_SUB_VF gp{gp_src}, gp{gp_src}, f1, 0, 0")
            lines.append(f"S_ADDI_INT gp{gp_src}, gp{gp_src}, {row_step * self.mlen}")
            lines.append(f"S_ADDI_INT gp{gp_fp}, gp{gp_fp}, {fp_step}")
            lines.append(f"C_LOOP_END gp{gp_loop}")
        else:
            for row_idx, fpram_addr in row_map:
                row_addr = vram_addr + row_idx * self.mlen
                lines.append(f"S_ADDI_INT gp{gp_src}, gp0, {row_addr}")
                lines.append(f"S_ADDI_INT gp{gp_fp}, gp0, {fpram_addr}")
                lines.append(f"S_LD_FP f1, gp{gp_fp}, 0")
                lines.append(f"V_SUB_VF gp{gp_src}, gp{gp_src}, f1, 0, 0")
        self.register_allocator.free_gp(gp)
        isa_code = "\n".join(lines) + "\n"
        self.generated_code += isa_code
        return isa_code

    def tile_row_mul_fp_asm(self, vram_addr: int, row_map: List[Tuple[int, int]]) -> str:
        gp = self.register_allocator.allocate_gp(3)
        gp_src, gp_fp, gp_loop = gp
        lines = [f"; Tile Row Mul FP on VRAM[{vram_addr}]"]
        rows = [r for r, _ in row_map]
        fp_addrs = [a for _, a in row_map]
        row_prog = self._arith_progression(rows)
        fp_prog = self._arith_progression(fp_addrs)
        if row_prog is not None and fp_prog is not None:
            row_start, row_count, row_step = row_prog
            fp_start, _, fp_step = fp_prog
            lines.append(f"S_ADDI_INT gp{gp_src}, gp0, {vram_addr + row_start * self.mlen}")
            lines.append(f"S_ADDI_INT gp{gp_fp}, gp0, {fp_start}")
            lines.append(f"C_LOOP_START gp{gp_loop}, {row_count}")
            lines.append(f"S_LD_FP f1, gp{gp_fp}, 0")
            lines.append(f"V_MUL_VF gp{gp_src}, gp{gp_src}, f1, 0")
            lines.append(f"S_ADDI_INT gp{gp_src}, gp{gp_src}, {row_step * self.mlen}")
            lines.append(f"S_ADDI_INT gp{gp_fp}, gp{gp_fp}, {fp_step}")
            lines.append(f"C_LOOP_END gp{gp_loop}")
        else:
            for row_idx, fpram_addr in row_map:
                row_addr = vram_addr + row_idx * self.mlen
                lines.append(f"S_ADDI_INT gp{gp_src}, gp0, {row_addr}")
                lines.append(f"S_ADDI_INT gp{gp_fp}, gp0, {fpram_addr}")
                lines.append(f"S_LD_FP f1, gp{gp_fp}, 0")
                lines.append(f"V_MUL_VF gp{gp_src}, gp{gp_src}, f1, 0")
        self.register_allocator.free_gp(gp)
        isa_code = "\n".join(lines) + "\n"
        self.generated_code += isa_code
        return isa_code

    def tile_row_add_fp_asm(self, vram_addr: int, row_map: List[Tuple[int, int]]) -> str:
        gp = self.register_allocator.allocate_gp(3)
        gp_src, gp_fp, gp_loop = gp
        lines = [f"; Tile Row Add FP on VRAM[{vram_addr}]"]
        rows = [r for r, _ in row_map]
        fp_addrs = [a for _, a in row_map]
        row_prog = self._arith_progression(rows)
        fp_prog = self._arith_progression(fp_addrs)
        if row_prog is not None and fp_prog is not None:
            row_start, row_count, row_step = row_prog
            fp_start, _, fp_step = fp_prog
            lines.append(f"S_ADDI_INT gp{gp_src}, gp0, {vram_addr + row_start * self.mlen}")
            lines.append(f"S_ADDI_INT gp{gp_fp}, gp0, {fp_start}")
            lines.append(f"C_LOOP_START gp{gp_loop}, {row_count}")
            lines.append(f"S_LD_FP f1, gp{gp_fp}, 0")
            lines.append(f"V_ADD_VF gp{gp_src}, gp{gp_src}, f1, 0")
            lines.append(f"S_ADDI_INT gp{gp_src}, gp{gp_src}, {row_step * self.mlen}")
            lines.append(f"S_ADDI_INT gp{gp_fp}, gp{gp_fp}, {fp_step}")
            lines.append(f"C_LOOP_END gp{gp_loop}")
        else:
            for row_idx, fpram_addr in row_map:
                row_addr = vram_addr + row_idx * self.mlen
                lines.append(f"S_ADDI_INT gp{gp_src}, gp0, {row_addr}")
                lines.append(f"S_ADDI_INT gp{gp_fp}, gp0, {fpram_addr}")
                lines.append(f"S_LD_FP f1, gp{gp_fp}, 0")
                lines.append(f"V_ADD_VF gp{gp_src}, gp{gp_src}, f1, 0")
        self.register_allocator.free_gp(gp)
        isa_code = "\n".join(lines) + "\n"
        self.generated_code += isa_code
        return isa_code

    def tile_row_add_asm(self, dst_addr: int, src_addr: int, rows: List[int]) -> str:
        gp = self.register_allocator.allocate_gp(3)
        gp_dst, gp_src, gp_loop = gp
        lines = [f"; Tile Row Add: VRAM[{dst_addr}] += VRAM[{src_addr}]"]
        prog = self._arith_progression(rows)
        if prog is not None:
            row_start, row_count, row_step = prog
            lines.append(f"S_ADDI_INT gp{gp_dst}, gp0, {dst_addr + row_start * self.mlen}")
            lines.append(f"S_ADDI_INT gp{gp_src}, gp0, {src_addr + row_start * self.mlen}")
            lines.append(f"C_LOOP_START gp{gp_loop}, {row_count}")
            lines.append(f"V_ADD_VV gp{gp_dst}, gp{gp_dst}, gp{gp_src}, 0")
            lines.append(f"S_ADDI_INT gp{gp_dst}, gp{gp_dst}, {row_step * self.mlen}")
            lines.append(f"S_ADDI_INT gp{gp_src}, gp{gp_src}, {row_step * self.mlen}")
            lines.append(f"C_LOOP_END gp{gp_loop}")
        else:
            for row_idx in rows:
                d = dst_addr + row_idx * self.mlen
                s = src_addr + row_idx * self.mlen
                lines.append(f"S_ADDI_INT gp{gp_dst}, gp0, {d}")
                lines.append(f"S_ADDI_INT gp{gp_src}, gp0, {s}")
                lines.append(f"V_ADD_VV gp{gp_dst}, gp{gp_dst}, gp{gp_src}, 0")
        self.register_allocator.free_gp(gp)
        isa_code = "\n".join(lines) + "\n"
        self.generated_code += isa_code
        return isa_code

    def tile_row_sub_asm(self, dst_addr: int, src_addr: int, rows: List[int]) -> str:
        gp = self.register_allocator.allocate_gp(3)
        gp_dst, gp_src, gp_loop = gp
        lines = [f"; Tile Row Sub: VRAM[{dst_addr}] -= VRAM[{src_addr}]"]
        prog = self._arith_progression(rows)
        if prog is not None:
            row_start, row_count, row_step = prog
            lines.append(f"S_ADDI_INT gp{gp_dst}, gp0, {dst_addr + row_start * self.mlen}")
            lines.append(f"S_ADDI_INT gp{gp_src}, gp0, {src_addr + row_start * self.mlen}")
            lines.append(f"C_LOOP_START gp{gp_loop}, {row_count}")
            lines.append(f"V_SUB_VV gp{gp_dst}, gp{gp_dst}, gp{gp_src}, 0")
            lines.append(f"S_ADDI_INT gp{gp_dst}, gp{gp_dst}, {row_step * self.mlen}")
            lines.append(f"S_ADDI_INT gp{gp_src}, gp{gp_src}, {row_step * self.mlen}")
            lines.append(f"C_LOOP_END gp{gp_loop}")
        else:
            for row_idx in rows:
                d = dst_addr + row_idx * self.mlen
                s = src_addr + row_idx * self.mlen
                lines.append(f"S_ADDI_INT gp{gp_dst}, gp0, {d}")
                lines.append(f"S_ADDI_INT gp{gp_src}, gp0, {s}")
                lines.append(f"V_SUB_VV gp{gp_dst}, gp{gp_dst}, gp{gp_src}, 0")
        self.register_allocator.free_gp(gp)
        isa_code = "\n".join(lines) + "\n"
        self.generated_code += isa_code
        return isa_code

    def tile_row_mul_asm(self, dst_addr: int, src_addr: int, rows: List[int]) -> str:
        gp = self.register_allocator.allocate_gp(3)
        gp_dst, gp_src, gp_loop = gp
        lines = [f"; Tile Row Mul: VRAM[{dst_addr}] *= VRAM[{src_addr}]"]
        prog = self._arith_progression(rows)
        if prog is not None:
            row_start, row_count, row_step = prog
            lines.append(f"S_ADDI_INT gp{gp_dst}, gp0, {dst_addr + row_start * self.mlen}")
            lines.append(f"S_ADDI_INT gp{gp_src}, gp0, {src_addr + row_start * self.mlen}")
            lines.append(f"C_LOOP_START gp{gp_loop}, {row_count}")
            lines.append(f"V_MUL_VV gp{gp_dst}, gp{gp_dst}, gp{gp_src}, 0")
            lines.append(f"S_ADDI_INT gp{gp_dst}, gp{gp_dst}, {row_step * self.mlen}")
            lines.append(f"S_ADDI_INT gp{gp_src}, gp{gp_src}, {row_step * self.mlen}")
            lines.append(f"C_LOOP_END gp{gp_loop}")
        else:
            for row_idx in rows:
                d = dst_addr + row_idx * self.mlen
                s = src_addr + row_idx * self.mlen
                lines.append(f"S_ADDI_INT gp{gp_dst}, gp0, {d}")
                lines.append(f"S_ADDI_INT gp{gp_src}, gp0, {s}")
                lines.append(f"V_MUL_VV gp{gp_dst}, gp{gp_dst}, gp{gp_src}, 0")
        self.register_allocator.free_gp(gp)
        isa_code = "\n".join(lines) + "\n"
        self.generated_code += isa_code
        return isa_code

    def tile_row_mul_fp_broadcast_asm(self, vram_addr: int, fpram_scalar_addr: int, rows: List[int]) -> str:
        row_map = [(r, fpram_scalar_addr) for r in rows]
        return self.tile_row_mul_fp_asm(vram_addr, row_map)

    def vram_fill_zero_asm(
        self,
        vram_addr: int,
        rows: List[int],
    ) -> str:
        """
        VRAM Fill Zero: fill specified rows with 0.

        For each row_idx in rows:
            VRAM[row] = 0
        """
        if not rows:
            isa_code = f"; === VRAM Fill Zero: VRAM[{vram_addr}] rows [] = 0 ===\n"
            self.generated_code += isa_code
            return isa_code

        gp_regs = self.register_allocator.allocate_gp(2)
        gp_dst, gp_loop = gp_regs

        lines = []
        lines.append(f"; === VRAM Fill Zero: VRAM[{vram_addr}] rows {rows} = 0 ===")
        prog = self._arith_progression(rows)

        if prog is not None:
            row_start, row_count, row_step = prog
            lines.append(f"S_ADDI_INT gp{gp_dst}, gp0, {vram_addr + row_start * self.mlen}")
            lines.append(f"C_LOOP_START gp{gp_loop}, {row_count}")
            lines.append(f"V_MUL_VF gp{gp_dst}, gp{gp_dst}, f0, 0")
            lines.append(f"S_ADDI_INT gp{gp_dst}, gp{gp_dst}, {row_step * self.mlen}")
            lines.append(f"C_LOOP_END gp{gp_loop}")
        else:
            for row_idx in rows:
                row_addr = vram_addr + row_idx * self.mlen
                lines.append(f"S_ADDI_INT gp{gp_dst}, gp0, {row_addr}")
                lines.append(f"V_MUL_VF gp{gp_dst}, gp{gp_dst}, f0, 0")

        self.register_allocator.free_gp(gp_regs)

        isa_code = "\n".join(lines) + "\n"
        self.generated_code += isa_code
        return isa_code

    def reset_mram(self) -> str:
        """
        Reset MRAM allocator, free all allocated space
        Used in scenarios where sub-blocks need to be reloaded within a for loop
        """
        self.sub_matrix_manager.mram_allocator.reset()
        self.sub_matrix_manager.loaded_sub_blocks.clear()
        
        isa_code = "; === Reset MRAM ===\n"
        self.generated_code += isa_code
        return isa_code
    
    def load_sub_matrix_row(
        self,
        name: str,
        row_idx: int,
        mram_start_addr: Optional[int] = None,
    ) -> str:
        """
        Load entire row sub-blocks from HBM to MRAM: matrix[row_idx][:]

        Args:
            name: 矩阵名称
            row_idx: 子块行索引
            mram_start_addr: MRAM 起始地址（如果 None，自动分配）

        Returns:
            生成的 ISA 代码
        """
        layout = self.sub_matrix_manager.get_hbm_layout(name)
        num_col_blocks = layout.num_col_blocks
        block_size = self.mlen * self.mlen

        # Automatically allocate MRAM address
        if mram_start_addr is None:
            total_size = num_col_blocks * block_size
            mram_start_addr = self.sub_matrix_manager.mram_allocator.allocate(
                f"{name}[{row_idx}][:]", total_size
            )

        # Allocate registers
        gp_regs = self.register_allocator.allocate_gp(4)
        gp_for_addr = self.register_allocator.allocate_gp(2)
        addr_reg = self.register_allocator.allocate_addr(1)[0]

        # Set HBM address register
        isa_code = preload_addr_reg_asm(
            addr_reg_to_set=[addr_reg],
            available_registers=gp_for_addr,
            addr_reg_val=[layout.hbm_base_addr]
        )

        # Generate load code
        isa_code += self.sub_matrix_manager.load_row_sub_matrices_asm(
            name=name,
            row_idx=row_idx,
            mram_start_addr=mram_start_addr,
            hbm_addr_reg=addr_reg,
            gp_regs=gp_regs
        )

        # 释放寄存器
        self.register_allocator.free_gp(gp_regs)
        self.register_allocator.free_gp(gp_for_addr)
        self.register_allocator.free_addr([addr_reg])

        self.generated_code += isa_code
        return isa_code
    
    def load_sub_matrix_col(
        self,
        name: str,
        col_idx: int,
        mram_start_addr: Optional[int] = None,
        k_block_start: int = 0,
        k_block_count: int = None,
    ) -> str:
        """
        Load entire column sub-blocks from HBM to MRAM: matrix[:][col_idx]

        Used for sub_projection: A @ W[:, col_idx*mlen:(col_idx+1)*mlen]

        Args:
            name: 矩阵名称
            col_idx: 子块列索引
            mram_start_addr: MRAM 起始地址（如果 None，自动分配）

        Returns:
            生成的 ISA 代码
        """
        layout = self.sub_matrix_manager.get_hbm_layout(name)
        num_row_blocks = layout.num_row_blocks
        block_size = self.mlen * self.mlen

        # Automatically allocate MRAM address
        if mram_start_addr is None:
            effective_count = k_block_count if k_block_count is not None else num_row_blocks
            total_size = effective_count * block_size
            mram_start_addr = self.sub_matrix_manager.mram_allocator.allocate(
                f"{name}[:][{col_idx}]", total_size
            )

        # Allocate registers
        gp_regs = self.register_allocator.allocate_gp(3)
        gp_for_addr = self.register_allocator.allocate_gp(2)
        addr_reg = self.register_allocator.allocate_addr(1)[0]

        # Set HBM address register
        isa_code = preload_addr_reg_asm(
            addr_reg_to_set=[addr_reg],
            available_registers=gp_for_addr,
            addr_reg_val=[layout.hbm_base_addr]
        )

        # Generate load code（加载列子块）
        isa_code += self.sub_matrix_manager.load_col_sub_matrices_asm(
            name=name,
            col_idx=col_idx,
            mram_start_addr=mram_start_addr,
            hbm_addr_reg=addr_reg,
            gp_regs=gp_regs,
            k_block_start=k_block_start,
            k_block_count=k_block_count,
        )

        # 释放寄存器
        self.register_allocator.free_gp(gp_regs)
        self.register_allocator.free_gp(gp_for_addr)
        self.register_allocator.free_addr([addr_reg])

        self.generated_code += isa_code
        return isa_code
    
   
    def allocate_vram_matrix(
        self,
        name: str,
        rows: int,
        cols: int,
        strict: bool = True,
    ) -> int:
        """
        Allocate a large VRAM matrix to store combined results of multiple sub-blocks
        
        Args:
            name: 矩阵名称
            rows: 行数
            cols: 列数
            
        Returns:
            VRAM 基地址
        """
        size = rows * cols
        vram_addr = self.sub_matrix_manager.vram_allocator.allocate(size, name=name)
        
        self.sub_matrix_manager.add_vram_object(
            name=name,
            shape=(rows, cols),
            vram_addr=vram_addr,
            dtype="fp32",
            kind="VRAMMatrix",
            allocate_if_none=False,
            strict=strict,
        )
        
        isa_code = f"; Allocate VRAM Matrix {name}: ({rows}, {cols}) at VRAM[{vram_addr}]\n"
        self.generated_code += isa_code
        
        return vram_addr

    def _ensure_vram_matrix_layout(self, matrix_name: str):
        """Ensure a VRAM-resident tensor has a block layout in SubMatrixManager."""
        if matrix_name not in self.sub_matrix_manager:
            raise KeyError(f"Matrix '{matrix_name}' not found in symbol table")

        info = self.sub_matrix_manager[matrix_name]
        if info.vram_addr is None:
            raise ValueError(f"Matrix '{matrix_name}' has no VRAM address")

        try:
            self.sub_matrix_manager.get_vram_layout(matrix_name)
        except KeyError:
            self.sub_matrix_manager.register_vram_matrix(
                name=matrix_name,
                shape=info.shape,
                vram_base_addr=info.vram_addr,
            )

    def vram_block_add_to(
        self,
        src1_matrix: str,
        src1_row_idx: int,
        src1_col_idx: int,
        src2_matrix: str,
        src2_row_idx: int,
        src2_col_idx: int,
        target_matrix: str,
        target_row_idx: int,
        target_col_idx: int,
    ) -> str:
        """
        mlen x mlen block add:
            target[rt][ct] = src1[r1][c1] + src2[r2][c2]

        Source/target may be the same matrix (supports in-place overwrite).
        """
        self._ensure_vram_matrix_layout(src1_matrix)
        self._ensure_vram_matrix_layout(src2_matrix)
        self._ensure_vram_matrix_layout(target_matrix)

        gp_regs = self.register_allocator.allocate_gp(4)
        isa_code = self.sub_matrix_manager.vram_block_add_asm(
            src1_name=src1_matrix,
            src1_row_idx=src1_row_idx,
            src1_col_idx=src1_col_idx,
            src2_name=src2_matrix,
            src2_row_idx=src2_row_idx,
            src2_col_idx=src2_col_idx,
            target_name=target_matrix,
            target_row_idx=target_row_idx,
            target_col_idx=target_col_idx,
            gp_regs=gp_regs,
        )
        self.register_allocator.free_gp(gp_regs)

        self.generated_code += isa_code
        return isa_code
    
    def vram_matrix_add(
        self,
        dst_matrix: str,
        src_matrix: str,
        dst_row_offset: int = 0,
        src_row_offset: int = 0,
        num_rows: int = None,
    ) -> str:
        """
        General VRAM Matrix Addition: dst[row_offset:] += src
        
        Args:
            dst_matrix: 目标矩阵名称
            src_matrix: 源矩阵名称
            dst_row_offset: Target matrix row offset (logical row, not VRAM address)
            src_row_offset: Source matrix row offset
            num_rows: Number of rows to process (defaults to source matrix row count)
        """
        dst_info = self.sub_matrix_manager[dst_matrix]
        src_info = self.sub_matrix_manager[src_matrix]

        # Block-add path depends on SubMatrixManager VRAM layouts.
        self._ensure_vram_matrix_layout(dst_matrix)
        self._ensure_vram_matrix_layout(src_matrix)
        
        dst_addr = dst_info.vram_addr
        src_addr = src_info.vram_addr
        
        dst_rows, dst_cols = dst_info.shape
        src_rows, src_cols = src_info.shape
        
        if num_rows is None:
            num_rows = src_rows
        
        # Ensure column count matches
        assert dst_cols == src_cols, f"Column mismatch: dst={dst_cols}, src={src_cols}"
        assert dst_row_offset + num_rows <= dst_rows, (
            f"dst row range out of bounds: offset={dst_row_offset}, num_rows={num_rows}, dst_rows={dst_rows}"
        )
        assert src_row_offset + num_rows <= src_rows, (
            f"src row range out of bounds: offset={src_row_offset}, num_rows={num_rows}, src_rows={src_rows}"
        )
        lines = []
        lines.append(
            f"; === VRAM Matrix Add: "
            f"{dst_matrix}[{dst_row_offset}:{dst_row_offset + num_rows}] += "
            f"{src_matrix}[{src_row_offset}:{src_row_offset + num_rows}] ==="
        )
        lines.append(f"; dst shape: {dst_info.shape}, src shape: {src_info.shape}")

        # Prefer block add path so we can reuse the compact C_LOOP-based add kernel.
        block_aligned = (
            dst_cols % self.mlen == 0 and
            src_cols % self.mlen == 0 and
            dst_row_offset % self.mlen == 0 and
            src_row_offset % self.mlen == 0 and
            num_rows % self.mlen == 0
        )

        if block_aligned:
            num_row_blocks = num_rows // self.mlen
            num_col_blocks = dst_cols // self.mlen
            dst_row_block_base = dst_row_offset // self.mlen
            src_row_block_base = src_row_offset // self.mlen
            lines.append(f"; block add path: row_blocks={num_row_blocks}, col_blocks={num_col_blocks}")

            for row_block in range(num_row_blocks):
                for col_block in range(num_col_blocks):
                    gp_regs = self.register_allocator.allocate_gp(4)
                    lines.append(
                        self.sub_matrix_manager.vram_block_add_asm(
                            src1_name=dst_matrix,
                            src1_row_idx=dst_row_block_base + row_block,
                            src1_col_idx=col_block,
                            src2_name=src_matrix,
                            src2_row_idx=src_row_block_base + row_block,
                            src2_col_idx=col_block,
                            target_name=dst_matrix,
                            target_row_idx=dst_row_block_base + row_block,
                            target_col_idx=col_block,
                            gp_regs=gp_regs,
                        ).rstrip("\n")
                    )
                    self.register_allocator.free_gp(gp_regs)
        else:
            # Fallback for non-mlen-aligned ranges.
            gp_regs = self.register_allocator.allocate_gp(2)
            gp_dst = gp_regs[0]
            gp_src = gp_regs[1]
            num_col_blocks = dst_cols // self.mlen
            lines.append(f"; fallback row-wise path: num_rows={num_rows}, num_col_blocks={num_col_blocks}")

            for row in range(num_rows):
                dst_actual_row = dst_row_offset + row
                src_actual_row = src_row_offset + row

                for col_block in range(num_col_blocks):
                    dst_block_addr = dst_addr + col_block * dst_rows * self.mlen + dst_actual_row * self.mlen
                    src_block_addr = src_addr + col_block * src_rows * self.mlen + src_actual_row * self.mlen

                    lines.append(f"S_ADDI_INT gp{gp_dst}, gp0, {dst_block_addr}")
                    lines.append(f"S_ADDI_INT gp{gp_src}, gp0, {src_block_addr}")
                    lines.append(f"V_ADD_VV gp{gp_dst}, gp{gp_dst}, gp{gp_src}, 0")
            self.register_allocator.free_gp(gp_regs)

        isa_code = "\n".join(lines) + "\n"
        self.generated_code += isa_code
        return isa_code
    
    def vram_sub_projection_to(
        self,
        vram_mat_name: str,
        vram_row_idx: int,
        mram_mat_name: str,
        mram_col_idx: int,
        target_matrix: str,
        target_row_idx: int,
        target_col_idx: int,
        k_block_start: int = 0,
        k_block_count: int = None,
    ) -> str:
        """
        VRAM Sub-block Multiplication, result written to specified sub-block position of target matrix
        
        Computes: target[target_row_idx][target_col_idx] = VRAM_A[vram_row_idx][:] @ MRAM_W[:][mram_col_idx]
        
        Args:
            vram_mat_name: VRAM 子矩阵名称
            vram_row_idx: VRAM 矩阵行索引
            mram_mat_name: MRAM 子矩阵名称
            mram_col_idx: MRAM 矩阵列索引
            target_matrix: 目标矩阵名称（必须已通过 allocate_vram_matrix 分配）
            target_row_idx: 目标矩阵的行子块索引
            target_col_idx: 目标矩阵的列子块索引
            
        Returns:
            生成的 ISA 代码
        """
        # Get target matrix information
        if target_matrix not in self.sub_matrix_manager:
            raise KeyError(f"Target matrix '{target_matrix}' not found. Use allocate_vram_matrix first.")
        
        target_info = self.sub_matrix_manager[target_matrix]
        target_rows, target_cols = target_info.shape
        target_base_addr = target_info.vram_addr
        
        # Calculate VRAM address of target sub-block
        # VRAM storage format: [batch, mlen, hidden/mlen] column-block major
        # 子块 (r, c) 的地址 = base + c * rows * mlen + r * mlen * mlen
        result_vram_addr = target_base_addr + target_col_idx * target_rows * self.mlen + target_row_idx * self.mlen * self.mlen
        
        # 分配寄存器
        gp_regs = self.register_allocator.allocate_gp(9)
        
        # Generate ISA
        isa_code = f"; VRAM Sub Projection To: {vram_mat_name}[{vram_row_idx}][:] @ {mram_mat_name}[:][{mram_col_idx}] -> {target_matrix}[{target_row_idx}][{target_col_idx}]\n"
        isa_code += f"; Target VRAM addr: {result_vram_addr} (base={target_base_addr}, offset=col*{target_rows}*{self.mlen} + row*{self.mlen}*{self.mlen})\n"
        isa_code += self.sub_matrix_manager.vram_sub_projection_asm(
            vram_mat_name=vram_mat_name,
            vram_row_idx=vram_row_idx,
            mram_mat_name=mram_mat_name,
            mram_col_idx=mram_col_idx,
            result_vram_addr=result_vram_addr,
            gp_regs=gp_regs,
            k_block_start=k_block_start,
            k_block_count=k_block_count,
        )
        
        # 释放寄存器
        self.register_allocator.free_gp(gp_regs)
        
        self.generated_code += isa_code
        return isa_code
    
    def vram_sub_projection_T_to(
        self,
        vram_mat_name: str,
        vram_row_idx: int,
        mram_mat_name: str,
        mram_row_idx: int,
        target_matrix: str,
        target_row_idx: int,
        target_col_idx: int,
    ) -> str:
        """
        VRAM Sub-block Transposed Multiplication, result written to specified sub-block position of target matrix
        
        Computes: target[target_row_idx][target_col_idx] = VRAM_A[vram_row_idx][:] @ MRAM_W[mram_row_idx][:]^T
        
        Used in Flash Attention for S = Q @ K^T
        - Q[i][:]: (mlen, hidden_size) 行子块
        - K[j][:]: (mlen, hidden_size) 行子块，转置后 (hidden_size, mlen)
        - S[i][j]: (mlen, mlen)
        
        Args:
            vram_mat_name: VRAM 子矩阵名称
            vram_row_idx: VRAM 矩阵行索引
            mram_mat_name: MRAM 子矩阵名称
            mram_row_idx: MRAM 矩阵行索引
            target_matrix: 目标矩阵名称（必须已通过 allocate_vram_matrix 分配）
            target_row_idx: 目标矩阵的行子块索引
            target_col_idx: 目标矩阵的列子块索引
            
        Returns:
            生成的 ISA 代码
        """
        # Get target matrix information
        if target_matrix not in self.sub_matrix_manager:
            raise KeyError(f"Target matrix '{target_matrix}' not found. Use allocate_vram_matrix first.")
        
        target_info = self.sub_matrix_manager[target_matrix]
        target_rows, target_cols = target_info.shape
        target_base_addr = target_info.vram_addr
        
        # Calculate VRAM address of target sub-block
        # VRAM storage format: [batch, mlen, hidden/mlen] column-block major
        # 子块 (r, c) 的地址 = base + c * rows * mlen + r * mlen * mlen
        result_vram_addr = target_base_addr + target_col_idx * target_rows * self.mlen + target_row_idx * self.mlen * self.mlen
        
        # 分配寄存器
        gp_regs = self.register_allocator.allocate_gp(9)
        
        # Generate ISA
        isa_code = f"; VRAM Sub Projection T To: {vram_mat_name}[{vram_row_idx}][:] @ {mram_mat_name}[{mram_row_idx}][:]^T -> {target_matrix}[{target_row_idx}][{target_col_idx}]\n"
        isa_code += f"; Target VRAM addr: {result_vram_addr} (base={target_base_addr}, offset=col*{target_rows}*{self.mlen} + row*{self.mlen}*{self.mlen})\n"
        isa_code += self.sub_matrix_manager.vram_sub_projection_T_asm(
            vram_mat_name=vram_mat_name,
            vram_row_idx=vram_row_idx,
            mram_mat_name=mram_mat_name,
            mram_row_idx=mram_row_idx,
            result_vram_addr=result_vram_addr,
            gp_regs=gp_regs
        )
        
        # 释放寄存器
        self.register_allocator.free_gp(gp_regs)
        
        self.generated_code += isa_code
        return isa_code
    
    # =========================================================================
    # Expanded Flash Attention Operations
    # =========================================================================
    
    def init_online_softmax(
        self,
        q_idx: int,
        o_matrix: str,
        seq_len: int,
        head_dim: int,
    ) -> str:
        """
        Initialize Online Softmax state
        
        Initialization:
        - m_old = -inf (FP SRAM)
        - l = 0 (FP SRAM)
        - O_row = 0 (VRAM)
        
        Args:
            q_idx: 当前 Q 块索引
            o_matrix: O 矩阵名称
            seq_len: 序列长度
            head_dim: head dimension
        """
        # FP SRAM Layout (fixed)
        fp_sram_start = self._ONLINE_SOFTMAX_FPSRAM_BASE
        m_old_addr = fp_sram_start
        l_addr = fp_sram_start + 2 * self.mlen  # 跳过 m_res
        
        o_info = self.sub_matrix_manager[o_matrix]
        o_vram_addr = o_info.vram_addr
        row_offset = q_idx * self.mlen
        
        isa_code = f"; === Init Online Softmax for Q block {q_idx} ===\n"
        
        # 初始化 m_old = -inf
        isa_code += self._reset_fpsram_asm(m_old_addr, self.mlen, 2)  # 2 = -inf 地址
        
        # 初始化 l = 0
        isa_code += self._reset_fpsram_asm(l_addr, self.mlen, 0)  # 0 = 0 地址
        
        # 初始化 O_row = 0
        isa_code += self._reset_vram_asm(
            start_address=o_vram_addr,
            rows=self.mlen,
            cols=head_dim,
            total_rows=seq_len,
            mlen=self.mlen,
            row_offset=row_offset
        )
        
        self.generated_code += isa_code
        return isa_code
    
    def online_softmax_block(
        self,
        s_block_matrix: str,
        scale: float,
    ) -> str:
        """
        Perform Online Softmax on single S block
        
        输入：S_block (mlen x mlen) 在 VRAM
        输出：P (mlen x mlen) in-place 在 VRAM
        Updates: m_old, m_res, l in FP SRAM
        
        Args:
            s_block_matrix: S 块矩阵名称
            scale: 缩放因子 (1/sqrt(d))
        """
        s_info = self.sub_matrix_manager[s_block_matrix]
        s_address = s_info.vram_addr
        
        # FP SRAM 布局
        fp_sram_start = self._ONLINE_SOFTMAX_FPSRAM_BASE
        m_start_address = fp_sram_start
        
        isa_code = f"; === Online Softmax Block {s_block_matrix} ===\n"
        isa_code += self._online_softmax_asm(
            mlen=self.mlen,
            s_address=s_address,
            m_start_address=m_start_address,
            scale=scale
        )
        
        self.generated_code += isa_code
        return isa_code
    
    def compute_pv(
        self,
        s_block_matrix: str,
        v_sub_matrix: str,
        k_idx: int,
        pv_matrix: str,
        head_dim: int,
    ) -> str:
        """
        Compute PV = P @ V[k_idx]
        
        P is stored in s_block_matrix (result after softmax)
        V is prefetched from HBM
        PV is stored in VRAM
        
        Args:
            s_block_matrix: S/P 块矩阵名称（存储 P）
            v_sub_matrix: V 子矩阵名称
            k_idx: 当前 K/V 块索引
            pv_matrix: PV 临时矩阵名称
            head_dim: head dimension
        """
        # 获取地址
        s_info = self.sub_matrix_manager[s_block_matrix]
        p_address = s_info.vram_addr
        
        pv_info = self.sub_matrix_manager[pv_matrix]
        pv_address = pv_info.vram_addr
        
        # V offset in HBM
        v_layout = self.sub_matrix_manager.get_hbm_layout(v_sub_matrix)
        v_hbm_offset = k_idx * self.mlen * head_dim
        
        isa_code = f"; === Compute PV = P @ V[k_idx={k_idx}] ===\n"
        
        # 分配 HBM 地址寄存器
        addr_regs = self.register_allocator.allocate_addr(1)
        v_hbm_reg = addr_regs[0]
        gp_regs = self.register_allocator.allocate_gp(2)
        
        # 设置 V 的 HBM 地址寄存器
        from compiler.asm_templates import preload_addr_reg_asm
        isa_code += preload_addr_reg_asm(
            addr_reg_to_set=[v_hbm_reg],
            available_registers=gp_regs,
            addr_reg_val=[v_layout.hbm_base_addr]
        )
        
        # PV = P @ V
        isa_code += self._pv_multiply_asm(
            mlen=self.mlen,
            blen=self.blen,
            head_dim=head_dim,
            p_address=p_address,
            v_hbm_offset_reg=v_hbm_reg,
            v_hbm_offset=v_hbm_offset,
            pv_address=pv_address,
        )
        
        # 释放寄存器
        self.register_allocator.free_gp(gp_regs)
        self.register_allocator.free_addr(addr_regs)
        
        self.generated_code += isa_code
        return isa_code
    
    def scale_o_row(
        self,
        o_matrix: str,
        q_idx: int,
        seq_len: int,
        head_dim: int,
    ) -> str:
        """
        Scale current row block of O by m_res: O[q_idx] = O[q_idx] * m_res
        
        Args:
            o_matrix: O 矩阵名称
            q_idx: 当前 Q 块索引
            seq_len: 序列长度
            head_dim: head dimension
        """
        o_info = self.sub_matrix_manager[o_matrix]
        o_address = o_info.vram_addr
        
        # FP SRAM 布局
        fp_sram_start = self._ONLINE_SOFTMAX_FPSRAM_BASE
        m_res_addr = fp_sram_start + self.mlen
        
        row_offset = q_idx * self.mlen
        
        isa_code = f"; === Scale O[q_idx={q_idx}] by m_res ===\n"
        isa_code += self._scale_o_asm(
            mlen=self.mlen,
            head_dim=head_dim,
            seq_len=seq_len,
            m_res_address=m_res_addr,
            o_address=o_address,
            row_offset=row_offset,
        )
        
        self.generated_code += isa_code
        return isa_code
    
    def final_scale_o(
        self,
        q_idx: int,
        o_matrix: str,
        seq_len: int,
        head_dim: int,
    ) -> str:
        """
        Final scaling: O[q_idx] = O[q_idx] / l
        
        Args:
            q_idx: 当前 Q 块索引
            o_matrix: O 矩阵名称
            seq_len: 序列长度
            head_dim: head dimension
        """
        o_info = self.sub_matrix_manager[o_matrix]
        o_address = o_info.vram_addr
        
        # FP SRAM 布局
        fp_sram_start = self._ONLINE_SOFTMAX_FPSRAM_BASE
        l_addr = fp_sram_start + 2 * self.mlen
        
        row_offset = q_idx * self.mlen
        
        isa_code = f"; === Final Scale O for Q block {q_idx} ===\n"
        isa_code += self._final_scaling_asm(
            mlen=self.mlen,
            head_dim=head_dim,
            seq_len=seq_len,
            l_address=l_addr,
            o_address=o_address,
            row_offset=row_offset,
        )
        
        self.generated_code += isa_code
        return isa_code
    


# Example Usage
if __name__ == "__main__":
    compiler = DeveloperCompiler()
    real_data_ratio = (8*8 + 8) / (8 * 8)
    compiler.add_hbm_object("A", hbm_addr=0, shape=(8, 128), real_data_ratio=real_data_ratio)
    compiler.load_batch("A", "A")
    compiler.print_symbol_table()
    print(compiler.get_code())
