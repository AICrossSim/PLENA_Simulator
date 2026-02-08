"""
Auto Compiler Helper: 自动处理 HBM 地址分配、环境搭建和 golden value 计算
用户只需要提供 torch tensors 和伪代码，其他全部自动处理

支持 Sub Matrix 操作：
- Register_SubMatrix(W, weight)  # 注册子矩阵
- Load_SubMatrix(W, 1)  # 加载 W[1][:] 到 MRAM
- C = A @ SubMatrix(W, 1)  # C = A @ W[1][:]
- C = A @T SubMatrix(W, 1)  # C = A @ W[1][:]^T

支持 For Loop 操作：
- for i in range(0, seq_len, mlen):
      # loop body
  endfor

支持 Softmax 相关操作（用于 Flash Attention 展开版本）：
- m = Init(-inf, mlen)  # 初始化为 -inf
- m = RowMax(S)  # 每行最大值
- l = RowSum(S)  # 每行求和
- P = RowSoftmax(S)  # 行 softmax
- S = Scale(S, factor)  # 标量乘法
- m_new = Max(m_old, m_cur)  # 逐元素最大
- r = Exp(x)  # exp
- r = Sub(a, b)  # 减法
- r = Add(a, b)  # 加法
- r = Mul(a, b)  # 乘法
- r = Div(a, b)  # 除法
"""

import torch
import json
import re
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from simple_compiler import SimpleCompiler
from behavioral_simulator.tools.create_sim_env import create_sim_env
from compiler.sim_env_utils import create_mem_for_sim


class AutoCompilerHelper:
    """自动编译器助手：自动处理 HBM 地址、环境搭建和 golden value"""
    
    def __init__(self, real_data_ratio: float = 1.125, mlen: int = 64, blen: int = 4):
        """
        Args:
            real_data_ratio: HBM 数据比例（考虑 MXFP 格式的 scalar）
            mlen: Matrix tile size
            blen: Vector tile size
        """
        self.real_data_ratio = real_data_ratio
        self.mlen = mlen
        self.blen = blen
        self.tensors: Dict[str, torch.Tensor] = {}
        self.tensor_shapes: Dict[str, Tuple[int, int]] = {}
        self.hbm_layout: Dict[str, int] = {}  # tensor_name -> hbm_addr
        self.hbm_sizes: Dict[str, int] = {}  # tensor_name -> hbm_size
        self.stored_tensors: Dict[str, str] = {}  # store_name -> original tensor alias
        
    def add_tensor(self, name: str, tensor: torch.Tensor, is_batch: bool = False):
        """
        添加一个 tensor
        
        Args:
            name: tensor 名称
            tensor: torch.Tensor，可以是 2D (batch, features) 或 2D (h, w)
            is_batch: 是否为 Batch 类型（否则为 Matrix）
        """
        if tensor.dim() != 2:
            raise ValueError(f"Tensor '{name}' must be 2D, got shape {tensor.shape}")
        
        h, w = tensor.shape
        self.tensors[name] = tensor
        self.tensor_shapes[name] = (h, w)
        
        # 计算 HBM size（考虑 real_data_ratio）
        size = h * w
        hbm_size = int(size * self.real_data_ratio)
        self.hbm_sizes[name] = hbm_size
        
        # 自动分配 HBM 地址（按添加顺序连续分配，64字节对齐）
        if not self.hbm_layout:
            self.hbm_layout[name] = 0
        else:
            # 找到最后一个 tensor 的结束地址，并对齐到 64 字节边界
            last_name = max(self.hbm_layout.keys(), key=lambda k: self.hbm_layout[k] + self.hbm_sizes[k])
            last_end = self.hbm_layout[last_name] + self.hbm_sizes[last_name]
            # 向上对齐到 64 字节边界（与 working test 一致）
            aligned_addr = ((last_end + 63) // 64) * 64
            self.hbm_layout[name] = aligned_addr
    
    def compile_and_setup(
        self,
        code: str,
        fp_preload: Optional[List[float]] = None,
        build_dir: Optional[Path] = None,
        asm_name: str = "linear_compiler"
    ) -> Dict:
        """
        编译伪代码并自动设置仿真环境
        
        Args:
            code: 伪代码字符串，例如：
                '''
                B = Load_Batch(batch)
                M1 = Load_Matrix(matrix1)
                M2 = Load_Matrix(matrix2)
                I = B @ M1
                R = I @ M2
                Result R
                '''
            fp_preload: FP SRAM 预加载值
            build_dir: 构建目录（默认在 testbench/build）
            asm_name: 汇编文件名
            
        Returns:
            dict 包含：
                - compiler: SimpleCompiler 实例
                - generated_code: 生成的 ISA 代码
                - symbol_table: 符号表
                - golden_result: golden result dict
                - input_tensor: input tensor dict
                - comparison_params: 比较参数
        """
        if build_dir is None:
            build_dir = Path(__file__).parent / "build"
        build_dir.mkdir(exist_ok=True)
        
        # 1. 生成文本配置文件
        config_file = self._generate_config_file(code, build_dir)
        
        # 2. 使用 SimpleCompiler 编译
        compiler = SimpleCompiler(real_data_ratio=self.real_data_ratio, mlen=self.mlen, blen=self.blen)
        generated_code = compiler.parse_file(str(config_file))
        
        # 2.1 记录 Store 操作的映射（用于 golden result 计算）
        self._allocate_store_hbm_addresses(code, compiler)
        self.stored_tensors.update(compiler.stored_tensors)
        
        # 3. 计算 golden result（通过解析伪代码）
        golden_result = self._compute_golden_result(code)
        
        # 4. 准备 input_tensor dict（按 specified_data_order 顺序）
        # 重要：需要 reshape 到 (1, -1) 格式，create_sim_env 期望这种格式
        input_tensor = {name: tensor.reshape(1, -1) for name, tensor in self.tensors.items()}
        specified_data_order = list(self.tensors.keys())
        
        # 5. 设置 fp_preload
        if fp_preload is None:
            # 默认值：尝试从第一个 batch 的 shape 推断
            batch_shapes = [s for n, s in self.tensor_shapes.items() 
                          if self._is_batch_in_code(code, n)]
            if batch_shapes:
                M = batch_shapes[0][1]  # 取第一个 batch 的 features
                fp_preload = [0.0, 1e-6, 1/M]
            else:
                fp_preload = [0.0, 1e-6, 1.0]
        
        # 6. 创建仿真环境
        # 重要：golden_result["input_tensor"] 也需要是 reshaped 的版本
        golden_result["input_tensor"] = input_tensor
        create_sim_env(input_tensor, generated_code, golden_result, fp_preload)
        create_mem_for_sim(
            data_size=256,
            mode="behave_sim",
            asm=asm_name,
            data=None,
            specified_data_order=specified_data_order
        )
        
        # 7. 保存生成的 ISA 代码
        generated_code_file = build_dir / "generated_asm_code_before_run.asm"
        with open(generated_code_file, "w") as f:
            f.write(generated_code)
        
        # 7.5 保存完整的 golden result 到文件（用于调试比较）
        self._save_golden_result(golden_result, build_dir)
        
        # 8. 检查 H_PREFETCH_M 对齐问题
        alignment_warnings = self._check_hprefetch_alignment(generated_code)
        
        # 9. 计算 comparison params（从 Result 语句中获取结果 tensor）
        # 输出格式统一为 (num_out_blocks, batch, mlen)，无论是 TMM 还是普通 matmul
        result_tensor_name = self._extract_result_tensor(code)
        symbol_table = compiler.get_symbol_table()
        
        if result_tensor_name in symbol_table:
            result_info = symbol_table[result_tensor_name]
            result_start_row = result_info.vram_addr // 64
            result_shape = result_info.shape
            num_result_rows = (result_shape[0] * result_shape[1]) // 64
            
            # 统一格式: num_batches=rows, elements_per_batch=cols
            comparison_params = {
                "start_row_idx": result_start_row,
                "num_rows": num_result_rows,
                "num_batches": result_shape[0],
                "elements_per_batch": result_shape[1]
            }
        else:
            comparison_params = {}
        
        # 保存 comparison params
        with open(build_dir / "comparison_params.json", "w") as f:
            json.dump(comparison_params, f, indent=2)
        
        return {
            "compiler": compiler,
            "generated_code": generated_code,
            "symbol_table": symbol_table,
            "golden_result": golden_result,
            "input_tensor": input_tensor,
            "comparison_params": comparison_params,
            "alignment_warnings": alignment_warnings,
            "build_dir": build_dir
        }
    
    def _generate_config_file(self, code: str, build_dir: Path) -> Path:
        """生成文本配置文件"""
        config_file = build_dir / "auto_config.txt"
        
        # 生成 Input tensors 部分
        lines = ["# Input tensors (auto-generated)"]
        for name in self.tensors.keys():
            h, w = self.tensor_shapes[name]
            hbm_addr = self.hbm_layout[name]
            hbm_size = self.hbm_sizes[name]
            lines.append(f"{name}: hbm_addr={hbm_addr}, hbm_size={hbm_size}, shape=({h}, {w})")
        
        lines.append("")
        lines.append("# Code")
        lines.append(code.strip())
        
        with open(config_file, "w") as f:
            f.write("\n".join(lines))
        
        return config_file
    
    def _preprocess_for_loops(self, lines: list) -> list:
        """
        预处理 for 循环：在编译时展开所有循环
        
        支持语法：
        - for i in range(start, end, step):
              # body
          endfor
        """
        expanded_lines = []
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            
            # 匹配 for 循环: for i in range(...)
            match = re.match(r'for\s+(\w+)\s+in\s+range\s*\(\s*([^)]+)\s*\)\s*:', line, re.IGNORECASE)
            if match:
                loop_var = match.group(1)
                range_args = match.group(2)
                
                # 解析 range 参数
                args = [a.strip() for a in range_args.split(',')]
                if len(args) == 1:
                    start, end, step = 0, int(args[0]), 1
                elif len(args) == 2:
                    start, end, step = int(args[0]), int(args[1]), 1
                else:
                    start, end, step = int(args[0]), int(args[1]), int(args[2])
                
                # 收集循环体直到 endfor
                loop_body = []
                nest_level = 1
                i += 1
                while i < len(lines) and nest_level > 0:
                    body_line = lines[i]
                    stripped = body_line.strip().lower()
                    
                    if stripped.startswith('for ') and ':' in stripped:
                        nest_level += 1
                        loop_body.append(body_line)
                    elif stripped == 'endfor':
                        nest_level -= 1
                        if nest_level > 0:
                            loop_body.append(body_line)
                    else:
                        loop_body.append(body_line)
                    i += 1
                
                # 展开循环
                for val in range(start, end, step):
                    for body_line in loop_body:
                        # 替换循环变量
                        expanded_line = re.sub(
                            rf'\b{loop_var}\b', 
                            str(val), 
                            body_line
                        )
                        expanded_lines.append(expanded_line)
            else:
                expanded_lines.append(lines[i])
                i += 1
        
        # 递归处理嵌套循环
        if any('for ' in line.lower() and 'range' in line.lower() for line in expanded_lines):
            return self._preprocess_for_loops(expanded_lines)
        
        return expanded_lines
    
    def _compute_golden_result(self, code: str) -> Dict:
        """
        通过解析伪代码计算 golden result
        支持链式矩阵乘法：A @ B @ C -> (A @ B) @ C
        支持 Store(A, a) 和 A = Load(a) 操作
        支持 Load_Matrix 从 stored 数据加载（会在后续处理中解析）
        支持 Sub Matrix 操作
        支持 For Loop 操作
        支持 Softmax 相关操作
        """
        # 首先建立别名到原始 tensor 的映射
        alias_to_tensor = {}  # alias -> original_tensor_name
        deferred_loads = []  # 延迟处理的 Load_Matrix (可能从 stored 加载)
        sub_matrices = {}  # 子矩阵别名 -> 原始 tensor 名称
        allocated_matrices = {}  # 已分配的大 VRAM 矩阵
        
        # 预处理：展开 for 循环
        raw_lines = code.strip().split('\n')
        lines = self._preprocess_for_loops(raw_lines)
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # 匹配: alias = Load_Batch(original_name) 或 alias = Load_Matrix(original_name)
            match = re.match(r'(\w+)\s*=\s*Load_(?:Batch|Matrix)\s*\(\s*(\w+)\s*\)', line, re.IGNORECASE)
            if match:
                alias = match.group(1)  # 例如: B
                original_name = match.group(2)  # 例如: batch
                if original_name in self.tensors:
                    alias_to_tensor[alias] = original_name
                else:
                    # 可能是从 stored 数据加载，延迟处理
                    deferred_loads.append((alias, original_name))
            
            # 匹配: W = Register_SubMatrix(weight) 或 W = Register_SubMatrix(weight, h, w)
            match = re.match(r'(\w+)\s*=\s*Register_SubMatrix\s*\(\s*(\w+)', line, re.IGNORECASE)
            if match:
                alias = match.group(1)
                original_name = match.group(2)
                if original_name in self.tensors:
                    sub_matrices[alias] = original_name
                    alias_to_tensor[alias] = original_name
            
            # 匹配: A_sub = Register_VRAMSubMatrix(A)
            match = re.match(r'(\w+)\s*=\s*Register_VRAMSubMatrix\s*\(\s*(\w+)\s*\)', line, re.IGNORECASE)
            if match:
                alias = match.group(1)
                source_alias = match.group(2)
                # 将 VRAM 子矩阵别名映射到源 tensor 的别名
                if source_alias in alias_to_tensor:
                    sub_matrices[alias] = alias_to_tensor[source_alias]
                    alias_to_tensor[alias] = alias_to_tensor[source_alias]
                elif source_alias in self.tensors:
                    sub_matrices[alias] = source_alias
                    alias_to_tensor[alias] = source_alias
        
        # 提取所有操作（包括 Store, Load, Load_Matrix, Load_Batch）
        operations = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # 匹配: Store(A, a) - 将 A 存储到 HBM，命名为 a
            match = re.match(r'Store\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)', line, re.IGNORECASE)
            if match:
                tensor_alias = match.group(1)
                store_name = match.group(2)
                operations.append(('store', store_name, tensor_alias, None))
                continue
            
            # 匹配: a = Store(A)
            match = re.match(r'(\w+)\s*=\s*Store\s*\(\s*(\w+)\s*\)', line, re.IGNORECASE)
            if match:
                store_name = match.group(1)
                tensor_alias = match.group(2)
                operations.append(('store', store_name, tensor_alias, None))
                continue
            
            # 匹配: A = Load(a) - 从 HBM 加载之前存储的 a (作为 Batch)
            match = re.match(r'(\w+)\s*=\s*Load\s*\(\s*(\w+)\s*\)', line, re.IGNORECASE)
            if match:
                alias = match.group(1)
                store_name = match.group(2)
                operations.append(('load', alias, store_name, None))
                continue
            
            # 匹配: A = Load_Matrix(x) - 可能从 stored 加载 (作为 Matrix，会转置)
            match = re.match(r'(\w+)\s*=\s*Load_Matrix\s*\(\s*(\w+)\s*\)', line, re.IGNORECASE)
            if match:
                alias = match.group(1)
                source_name = match.group(2)
                # 检查是否是延迟加载（从 stored 加载）
                if source_name not in self.tensors:
                    operations.append(('load_matrix_from_stored', alias, source_name, None))
                continue
            
            # 匹配: FlashAttention(Q, K, V) -> O = softmax(Q @ K^T) @ V
            match = re.match(r'(\w+)\s*=\s*FlashAttention\s*\(\s*(\w+)\s*,\s*(\w+)\s*,\s*(\w+)\s*\)', line, re.IGNORECASE)
            if match:
                result_name = match.group(1)
                q_name = match.group(2)
                k_name = match.group(3)
                v_name = match.group(4)
                operations.append(('flash_attention', result_name, q_name, k_name, v_name))
                continue
            
            # 匹配: TMM_Matmul(A, B) 或 A @T B 或 A @ TransposeMatrix(B)
            # TMM MatMul: Batch @ Matrix^T
            # 注意：不要匹配 A @T SubMatrix(W, 1) 这种格式
            match = re.match(r'(\w+)\s*=\s*TMM_Matmul\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)', line, re.IGNORECASE)
            if not match:
                # 排除 SubMatrix 格式
                if 'SubMatrix' not in line:
                    match = re.match(r'(\w+)\s*=\s*(\w+)\s*@T\s*(\w+)', line, re.IGNORECASE)
            if not match:
                match = re.match(r'(\w+)\s*=\s*(\w+)\s*@\s*TransposeMatrix\s*\(\s*(\w+)\s*\)', line, re.IGNORECASE)
            
            if match:
                result_name = match.group(1)
                left_name = match.group(2)
                right_name = match.group(3)
                operations.append(('tmm_matmul', result_name, left_name, right_name))
                continue
            
            # 匹配: C = A @ SubMatrix(W, 1)
            # Sub Projection: Batch @ SubMatrix[row_idx][:]
            match = re.match(r'(\w+)\s*=\s*(\w+)\s*@\s*SubMatrix\s*\(\s*(\w+)\s*,\s*(\d+)\s*\)', line, re.IGNORECASE)
            if match:
                result_name = match.group(1)
                left_name = match.group(2)
                mat_alias = match.group(3)
                row_idx = int(match.group(4))
                operations.append(('sub_projection', result_name, left_name, mat_alias, row_idx))
                continue
            
            # 匹配: C = A @T SubMatrix(W, 1)
            # Sub Projection T: Batch @ SubMatrix[row_idx][:]^T
            match = re.match(r'(\w+)\s*=\s*(\w+)\s*@T\s*SubMatrix\s*\(\s*(\w+)\s*,\s*(\d+)\s*\)', line, re.IGNORECASE)
            if match:
                result_name = match.group(1)
                left_name = match.group(2)
                mat_alias = match.group(3)
                row_idx = int(match.group(4))
                operations.append(('sub_projection_t', result_name, left_name, mat_alias, row_idx))
                continue
            
            # 匹配: C = Allocate_VRAMMatrix(rows, cols)
            # 分配大的 VRAM 矩阵用于存储子块组合结果
            match = re.match(r'(\w+)\s*=\s*Allocate_VRAMMatrix\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)', line, re.IGNORECASE)
            if match:
                mat_name = match.group(1)
                rows = int(match.group(2))
                cols = int(match.group(3))
                operations.append(('allocate_vram_matrix', mat_name, rows, cols))
                continue
            
            # 匹配: C[r][c] = VRAMSubMatrix(A, i) @ SubMatrix(W, j)
            # 将子块乘法结果写入大矩阵的指定位置
            match = re.match(
                r'(\w+)\s*\[\s*(\d+)\s*\]\s*\[\s*(\d+)\s*\]\s*=\s*VRAMSubMatrix\s*\(\s*(\w+)\s*,\s*(\d+)\s*\)\s*@\s*SubMatrix\s*\(\s*(\w+)\s*,\s*(\d+)\s*\)',
                line, re.IGNORECASE
            )
            if match:
                target_mat = match.group(1)
                target_row_idx = int(match.group(2))
                target_col_idx = int(match.group(3))
                vram_alias = match.group(4)
                vram_row_idx = int(match.group(5))
                mram_alias = match.group(6)
                mram_col_idx = int(match.group(7))
                operations.append(('vram_sub_projection_to', target_mat, target_row_idx, target_col_idx, 
                                   vram_alias, vram_row_idx, mram_alias, mram_col_idx))
                continue
            
            # 匹配: C = VRAMSubMatrix(A, 1) @ SubMatrix(W, 2)
            # VRAM Sub Projection: VRAM_A[row_idx][:] @ MRAM_W[:][col_idx]
            match = re.match(
                r'(\w+)\s*=\s*VRAMSubMatrix\s*\(\s*(\w+)\s*,\s*(\d+)\s*\)\s*@\s*SubMatrix\s*\(\s*(\w+)\s*,\s*(\d+)\s*\)',
                line, re.IGNORECASE
            )
            if match:
                result_name = match.group(1)
                vram_alias = match.group(2)
                vram_row_idx = int(match.group(3))
                mram_alias = match.group(4)
                mram_col_idx = int(match.group(5))
                operations.append(('vram_sub_projection', result_name, vram_alias, vram_row_idx, mram_alias, mram_col_idx))
                continue
            
            # 匹配: C = VRAMSubMatrix(A, 1) @T SubMatrix(W, 1)
            # VRAM Sub Projection T: VRAM_A[row_idx][:] @ MRAM_W[row_idx][:]^T
            match = re.match(
                r'(\w+)\s*=\s*VRAMSubMatrix\s*\(\s*(\w+)\s*,\s*(\d+)\s*\)\s*@T\s*SubMatrix\s*\(\s*(\w+)\s*,\s*(\d+)\s*\)',
                line, re.IGNORECASE
            )
            if match:
                result_name = match.group(1)
                vram_alias = match.group(2)
                vram_row_idx = int(match.group(3))
                mram_alias = match.group(4)
                mram_row_idx = int(match.group(5))
                operations.append(('vram_sub_projection_t', result_name, vram_alias, vram_row_idx, mram_alias, mram_row_idx))
                continue
            
            # 匹配: S = Q @ Transpose(K) 或 S = MatMul(Q, Transpose(K))
            match = re.match(r'(\w+)\s*=\s*(\w+)\s*@\s*Transpose\s*\(\s*(\w+)\s*\)', line, re.IGNORECASE)
            if not match:
                match = re.match(r'(\w+)\s*=\s*MatMul\s*\(\s*(\w+)\s*,\s*Transpose\s*\(\s*(\w+)\s*\)\s*\)', line, re.IGNORECASE)
            
            if match:
                result_name = match.group(1)
                left_name = match.group(2)
                right_name = match.group(3)
                operations.append(('qkt', result_name, left_name, right_name))
                continue
            
            # ================================================================
            # Softmax 相关操作（用于 Flash Attention 展开版本）
            # ================================================================
            
            # 匹配: m = Init(-inf, mlen) 或 O = Init(0, rows, cols)
            match = re.match(r'(\w+)\s*=\s*Init\s*\(\s*([^,]+)\s*,\s*(\d+)\s*(?:,\s*(\d+))?\s*\)', line, re.IGNORECASE)
            if match:
                var_name = match.group(1)
                init_val = match.group(2).strip()
                dim1 = int(match.group(3))
                dim2 = match.group(4)
                operations.append(('init', var_name, init_val, dim1, int(dim2) if dim2 else None))
                continue
            
            # 匹配: m = RowMax(S)
            match = re.match(r'(\w+)\s*=\s*RowMax\s*\(\s*(\w+)\s*\)', line, re.IGNORECASE)
            if match:
                result_name = match.group(1)
                source_name = match.group(2)
                operations.append(('row_max', result_name, source_name))
                continue
            
            # 匹配: l = RowSum(S)
            match = re.match(r'(\w+)\s*=\s*RowSum\s*\(\s*(\w+)\s*\)', line, re.IGNORECASE)
            if match:
                result_name = match.group(1)
                source_name = match.group(2)
                operations.append(('row_sum', result_name, source_name))
                continue
            
            # 匹配: P = RowSoftmax(S)
            match = re.match(r'(\w+)\s*=\s*RowSoftmax\s*\(\s*(\w+)\s*\)', line, re.IGNORECASE)
            if match:
                result_name = match.group(1)
                source_name = match.group(2)
                operations.append(('row_softmax', result_name, source_name))
                continue
            
            # 匹配: S = Scale(S, factor)
            match = re.match(r'(\w+)\s*=\s*Scale\s*\(\s*(\w+)\s*,\s*([^)]+)\s*\)', line, re.IGNORECASE)
            if match:
                result_name = match.group(1)
                source_name = match.group(2)
                factor = match.group(3).strip()
                operations.append(('scale', result_name, source_name, factor))
                continue
            
            # 匹配: m_new = Max(m_old, m_cur)
            match = re.match(r'(\w+)\s*=\s*Max\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)', line, re.IGNORECASE)
            if match:
                result_name = match.group(1)
                left_name = match.group(2)
                right_name = match.group(3)
                operations.append(('elem_max', result_name, left_name, right_name))
                continue
            
            # 匹配: r = Exp(x)
            match = re.match(r'(\w+)\s*=\s*Exp\s*\(\s*(\w+)\s*\)', line, re.IGNORECASE)
            if match:
                result_name = match.group(1)
                source_name = match.group(2)
                operations.append(('exp', result_name, source_name))
                continue
            
            # 匹配: r = Sub(a, b)
            match = re.match(r'(\w+)\s*=\s*Sub\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)', line, re.IGNORECASE)
            if match:
                result_name = match.group(1)
                left_name = match.group(2)
                right_name = match.group(3)
                operations.append(('sub', result_name, left_name, right_name))
                continue
            
            # 匹配: r = Add(a, b)
            match = re.match(r'(\w+)\s*=\s*Add\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)', line, re.IGNORECASE)
            if match:
                result_name = match.group(1)
                left_name = match.group(2)
                right_name = match.group(3)
                operations.append(('add', result_name, left_name, right_name))
                continue
            
            # 匹配: r = Mul(a, b)
            match = re.match(r'(\w+)\s*=\s*Mul\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)', line, re.IGNORECASE)
            if match:
                result_name = match.group(1)
                left_name = match.group(2)
                right_name = match.group(3)
                operations.append(('mul', result_name, left_name, right_name))
                continue
            
            # 匹配: r = Div(a, b)
            match = re.match(r'(\w+)\s*=\s*Div\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)', line, re.IGNORECASE)
            if match:
                result_name = match.group(1)
                left_name = match.group(2)
                right_name = match.group(3)
                operations.append(('div', result_name, left_name, right_name))
                continue
            
            # 匹配: result = left @ right
            match = re.match(r'(\w+)\s*=\s*(\w+)\s*@\s*(\w+)', line)
            if match:
                result_name = match.group(1)
                left_name = match.group(2)
                right_name = match.group(3)
                operations.append(('matmul', result_name, left_name, right_name))
        
        # 按顺序执行操作
        computed = {}  # 存储已计算的结果（使用别名）
        stored = {}  # 存储 Store 的数据（store_name -> tensor）
        intermediate_results = []
        
        def get_tensor(name):
            """获取 tensor，按优先级查找"""
            if name in self.tensors:
                return self.tensors[name]
            elif name in alias_to_tensor:
                return self.tensors[alias_to_tensor[name]]
            elif name in computed:
                return computed[name]
            elif name in stored:
                return stored[name]
            else:
                raise ValueError(f"Tensor '{name}' not found (not in tensors, aliases, computed, or stored)")
        
        for op in operations:
            op_type = op[0]
            
            if op_type == 'store':
                # Store(A, a): 将 A 的数据存储到 stored[a]
                _, store_name, tensor_alias, _ = op
                tensor = get_tensor(tensor_alias)
                stored[store_name] = tensor.clone()
                continue
            
            if op_type == 'load':
                # A = Load(a): 从 stored[a] 加载数据到 computed[A] (作为 Batch)
                _, alias, store_name, _ = op
                if store_name not in stored:
                    raise ValueError(f"Store name '{store_name}' not found. Store it first before loading.")
                computed[alias] = stored[store_name].clone()
                continue
            
            if op_type == 'load_matrix_from_stored':
                # A = Load_Matrix(a): 从 stored[a] 加载数据到 computed[A] (作为 Matrix)
                # 注意：硬件会将其当作 Matrix 处理，在 golden 计算中不需要转置
                # 因为 TMM_Matmul 已经处理了转置
                _, alias, store_name, _ = op
                if store_name not in stored:
                    raise ValueError(f"Store name '{store_name}' not found. Store it first before loading as Matrix.")
                computed[alias] = stored[store_name].clone()
                continue
            
            if op_type == 'flash_attention':
                # FlashAttention(Q, K, V): O = softmax(Q @ K^T / sqrt(d)) @ V
                # 完整实现，包括 softmax 和 P@V
                _, result_name, q_name, k_name, v_name = op
                
                q = get_tensor(q_name).to(torch.float32)
                k = get_tensor(k_name).to(torch.float32)
                v = get_tensor(v_name).to(torch.float32)
                
                # 获取 head_dim 用于计算 scale
                head_dim = q.shape[-1]
                scale = 1.0 / (head_dim ** 0.5)
                
                # 完整 Flash Attention:
                # 1. S = Q @ K^T / sqrt(d)
                # 2. P = softmax(S)  (row-wise softmax)
                # 3. O = P @ V
                s = scale * torch.matmul(q, k.T)  # (batch, seq_len)，应用 scale
                p = torch.softmax(s, dim=-1)  # row-wise softmax
                result = torch.matmul(p, v)  # (batch, head_dim)
                
                computed[result_name] = result
                intermediate_results.append((result_name, result))
                continue
            
            # Sub Projection 操作 (4 operands)
            # sub_projection: A @ W[idx][:] 意思是取 W 的第 idx 行子块
            # W[idx][:] 形状是 (mlen, hidden_size)
            # 要让 A @ W[idx][:] 可行，需要转置：A @ W[idx][:]^T
            # 
            # 语义重新定义：
            # - sub_projection: A @ SubMatrix(W, idx) = A @ W[:, idx*mlen:(idx+1)*mlen]
            #   即取 W 的第 idx 列子块，形状 (hidden, mlen)，结果 (batch, mlen)
            # - sub_projection_t: A @T SubMatrix(W, idx) = A @ W[idx*mlen:(idx+1)*mlen, :]^T
            #   即取 W 的第 idx 行子块并转置，形状 (hidden, mlen)，结果 (batch, mlen)
            
            if op_type == 'sub_projection':
                _, result_name, left_name, mat_alias, col_idx = op
                left = get_tensor(left_name).to(torch.float32)
                
                # 获取子矩阵对应的原始 tensor
                if mat_alias not in sub_matrices:
                    raise ValueError(f"SubMatrix '{mat_alias}' not registered")
                mat_tensor = get_tensor(sub_matrices[mat_alias]).to(torch.float32)
                
                # 提取子块列：mat[:, col_idx*mlen : (col_idx+1)*mlen]
                # 形状: (full_rows, mlen)
                mlen = self.mlen
                col_start = col_idx * mlen
                col_end = col_start + mlen
                sub_col = mat_tensor[:, col_start:col_end]  # (hidden_size, mlen)
                
                # 计算 left @ sub_col
                # left: (batch, hidden_size), sub_col: (hidden_size, mlen)
                # result: (batch, mlen)
                result = torch.matmul(left, sub_col)
                
                computed[result_name] = result
                intermediate_results.append((result_name, result))
                continue
            
            if op_type == 'sub_projection_t':
                _, result_name, left_name, mat_alias, row_idx = op
                left = get_tensor(left_name).to(torch.float32)
                
                # 获取子矩阵对应的原始 tensor
                if mat_alias not in sub_matrices:
                    raise ValueError(f"SubMatrix '{mat_alias}' not registered")
                mat_tensor = get_tensor(sub_matrices[mat_alias]).to(torch.float32)
                
                # 提取子块行：mat[row_idx*mlen : (row_idx+1)*mlen, :]
                # 形状: (mlen, full_cols)，转置后 (full_cols, mlen)
                mlen = self.mlen
                row_start = row_idx * mlen
                row_end = row_start + mlen
                sub_row = mat_tensor[row_start:row_end, :]  # (mlen, hidden_size)
                
                # 计算 left @ sub_row^T
                # left: (batch, hidden_size), sub_row^T: (hidden_size, mlen)
                # result: (batch, mlen)
                result = torch.matmul(left, sub_row.T)
                
                computed[result_name] = result
                intermediate_results.append((result_name, result))
                continue
            
            # 分配 VRAM 矩阵
            if op_type == 'allocate_vram_matrix':
                _, mat_name, rows, cols = op
                # 创建一个全零的大矩阵
                allocated_matrices[mat_name] = torch.zeros(rows, cols, dtype=torch.float32)
                computed[mat_name] = allocated_matrices[mat_name]
                continue
            
            # VRAM 子块乘法写入大矩阵指定位置 (8 operands)
            if op_type == 'vram_sub_projection_to':
                _, target_mat, target_row_idx, target_col_idx, vram_alias, vram_row_idx, mram_alias, mram_col_idx = op
                
                # 获取目标矩阵
                if target_mat not in allocated_matrices:
                    raise ValueError(f"Target matrix '{target_mat}' not allocated")
                target_tensor = allocated_matrices[target_mat]
                
                # 获取 VRAM 矩阵对应的原始 tensor
                if vram_alias not in sub_matrices:
                    raise ValueError(f"VRAM SubMatrix '{vram_alias}' not registered")
                vram_tensor = get_tensor(sub_matrices[vram_alias]).to(torch.float32)
                
                # 获取 MRAM 矩阵对应的原始 tensor
                if mram_alias not in sub_matrices:
                    raise ValueError(f"MRAM SubMatrix '{mram_alias}' not registered")
                mram_tensor = get_tensor(sub_matrices[mram_alias]).to(torch.float32)
                
                mlen = self.mlen
                
                # 提取 VRAM 行子块
                vram_row_start = vram_row_idx * mlen
                vram_row_end = vram_row_start + mlen
                vram_sub_row = vram_tensor[vram_row_start:vram_row_end, :]
                
                # 提取 MRAM 列子块
                mram_col_start = mram_col_idx * mlen
                mram_col_end = mram_col_start + mlen
                mram_sub_col = mram_tensor[:, mram_col_start:mram_col_end]
                
                # 计算子块乘法
                result = torch.matmul(vram_sub_row, mram_sub_col)
                
                # 写入目标矩阵的指定位置
                target_row_start = target_row_idx * mlen
                target_row_end = target_row_start + mlen
                target_col_start = target_col_idx * mlen
                target_col_end = target_col_start + mlen
                target_tensor[target_row_start:target_row_end, target_col_start:target_col_end] = result
                
                # 更新计算结果（整个目标矩阵）
                computed[target_mat] = target_tensor
                intermediate_results.append((f"{target_mat}[{target_row_idx}][{target_col_idx}]", result))
                continue
            
            # VRAM 子块乘法操作 (6 operands)
            if op_type == 'vram_sub_projection':
                _, result_name, vram_alias, vram_row_idx, mram_alias, mram_col_idx = op
                
                # 获取 VRAM 矩阵对应的原始 tensor
                if vram_alias not in sub_matrices:
                    raise ValueError(f"VRAM SubMatrix '{vram_alias}' not registered")
                vram_tensor = get_tensor(sub_matrices[vram_alias]).to(torch.float32)
                
                # 获取 MRAM 矩阵对应的原始 tensor
                if mram_alias not in sub_matrices:
                    raise ValueError(f"MRAM SubMatrix '{mram_alias}' not registered")
                mram_tensor = get_tensor(sub_matrices[mram_alias]).to(torch.float32)
                
                mlen = self.mlen
                
                # 提取 VRAM 行子块：vram[vram_row_idx*mlen : (vram_row_idx+1)*mlen, :]
                # 形状: (mlen, hidden_size)
                vram_row_start = vram_row_idx * mlen
                vram_row_end = vram_row_start + mlen
                vram_sub_row = vram_tensor[vram_row_start:vram_row_end, :]  # (mlen, hidden_size)
                
                # 提取 MRAM 列子块：mram[:, mram_col_idx*mlen : (mram_col_idx+1)*mlen]
                # 形状: (hidden_size, mlen)
                mram_col_start = mram_col_idx * mlen
                mram_col_end = mram_col_start + mlen
                mram_sub_col = mram_tensor[:, mram_col_start:mram_col_end]  # (hidden_size, mlen)
                
                # 计算 vram_sub_row @ mram_sub_col
                # vram_sub_row: (mlen, hidden_size), mram_sub_col: (hidden_size, mlen)
                # result: (mlen, mlen)
                result = torch.matmul(vram_sub_row, mram_sub_col)
                
                computed[result_name] = result
                intermediate_results.append((result_name, result))
                continue
            
            if op_type == 'vram_sub_projection_t':
                _, result_name, vram_alias, vram_row_idx, mram_alias, mram_row_idx = op
                
                # 获取 VRAM 矩阵对应的原始 tensor
                if vram_alias not in sub_matrices:
                    raise ValueError(f"VRAM SubMatrix '{vram_alias}' not registered")
                vram_tensor = get_tensor(sub_matrices[vram_alias]).to(torch.float32)
                
                # 获取 MRAM 矩阵对应的原始 tensor
                if mram_alias not in sub_matrices:
                    raise ValueError(f"MRAM SubMatrix '{mram_alias}' not registered")
                mram_tensor = get_tensor(sub_matrices[mram_alias]).to(torch.float32)
                
                mlen = self.mlen
                
                # 提取 VRAM 行子块：vram[vram_row_idx*mlen : (vram_row_idx+1)*mlen, :]
                # 形状: (mlen, hidden_size)
                vram_row_start = vram_row_idx * mlen
                vram_row_end = vram_row_start + mlen
                vram_sub_row = vram_tensor[vram_row_start:vram_row_end, :]  # (mlen, hidden_size)
                
                # 提取 MRAM 行子块：mram[mram_row_idx*mlen : (mram_row_idx+1)*mlen, :]
                # 形状: (mlen, hidden_size)，转置后 (hidden_size, mlen)
                mram_row_start = mram_row_idx * mlen
                mram_row_end = mram_row_start + mlen
                mram_sub_row = mram_tensor[mram_row_start:mram_row_end, :]  # (mlen, hidden_size)
                
                # 计算 vram_sub_row @ mram_sub_row^T
                # vram_sub_row: (mlen, hidden_size), mram_sub_row^T: (hidden_size, mlen)
                # result: (mlen, mlen)
                result = torch.matmul(vram_sub_row, mram_sub_row.T)
                
                computed[result_name] = result
                intermediate_results.append((result_name, result))
                continue
            
            # ================================================================
            # Softmax 相关操作的执行
            # ================================================================
            
            # Init 操作
            if op_type == 'init':
                _, var_name, init_val, dim1, dim2 = op
                # 解析初始值
                if init_val == '-inf':
                    value = float('-inf')
                else:
                    value = float(init_val)
                
                if dim2 is None:
                    # 向量初始化
                    result = torch.full((dim1,), value, dtype=torch.float32)
                else:
                    # 矩阵初始化
                    result = torch.full((dim1, dim2), value, dtype=torch.float32)
                
                computed[var_name] = result
                continue
            
            # RowMax 操作
            if op_type == 'row_max':
                _, result_name, source_name = op
                source = get_tensor(source_name).to(torch.float32)
                result = source.max(dim=-1).values  # 每行最大值
                computed[result_name] = result
                intermediate_results.append((result_name, result))
                continue
            
            # RowSum 操作
            if op_type == 'row_sum':
                _, result_name, source_name = op
                source = get_tensor(source_name).to(torch.float32)
                result = source.sum(dim=-1)  # 每行求和
                computed[result_name] = result
                intermediate_results.append((result_name, result))
                continue
            
            # RowSoftmax 操作
            if op_type == 'row_softmax':
                _, result_name, source_name = op
                source = get_tensor(source_name).to(torch.float32)
                result = torch.softmax(source, dim=-1)  # 行 softmax
                computed[result_name] = result
                intermediate_results.append((result_name, result))
                continue
            
            # Scale 操作
            if op_type == 'scale':
                _, result_name, source_name, factor_str = op
                source = get_tensor(source_name).to(torch.float32)
                # 解析 factor（可能是数字或表达式如 1/sqrt(64)）
                try:
                    factor = eval(factor_str, {"sqrt": math.sqrt, "math": math})
                except:
                    factor = float(factor_str)
                result = source * factor
                computed[result_name] = result
                intermediate_results.append((result_name, result))
                continue
            
            # 逐元素最大
            if op_type == 'elem_max':
                _, result_name, left_name, right_name = op
                left = get_tensor(left_name).to(torch.float32)
                right = get_tensor(right_name).to(torch.float32)
                result = torch.maximum(left, right)
                computed[result_name] = result
                intermediate_results.append((result_name, result))
                continue
            
            # Exp 操作
            if op_type == 'exp':
                _, result_name, source_name = op
                source = get_tensor(source_name).to(torch.float32)
                result = torch.exp(source)
                computed[result_name] = result
                intermediate_results.append((result_name, result))
                continue
            
            # Sub 操作
            if op_type == 'sub':
                _, result_name, left_name, right_name = op
                left = get_tensor(left_name).to(torch.float32)
                right = get_tensor(right_name).to(torch.float32)
                # 支持广播：如果 right 是向量，扩展为矩阵
                if left.dim() == 2 and right.dim() == 1:
                    right = right.unsqueeze(-1)  # (rows,) -> (rows, 1)
                result = left - right
                computed[result_name] = result
                intermediate_results.append((result_name, result))
                continue
            
            # Add 操作
            if op_type == 'add':
                _, result_name, left_name, right_name = op
                left = get_tensor(left_name).to(torch.float32)
                right = get_tensor(right_name).to(torch.float32)
                result = left + right
                computed[result_name] = result
                intermediate_results.append((result_name, result))
                continue
            
            # Mul 操作
            if op_type == 'mul':
                _, result_name, left_name, right_name = op
                left = get_tensor(left_name).to(torch.float32)
                right = get_tensor(right_name).to(torch.float32)
                # 支持广播
                if left.dim() == 2 and right.dim() == 1:
                    right = right.unsqueeze(-1)  # 行向量 -> 列向量广播
                result = left * right
                computed[result_name] = result
                intermediate_results.append((result_name, result))
                continue
            
            # Div 操作
            if op_type == 'div':
                _, result_name, left_name, right_name = op
                left = get_tensor(left_name).to(torch.float32)
                right = get_tensor(right_name).to(torch.float32)
                # 支持广播
                if left.dim() == 2 and right.dim() == 1:
                    right = right.unsqueeze(-1)  # 行向量 -> 列向量广播
                result = left / right
                computed[result_name] = result
                intermediate_results.append((result_name, result))
                continue
            
            # 矩阵乘法操作 (3 operands)
            _, result_name, left_name, right_name = op
            left = get_tensor(left_name)
            right = get_tensor(right_name)
            
            # 执行操作 (使用 float32 精度计算 golden，匹配硬件行为)
            left_f32 = left.to(torch.float32)
            right_f32 = right.to(torch.float32)
            
            if op_type == 'qkt':
                # Q @ K^T (Batch @ Batch^T)
                result = torch.matmul(left_f32, right_f32.T)
            elif op_type == 'tmm_matmul':
                # Batch @ Matrix^T (TMM MatMul)
                result = torch.matmul(left_f32, right_f32.T)
            else:
                # 标准矩阵乘法
                result = torch.matmul(left_f32, right_f32)
            
            computed[result_name] = result
            intermediate_results.append((result_name, result))
        
        # 找到最终结果
        result_tensor_name = self._extract_result_tensor(code)
        if result_tensor_name not in computed:
            raise ValueError(f"Result tensor '{result_tensor_name}' not computed")
        
        final_output = computed[result_tensor_name]
        
        return {
            "input_tensor": self.tensors.copy(),
            "original_output": final_output,
            "intermediate": dict(intermediate_results) if intermediate_results else None,
            "stored": stored if stored else None
        }
    
    def _extract_result_tensor(self, code: str) -> str:
        """从代码中提取 Result 语句指定的 tensor"""
        lines = code.strip().split('\n')
        for line in lines:
            line = line.strip()
            match = re.match(r'Result\s+(\w+)', line, re.IGNORECASE)
            if match:
                return match.group(1)
        raise ValueError("No 'Result' statement found in code")
    
    def _is_batch_in_code(self, code: str, tensor_name: str) -> bool:
        """检查 tensor 是否在代码中被 Load_Batch"""
        return f"Load_Batch({tensor_name})" in code or f"Load_Batch({tensor_name.lower()})" in code
    
    def _parse_store_operations(self, code: str) -> List[Tuple[str, str]]:
        """
        解析代码中的 Store 操作
        
        Returns:
            List of (tensor_alias, store_name) tuples
        """
        stores = []
        lines = code.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # 匹配: Store(A, a)
            match = re.match(r'Store\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)', line, re.IGNORECASE)
            if match:
                tensor_alias = match.group(1)
                store_name = match.group(2)
                stores.append((tensor_alias, store_name))
                continue
            
            # 匹配: a = Store(A)
            match = re.match(r'(\w+)\s*=\s*Store\s*\(\s*(\w+)\s*\)', line, re.IGNORECASE)
            if match:
                store_name = match.group(1)
                tensor_alias = match.group(2)
                stores.append((tensor_alias, store_name))
        
        return stores
    
    def _allocate_store_hbm_addresses(self, code: str, compiler: SimpleCompiler):
        """
        为 Store 操作预分配 HBM 地址
        
        这样在 golden result 计算时可以知道 Store 后的数据位置
        """
        stores = self._parse_store_operations(code)
        
        for tensor_alias, store_name in stores:
            # 记录 Store 映射
            self.stored_tensors[store_name] = tensor_alias
    
    def _save_golden_result(self, golden_result: Dict, build_dir: Path):
        """
        保存完整的 golden result 到文件，用于调试比较
        
        保存格式：
        1. golden_output.npy - numpy 格式，可以用 np.load 加载
        2. golden_output.txt - 人类可读的文本格式
        3. golden_output_flat.txt - 扁平化的值列表（按行优先顺序）
        """
        import numpy as np
        
        output = golden_result.get("original_output")
        if output is None:
            return
        
        # 转换为 numpy
        if hasattr(output, 'detach'):
            output_np = output.detach().cpu().float().numpy()
        else:
            output_np = np.array(output, dtype=np.float32)
        
        # 1. 保存 numpy 格式
        np.save(build_dir / "___GOLDEN_OUTPUT___.npy", output_np)
        
        # 2. 保存人类可读格式（完整矩阵）
        with open(build_dir / "___GOLDEN_OUTPUT_MATRIX___.txt", "w") as f:
            f.write(f"# Golden Output Shape: {output_np.shape}\n")
            f.write(f"# dtype: {output_np.dtype}\n\n")
            
            if output_np.ndim == 2:
                rows, cols = output_np.shape
                f.write(f"# Matrix ({rows} x {cols}):\n")
                for i in range(rows):
                    row_str = ", ".join([f"{v:12.6f}" for v in output_np[i]])
                    f.write(f"Row {i:3d}: [{row_str}]\n")
            else:
                f.write(f"# Array: {output_np}\n")
        
        # 3. 保存扁平化格式（用于逐元素比较）
        with open(build_dir / "___GOLDEN_OUTPUT_FLAT___.txt", "w") as f:
            f.write(f"# Flattened golden output (row-major order)\n")
            f.write(f"# Shape: {output_np.shape}\n")
            f.write(f"# Total elements: {output_np.size}\n\n")
            
            flat = output_np.flatten()
            for i, v in enumerate(flat):
                f.write(f"{i:6d}: {v:15.8f}\n")
        
        # 4. 保存列块优先格式（与 VRAM 存储格式一致）
        if output_np.ndim == 2:
            rows, cols = output_np.shape
            mlen = 64  # 假设 mlen = 64
            num_col_blocks = (cols + mlen - 1) // mlen
            
            with open(build_dir / "___GOLDEN_OUTPUT_VRAM_FORMAT___.txt", "w") as f:
                f.write(f"# Golden output in column-block-major format (VRAM layout)\n")
                f.write(f"# Shape: {output_np.shape}, mlen={mlen}, num_col_blocks={num_col_blocks}\n")
                f.write(f"# Storage format: (batch, mlen, hidden/mlen)\n\n")
                
                vram_idx = 0
                for col_block in range(num_col_blocks):
                    col_start = col_block * mlen
                    col_end = min(col_start + mlen, cols)
                    f.write(f"# === Column Block {col_block} (cols {col_start}:{col_end}) ===\n")
                    
                    for row in range(rows):
                        for col in range(col_start, col_end):
                            v = output_np[row, col]
                            f.write(f"VRAM[{vram_idx:6d}] = golden[{row:3d},{col:3d}] = {v:15.8f}\n")
                            vram_idx += 1
                        f.write("\n")  # 每行后空行
        
        print(f"\n{'='*60}")
        print(f"[AutoCompilerHelper] ★★★ GOLDEN RESULT SAVED ★★★")
        print(f"{'='*60}")
        print(f"  ► {build_dir / '___GOLDEN_OUTPUT___.npy'}")
        print(f"  ► {build_dir / '___GOLDEN_OUTPUT_MATRIX___.txt'}")
        print(f"  ► {build_dir / '___GOLDEN_OUTPUT_FLAT___.txt'}")
        print(f"  ► {build_dir / '___GOLDEN_OUTPUT_VRAM_FORMAT___.txt'}")
        print(f"{'='*60}\n")
    
    def _check_hprefetch_alignment(self, generated_code: str) -> List[str]:
        """
        检查 H_PREFETCH_M 指令是否使用了非 4096 倍数的地址
        
        Returns:
            警告列表，如果没有问题则返回空列表
        """
        lines = generated_code.splitlines()
        warnings = []
        
        for i, line in enumerate(lines, 1):
            if "H_PREFETCH_M" in line:
                match = re.search(r'H_PREFETCH_M gp(\d+)', line)
                if match:
                    reg_num = int(match.group(1))
                    # 向前查找这个寄存器最近一次被设置的值
                    for j in range(i-1, max(0, i-20), -1):
                        reg_match = re.search(rf'S_ADDI_INT gp{reg_num}, gp0, (\d+)', lines[j])
                        if reg_match:
                            addr = int(reg_match.group(1))
                            if addr % 4096 != 0:
                                warnings.append(f"Line {i}: H_PREFETCH_M uses gp{reg_num} with address {addr} (not multiple of 4096!)")
                                warnings.append(f"  Line {j}: {lines[j]}")
                                warnings.append(f"  Line {i}: {line}")
                            break
        
        return warnings

