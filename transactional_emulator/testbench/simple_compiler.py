"""
Simple Compiler: 从文本文件读取配置和伪代码，生成 ISA
支持简单的伪代码语法：
- A = Load_Batch(X)  # 从输入表加载 X 到 VRAM，命名为 A
- B = Load_Matrix(Y)  # 从输入表加载 Y（在 HBM），命名为 B
- C = A @ B  # 矩阵乘法
- Result C  # 指定输出结果

MRAM Sub Matrix 语法（权重矩阵在 MRAM）：
- W = Register_SubMatrix(weight, 256, 256)  # 注册 256x256 矩阵为子块管理
- Load_SubMatrix(W, 1)  # 加载 W[:][1] 到 MRAM（列子块，用于 sub_projection）
- Load_SubMatrix_Row(W, 1)  # 加载 W[1][:] 到 MRAM（行子块，用于 sub_projection_T）
- C = A @ SubMatrix(W, 1)  # C = A @ W[:, col*mlen:(col+1)*mlen]
- C = A @T SubMatrix(W, 1)  # C = A @ W[row*mlen:(row+1)*mlen, :].T

VRAM Sub Matrix 语法（激活矩阵在 VRAM）：
- A_sub = Register_VRAMSubMatrix(A)  # 将 VRAM 中的 A 注册为子块管理
- C = VRAMSubMatrix(A_sub, 1) @ SubMatrix(W, 2)  # A[1][:] @ W[:][2] -> (mlen, mlen)
- C = VRAMSubMatrix(A_sub, 1) @T SubMatrix(W, 1)  # A[1][:] @ W[1][:]^T -> (mlen, mlen)

写入大矩阵指定位置：
- C = Allocate_VRAMMatrix(128, 128)  # 分配 128x128 的 VRAM 矩阵
- C[0][0] = VRAMSubMatrix(A_sub, 0) @ SubMatrix(W, 0)  # 写入 C 的 [0][0] 子块
- C[0][1] = VRAMSubMatrix(A_sub, 0) @ SubMatrix(W, 1)  # 写入 C 的 [0][1] 子块

For Loop 语法（用于 Flash Attention 等分块算法）：
- for i in range(0, seq_len, mlen):
      # loop body
  endfor

Softmax 相关操作：
- m = RowMax(S)  # 每行最大值 -> (mlen,)
- l = RowSum(S)  # 每行求和 -> (mlen,)
- P = RowSoftmax(S)  # 行 softmax -> (mlen, mlen)
- S = Scale(S, factor)  # 标量乘法

标量向量操作：
- m_new = Max(m_old, m_cur)  # 逐元素最大
- m_res = Exp(Sub(m_old, m_new))  # exp(m_old - m_new)
- O = Add(Mul(m_res, O_old), PV)  # m_res * O_old + PV

初始化操作：
- m = Init(-inf, mlen)  # 初始化为 -inf
- l = Init(0, mlen)  # 初始化为 0
- O = Init(0, mlen, head_dim)  # 初始化矩阵为 0
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from developer_compiler import DeveloperCompiler


@dataclass
class InputTensor:
    """输入张量信息"""
    name: str
    hbm_addr: int
    hbm_size: int
    shape: Tuple[int, int]  # (h, w)
    dtype: str = "fp16"
    real_data_ratio: float = 1.125


class SimpleCompiler:
    """简单编译器：从文本文件读取配置和伪代码"""
    
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
        self.compiler = DeveloperCompiler(mlen=mlen, blen=blen)
        self.input_tensors: Dict[str, InputTensor] = {}
        self.tensor_aliases: Dict[str, str] = {}  # 伪代码中的名字 -> symbol table 中的名字
        self.stored_tensors: Dict[str, str] = {}  # 存储名 -> 原始 tensor 别名（用于 Store(A, a)）
        self.next_hbm_addr: int = 0  # 用于动态分配 HBM 地址
        self.sub_matrices: Dict[str, str] = {}  # MRAM 子矩阵别名 -> 原始名称
        self.loaded_sub_cols: Dict[str, List[int]] = {}  # MRAM 子矩阵名 -> 已加载的列索引列表
        self.loaded_sub_rows: Dict[str, List[int]] = {}  # MRAM 子矩阵名 -> 已加载的行索引列表
        self.vram_sub_matrices: Dict[str, str] = {}  # VRAM 子矩阵别名 -> 源 tensor 名称
        
        # For loop 状态
        self.loop_stack: List[Dict] = []  # 循环栈，用于嵌套循环
        self.loop_variables: Dict[str, int] = {}  # 循环变量当前值
        
        # Softmax 相关状态
        self.scalar_vectors: Dict[str, str] = {}  # 标量向量名称 -> 类型 (max/sum/etc)
        
    def parse_file(self, file_path: str) -> str:
        """
        解析文件并生成 ISA 代码
        
        文件格式：
        # Input tensors
        X: hbm_addr=0, hbm_size=1152, shape=(8, 128)
        Y: hbm_addr=1152, hbm_size=36864, shape=(128, 256)
        
        # Code
        A = Load_Batch(X)
        B = Load_Matrix(Y)
        C = A @ B
        Result C
        
        Returns:
            生成的 ISA 代码
        """
        with open(file_path, 'r') as f:
            content = f.read()
        
        # 分割输入表和代码部分
        parts = content.split('# Code', 1)
        if len(parts) != 2:
            raise ValueError("File must contain '# Code' section")
        
        input_section = parts[0]
        code_section = parts[1]
        
        # 解析输入表
        self._parse_input_tensors(input_section)
        
        # 解析代码
        self._parse_code(code_section)
        
        # 返回生成的代码
        return self.compiler.get_code()
    
    def _parse_input_tensors(self, section: str):
        """解析输入张量表"""
        lines = section.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # 跳过注释行和空行
            if not line or line.startswith('#'):
                continue
            
            # 如果包含冒号，尝试解析为张量定义
            # 格式: X: hbm_addr=0, hbm_size=1152, shape=(8, 128)
            if ':' in line:
                match = re.match(r'(\w+)\s*:\s*(.+)', line)
                if match:
                    name = match.group(1)
                    params_str = match.group(2)
                    
                    # 解析参数
                    hbm_addr = self._extract_param(params_str, 'hbm_addr', int, 0)
                    hbm_size = self._extract_param(params_str, 'hbm_size', int, 0)
                    shape_str = self._extract_param(params_str, 'shape', str, None)
                    dtype = self._extract_param(params_str, 'dtype', str, 'fp16')
                    
                    if shape_str:
                        # 解析 shape=(8, 128)
                        shape_match = re.match(r'\((\d+),\s*(\d+)\)', shape_str)
                        if shape_match:
                            h, w = int(shape_match.group(1)), int(shape_match.group(2))
                            self.input_tensors[name] = InputTensor(
                                name=name,
                                hbm_addr=hbm_addr,
                                hbm_size=hbm_size,
                                shape=(h, w),
                                dtype=dtype,
                                real_data_ratio=self.real_data_ratio
                            )
    
    def _extract_param(self, params_str: str, key: str, param_type, default):
        """从参数字符串中提取参数值"""
        # 对于 shape 参数，需要特殊处理，因为它包含括号
        if key == 'shape':
            # 匹配 shape=(...)
            match = re.search(rf'{key}\s*=\s*(\([^)]+\))', params_str)
            if match:
                return match.group(1).strip()
        else:
            # 对于其他参数，匹配到逗号或行尾
            match = re.search(rf'{key}\s*=\s*([^,]+)', params_str)
            if match:
                value_str = match.group(1).strip()
                if param_type == int:
                    return int(value_str)
                elif param_type == float:
                    return float(value_str)
                else:
                    return value_str
        return default
    
    def _preprocess_for_loops(self, lines: List[str]) -> List[str]:
        """
        预处理 for 循环：在编译时展开所有循环
        
        支持语法：
        - for i in range(start, end, step):
              # body
          endfor
        - for i in range(end):  # 等价于 range(0, end, 1)
        - for i in range(start, end):  # 等价于 range(start, end, 1)
        
        支持嵌套循环和循环变量替换。
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
    
    def _parse_code(self, section: str):
        """解析伪代码"""
        raw_lines = section.split('\n')
        
        # 预处理：展开 for 循环
        lines = self._preprocess_for_loops(raw_lines)
        
        result_tensor = None
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # 解析 Load: A = Load(a) - 从之前 Store 的 HBM 位置加载
            # 这是通用的 Load，适用于之前 Store 过的数据
            match = re.match(r'(\w+)\s*=\s*Load\s*\(\s*(\w+)\s*\)', line, re.IGNORECASE)
            if match:
                alias = match.group(1)  # A
                store_name = match.group(2)  # a (之前 Store 的名字)
                
                # 检查是否在 stored_tensors 或 input_tensors 中
                if store_name not in self.input_tensors:
                    raise ValueError(f"Stored tensor '{store_name}' not found. Store it first or define in input table.")
                
                tensor = self.input_tensors[store_name]
                self.compiler.load_batch(
                    name=alias,
                    hbm_addr=tensor.hbm_addr,
                    h=tensor.shape[0],
                    w=tensor.shape[1],
                    real_data_ratio=self.real_data_ratio,
                    vlen=64,
                    preload_len=4
                )
                self.tensor_aliases[alias] = alias
                continue
            
            # 解析 Load_Batch: A = Load_Batch(X)
            match = re.match(r'(\w+)\s*=\s*Load_Batch\s*\(\s*(\w+)\s*\)', line, re.IGNORECASE)
            if match:
                alias = match.group(1)  # A
                input_name = match.group(2)  # X
                if input_name not in self.input_tensors:
                    raise ValueError(f"Input tensor '{input_name}' not found in input table")
                
                tensor = self.input_tensors[input_name]
                self.compiler.load_batch(
                    name=alias,
                    hbm_addr=tensor.hbm_addr,
                    h=tensor.shape[0],
                    w=tensor.shape[1],
                    real_data_ratio=self.real_data_ratio,
                    vlen=64,
                    preload_len=4
                )
                self.tensor_aliases[alias] = alias
                continue
            
            # 解析 Load_Matrix: B = Load_Matrix(Y)
            match = re.match(r'(\w+)\s*=\s*Load_Matrix\s*\(\s*(\w+)\s*\)', line, re.IGNORECASE)
            if match:
                alias = match.group(1)  # B
                input_name = match.group(2)  # Y
                if input_name not in self.input_tensors:
                    raise ValueError(f"Input tensor '{input_name}' not found in input table")
                
                tensor = self.input_tensors[input_name]
                self.compiler.load_matrix(
                    name=alias,
                    hbm_addr=tensor.hbm_addr,
                    h=tensor.shape[0],
                    w=tensor.shape[1],
                    real_data_ratio=self.real_data_ratio
                )
                self.tensor_aliases[alias] = alias
                continue
            
            # 解析 TMM_Matmul: C = TMM_Matmul(A, B) 或 C = A @T B
            # TMM MatMul: Batch @ Matrix^T using M_TMM instruction
            # 支持三种格式：
            # 1. C = TMM_Matmul(A, B)
            # 2. C = A @T B
            # 3. C = A @ TransposeMatrix(B)
            # 注意：不要匹配 SubMatrix 相关的行（SubMatrix 由后面的匹配处理）
            match = None
            if 'SubMatrix' not in line:
                match = re.match(r'(\w+)\s*=\s*TMM_Matmul\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)', line, re.IGNORECASE)
                if not match:
                    match = re.match(r'(\w+)\s*=\s*(\w+)\s*@T\s*(\w+)', line, re.IGNORECASE)
                if not match:
                    match = re.match(r'(\w+)\s*=\s*(\w+)\s*@\s*TransposeMatrix\s*\(\s*(\w+)\s*\)', line, re.IGNORECASE)
            
            if match:
                result_alias = match.group(1)  # C
                act_alias = match.group(2)  # A (Batch)
                weight_alias = match.group(3)  # B (Matrix)
                
                # 检查张量是否存在
                if act_alias not in self.tensor_aliases:
                    raise ValueError(f"Tensor '{act_alias}' not found. Load it first.")
                if weight_alias not in self.tensor_aliases:
                    raise ValueError(f"Tensor '{weight_alias}' not found. Load it first.")
                
                act_name = self.tensor_aliases[act_alias]
                weight_name = self.tensor_aliases[weight_alias]
                
                # 检查类型：Batch @ Matrix^T
                act_info = self.compiler.symbol_table[act_name]
                weight_info = self.compiler.symbol_table[weight_name]
                
                if act_info.kind != "Batch":
                    raise ValueError(f"TMM_Matmul requires activation to be Batch. Got: {act_info.kind}")
                if weight_info.kind != "Matrix":
                    raise ValueError(f"TMM_Matmul requires weight to be Matrix. Got: {weight_info.kind}")
                
                # 执行 TMM MatMul: Batch @ Matrix^T
                self.compiler.tmm_matmul(
                    act_tensor=act_name,
                    weight_tensor=weight_name,
                    result_tensor=result_alias,
                    mlen=self.mlen,
                    blen=self.blen,
                    batch=act_info.shape[0],
                    hidden_size=act_info.shape[1],
                    out_features=weight_info.shape[0]  # Matrix shape is (out_features, hidden_size)
                )
                self.tensor_aliases[result_alias] = result_alias
                continue
            
            # 解析 FlashAttention: O = FlashAttention(Q, K, V)
            # Q: Batch (在 VRAM)
            # K, V: Matrix (在 HBM)
            match = re.match(r'(\w+)\s*=\s*FlashAttention\s*\(\s*(\w+)\s*,\s*(\w+)\s*,\s*(\w+)\s*\)', line, re.IGNORECASE)
            if match:
                result_alias = match.group(1)  # O
                q_alias = match.group(2)       # Q
                k_alias = match.group(3)       # K
                v_alias = match.group(4)       # V
                
                # 检查张量是否存在
                if q_alias not in self.tensor_aliases:
                    raise ValueError(f"Tensor '{q_alias}' not found. Load it first.")
                if k_alias not in self.tensor_aliases:
                    raise ValueError(f"Tensor '{k_alias}' not found. Load it first.")
                if v_alias not in self.tensor_aliases:
                    raise ValueError(f"Tensor '{v_alias}' not found. Load it first.")
                
                q_name = self.tensor_aliases[q_alias]
                k_name = self.tensor_aliases[k_alias]
                v_name = self.tensor_aliases[v_alias]
                
                # 获取 Q 的 shape 来确定 head_dim
                q_info = self.compiler.symbol_table[q_name]
                head_dim = q_info.shape[1]
                
                # 执行 FlashAttention（新接口）
                self.compiler.flash_attention(
                    q_tensor=q_name,
                    k_tensor=k_name,
                    v_tensor=v_name,
                    result_tensor=result_alias,
                    mlen=self.mlen,
                    blen=self.blen,
                    head_dim=head_dim,
                )
                self.tensor_aliases[result_alias] = result_alias
                continue
            
            # 解析 QK^T 乘法: S = Q @ Transpose(K) 或 S = MatMul(Q, Transpose(K))
            # 支持两种格式：
            # 1. S = Q @ Transpose(K)
            # 2. S = MatMul(Q, Transpose(K))
            match = re.match(r'(\w+)\s*=\s*(\w+)\s*@\s*Transpose\s*\(\s*(\w+)\s*\)', line, re.IGNORECASE)
            if not match:
                match = re.match(r'(\w+)\s*=\s*MatMul\s*\(\s*(\w+)\s*,\s*Transpose\s*\(\s*(\w+)\s*\)\s*\)', line, re.IGNORECASE)
            
            if match:
                result_alias = match.group(1)  # S
                q_alias = match.group(2)  # Q
                k_alias = match.group(3)  # K
                
                # 检查张量是否存在
                if q_alias not in self.tensor_aliases:
                    raise ValueError(f"Tensor '{q_alias}' not found. Load it first.")
                if k_alias not in self.tensor_aliases:
                    raise ValueError(f"Tensor '{k_alias}' not found. Load it first.")
                
                q_name = self.tensor_aliases[q_alias]
                k_name = self.tensor_aliases[k_alias]
                
                # 检查类型：Q 和 K 都应该是 Batch
                q_info = self.compiler.symbol_table[q_name]
                k_info = self.compiler.symbol_table[k_name]
                
                if q_info.kind != "Batch" or k_info.kind != "Batch":
                    raise ValueError(f"QK^T multiply requires both Q and K to be Batch. Got Q: {q_info.kind}, K: {k_info.kind}")
                
                # 执行 QK^T multiply
                self.compiler.qkt_multiply(
                    q_tensor=q_name,
                    k_tensor=k_name,
                    result_tensor=result_alias,
                    mlen=self.mlen,
                    blen=self.blen,
                    d=None  # 自动推断
                )
                self.tensor_aliases[result_alias] = result_alias
                continue
            
            # 解析矩阵乘法: C = A @ B 或 C = B @ A
            # 注意：不要匹配 SubMatrix 相关的行（SubMatrix 由后面的匹配处理）
            match = None
            if 'SubMatrix' not in line:
                match = re.match(r'(\w+)\s*=\s*(\w+)\s*@\s*(\w+)', line)
            if match:
                result_alias = match.group(1)  # C
                left_alias = match.group(2)  # A 或 B
                right_alias = match.group(3)  # B 或 A
                
                # 检查张量是否存在
                if left_alias not in self.tensor_aliases:
                    raise ValueError(f"Tensor '{left_alias}' not found. Load it first.")
                if right_alias not in self.tensor_aliases:
                    raise ValueError(f"Tensor '{right_alias}' not found. Load it first.")
                
                left_name = self.tensor_aliases[left_alias]
                right_name = self.tensor_aliases[right_alias]
                
                # 检查类型：Batch @ Matrix
                left_info = self.compiler.symbol_table[left_name]
                right_info = self.compiler.symbol_table[right_name]
                
                if left_info.kind == "Batch" and right_info.kind == "Matrix":
                    # A @ B: Batch @ Matrix
                    act_tensor = left_name
                    weight_tensor = right_name
                elif left_info.kind == "Matrix" and right_info.kind == "Batch":
                    # B @ A: Matrix @ Batch (需要转置或特殊处理)
                    raise ValueError("Matrix @ Batch is not supported. Use Batch @ Matrix.")
                else:
                    raise ValueError(f"Invalid operation: {left_info.kind} @ {right_info.kind}")
                
                # 执行 projection
                self.compiler.projection(
                    act_tensor=act_tensor,
                    weight_tensor=weight_tensor,
                    result_tensor=result_alias,
                    mlen=self.mlen,
                    blen=self.blen,
                    rope_enabled=False
                )
                self.tensor_aliases[result_alias] = result_alias
                continue
            
            # 解析 Store: Store(A, a) - 将 VRAM 中的 A 存储到 HBM，命名为 a
            # 支持两种格式：
            # 1. Store(A, a)
            # 2. a = Store(A)
            match = re.match(r'Store\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)', line, re.IGNORECASE)
            if not match:
                match = re.match(r'(\w+)\s*=\s*Store\s*\(\s*(\w+)\s*\)', line, re.IGNORECASE)
                if match:
                    # 交换顺序：a = Store(A) -> store_name=a, tensor_alias=A
                    store_name = match.group(1)
                    tensor_alias = match.group(2)
                    match = True  # 标记匹配成功
                else:
                    match = None
            else:
                # Store(A, a) -> tensor_alias=A, store_name=a
                tensor_alias = match.group(1)
                store_name = match.group(2)
            
            if match:
                # 检查 tensor 是否存在
                if tensor_alias not in self.tensor_aliases:
                    raise ValueError(f"Tensor '{tensor_alias}' not found. Load it first before storing.")
                
                tensor_name = self.tensor_aliases[tensor_alias]
                tensor_info = self.compiler.symbol_table[tensor_name]
                
                # 类型检查：只有 Batch 类型可以 Store
                if tensor_info.kind != "Batch":
                    raise ValueError(f"Store requires Batch tensor. Got: {tensor_info.kind}")
                
                # 计算 HBM 地址（如果还没有）
                # 使用 next_hbm_addr 动态分配
                if self.next_hbm_addr == 0:
                    # 找到现有 input tensors 中最大的地址
                    for t in self.input_tensors.values():
                        end_addr = t.hbm_addr + t.hbm_size
                        self.next_hbm_addr = max(self.next_hbm_addr, end_addr)
                    # 对齐到 64 字节
                    self.next_hbm_addr = ((self.next_hbm_addr + 63) // 64) * 64
                
                hbm_addr = self.next_hbm_addr
                tensor_size = tensor_info.shape[0] * tensor_info.shape[1]
                hbm_size = int(tensor_size * self.real_data_ratio)
                
                # 更新 next_hbm_addr
                self.next_hbm_addr = ((hbm_addr + hbm_size + 63) // 64) * 64
                
                # 调用 store_to_hbm 生成 ISA
                self.compiler.store_to_hbm(
                    tensor_name=tensor_name,
                    hbm_addr=hbm_addr,
                    vlen=64
                )
                
                # 记录存储信息
                self.stored_tensors[store_name] = tensor_alias
                
                # 注册到 input_tensors（这样未来可以 Load）
                self.input_tensors[store_name] = InputTensor(
                    name=store_name,
                    hbm_addr=hbm_addr,
                    hbm_size=hbm_size,
                    shape=tensor_info.shape,
                    dtype=tensor_info.dtype,
                    real_data_ratio=self.real_data_ratio
                )
                
                # 也添加到 tensor_aliases，这样未来可以引用
                self.tensor_aliases[store_name] = store_name
                continue
            
            # 解析 Register_SubMatrix: W = Register_SubMatrix(weight, 256, 256)
            # 或 Register_SubMatrix(W, weight)  (从 input_tensors 获取形状)
            match = re.match(r'(\w+)\s*=\s*Register_SubMatrix\s*\(\s*(\w+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)', line, re.IGNORECASE)
            if match:
                alias = match.group(1)  # W
                input_name = match.group(2)  # weight
                h = int(match.group(3))
                w = int(match.group(4))
                
                if input_name not in self.input_tensors:
                    raise ValueError(f"Input tensor '{input_name}' not found for Register_SubMatrix")
                
                tensor = self.input_tensors[input_name]
                
                # 注册子矩阵
                self.compiler.register_sub_matrix(
                    name=alias,
                    hbm_addr=tensor.hbm_addr,
                    h=h,
                    w=w,
                    real_data_ratio=self.real_data_ratio
                )
                self.sub_matrices[alias] = input_name
                self.tensor_aliases[alias] = alias
                self.loaded_sub_rows[alias] = []
                continue
            
            # 简化版：W = Register_SubMatrix(weight) - 从 input_tensors 获取形状
            match = re.match(r'(\w+)\s*=\s*Register_SubMatrix\s*\(\s*(\w+)\s*\)', line, re.IGNORECASE)
            if match:
                alias = match.group(1)
                input_name = match.group(2)
                
                if input_name not in self.input_tensors:
                    raise ValueError(f"Input tensor '{input_name}' not found for Register_SubMatrix")
                
                tensor = self.input_tensors[input_name]
                h, w = tensor.shape
                
                self.compiler.register_sub_matrix(
                    name=alias,
                    hbm_addr=tensor.hbm_addr,
                    h=h,
                    w=w,
                    real_data_ratio=self.real_data_ratio
                )
                self.sub_matrices[alias] = input_name
                self.tensor_aliases[alias] = alias
                self.loaded_sub_cols[alias] = []  # 记录已加载的列子块
                self.loaded_sub_rows[alias] = []  # 记录已加载的行子块
                continue
            
            # 解析 Load_SubMatrix: Load_SubMatrix(W, 1) 或 Load_SubMatrix(W, 1, :)
            # 加载一整列子块（用于 sub_projection: A @ W[:, col_idx]）
            match = re.match(r'Load_SubMatrix\s*\(\s*(\w+)\s*,\s*(\d+)\s*(?:,\s*:)?\s*\)', line, re.IGNORECASE)
            if match:
                mat_alias = match.group(1)  # W
                col_idx = int(match.group(2))  # 列索引
                
                if mat_alias not in self.sub_matrices:
                    raise ValueError(f"SubMatrix '{mat_alias}' not registered. Use Register_SubMatrix first.")
                
                # 加载一整列子块到 MRAM（用于 sub_projection）
                self.compiler.load_sub_matrix_col(
                    name=mat_alias,
                    col_idx=col_idx
                )
                self.loaded_sub_cols[mat_alias].append(col_idx)
                continue
            
            # 解析 Reset_MRAM: Reset_MRAM()
            # 重置 MRAM 分配器，用于 for loop 中需要重新加载子块的场景
            match = re.match(r'Reset_MRAM\s*\(\s*\)', line, re.IGNORECASE)
            if match:
                self.compiler.reset_mram()
                # 清空已加载的子块记录
                for alias in self.loaded_sub_cols:
                    self.loaded_sub_cols[alias] = []
                for alias in self.loaded_sub_rows:
                    self.loaded_sub_rows[alias] = []
                continue
            
            # =========================================================================
            # 展开版 Flash Attention 操作
            # =========================================================================
            
            # 解析 Init_Online_Softmax: Init_Online_Softmax(q_idx, O)
            match = re.match(r'Init_Online_Softmax\s*\(\s*(\d+)\s*,\s*(\w+)\s*\)', line, re.IGNORECASE)
            if match:
                q_idx = int(match.group(1))
                o_alias = match.group(2)
                
                if o_alias not in self.tensor_aliases:
                    raise ValueError(f"O matrix '{o_alias}' not allocated. Use Allocate_VRAMMatrix first.")
                
                o_info = self.compiler.symbol_table[o_alias]
                seq_len, head_dim = o_info.shape
                
                self.compiler.init_online_softmax(
                    q_idx=q_idx,
                    o_matrix=o_alias,
                    seq_len=seq_len,
                    head_dim=head_dim,
                )
                continue
            
            # 解析 OnlineSoftmax_Block: OnlineSoftmax_Block(S_block, scale)
            match = re.match(r'OnlineSoftmax_Block\s*\(\s*(\w+)\s*,\s*([^)]+)\s*\)', line, re.IGNORECASE)
            if match:
                s_block_alias = match.group(1)
                scale_str = match.group(2).strip()
                
                if s_block_alias not in self.tensor_aliases:
                    raise ValueError(f"S block '{s_block_alias}' not found.")
                
                try:
                    scale = float(scale_str)
                except ValueError:
                    scale = eval(scale_str)  # 支持表达式如 1/sqrt(128)
                
                self.compiler.online_softmax_block(
                    s_block_matrix=s_block_alias,
                    scale=scale,
                )
                continue
            
            # 解析 Compute_PV: Compute_PV(S_block, V_sub, k_idx, PV)
            match = re.match(r'Compute_PV\s*\(\s*(\w+)\s*,\s*(\w+)\s*,\s*(\d+)\s*,\s*(\w+)\s*\)', line, re.IGNORECASE)
            if match:
                s_block_alias = match.group(1)
                v_sub_alias = match.group(2)
                k_idx = int(match.group(3))
                pv_alias = match.group(4)
                
                if s_block_alias not in self.tensor_aliases:
                    raise ValueError(f"S_block '{s_block_alias}' not allocated.")
                if v_sub_alias not in self.sub_matrices:
                    raise ValueError(f"V SubMatrix '{v_sub_alias}' not registered.")
                if pv_alias not in self.tensor_aliases:
                    raise ValueError(f"PV '{pv_alias}' not allocated.")
                
                pv_info = self.compiler.symbol_table[pv_alias]
                head_dim = pv_info.shape[1]
                
                self.compiler.compute_pv(
                    s_block_matrix=s_block_alias,
                    v_sub_matrix=v_sub_alias,
                    k_idx=k_idx,
                    pv_matrix=pv_alias,
                    head_dim=head_dim,
                )
                continue
            
            # 解析 Scale_O_Row: Scale_O_Row(O, q_idx)
            match = re.match(r'Scale_O_Row\s*\(\s*(\w+)\s*,\s*(\d+)\s*\)', line, re.IGNORECASE)
            if match:
                o_alias = match.group(1)
                q_idx = int(match.group(2))
                
                if o_alias not in self.tensor_aliases:
                    raise ValueError(f"O '{o_alias}' not allocated.")
                
                o_info = self.compiler.symbol_table[o_alias]
                seq_len, head_dim = o_info.shape
                
                self.compiler.scale_o_row(
                    o_matrix=o_alias,
                    q_idx=q_idx,
                    seq_len=seq_len,
                    head_dim=head_dim,
                )
                continue
            
            # 解析 VRAM_Add: VRAM_Add(dst, src, dst_row_offset)
            # 通用版本：dst[row_offset:] += src
            # dst_row_offset 可以是表达式如 "q_idx * 64"
            match = re.match(r'VRAM_Add\s*\(\s*(\w+)\s*,\s*(\w+)\s*,\s*([^)]+)\s*\)', line, re.IGNORECASE)
            if match:
                dst_alias = match.group(1)
                src_alias = match.group(2)
                dst_row_offset_expr = match.group(3).strip()
                # 计算表达式（支持简单的数学运算）
                dst_row_offset = int(eval(dst_row_offset_expr))
                
                if dst_alias not in self.tensor_aliases:
                    raise ValueError(f"Matrix '{dst_alias}' not allocated.")
                if src_alias not in self.tensor_aliases:
                    raise ValueError(f"Matrix '{src_alias}' not allocated.")
                
                self.compiler.vram_matrix_add(
                    dst_matrix=dst_alias,
                    src_matrix=src_alias,
                    dst_row_offset=dst_row_offset,
                    src_row_offset=0,
                )
                continue
            
            # 解析 Add_PV_to_O: Add_PV_to_O(O, PV, q_idx) - 使用通用 VRAM_Add
            match = re.match(r'Add_PV_to_O\s*\(\s*(\w+)\s*,\s*(\w+)\s*,\s*(\d+)\s*\)', line, re.IGNORECASE)
            if match:
                o_alias = match.group(1)
                pv_alias = match.group(2)
                q_idx = int(match.group(3))
                
                if o_alias not in self.tensor_aliases:
                    raise ValueError(f"O '{o_alias}' not allocated.")
                if pv_alias not in self.tensor_aliases:
                    raise ValueError(f"PV '{pv_alias}' not allocated.")
                
                # 使用通用的 vram_matrix_add
                dst_row_offset = q_idx * self.compiler.mlen
                self.compiler.vram_matrix_add(
                    dst_matrix=o_alias,
                    src_matrix=pv_alias,
                    dst_row_offset=dst_row_offset,
                    src_row_offset=0,
                )
                continue
            
            # 解析 Final_Scale_O: Final_Scale_O(q_idx, O)
            match = re.match(r'Final_Scale_O\s*\(\s*(\d+)\s*,\s*(\w+)\s*\)', line, re.IGNORECASE)
            if match:
                q_idx = int(match.group(1))
                o_alias = match.group(2)
                
                if o_alias not in self.tensor_aliases:
                    raise ValueError(f"O '{o_alias}' not allocated.")
                
                o_info = self.compiler.symbol_table[o_alias]
                seq_len, head_dim = o_info.shape
                
                self.compiler.final_scale_o(
                    q_idx=q_idx,
                    o_matrix=o_alias,
                    seq_len=seq_len,
                    head_dim=head_dim,
                )
                continue
            
            # 解析 Load_SubMatrix_Row: Load_SubMatrix_Row(W, 1)
            # 加载一整行子块（用于 sub_projection_T: A @ W[row_idx, :].T）
            match = re.match(r'Load_SubMatrix_Row\s*\(\s*(\w+)\s*,\s*(\d+)\s*\)', line, re.IGNORECASE)
            if match:
                mat_alias = match.group(1)  # W
                row_idx = int(match.group(2))  # 行索引
                
                if mat_alias not in self.sub_matrices:
                    raise ValueError(f"SubMatrix '{mat_alias}' not registered. Use Register_SubMatrix first.")
                
                # 加载一整行子块到 MRAM（用于 sub_projection_T）
                self.compiler.load_sub_matrix_row(
                    name=mat_alias,
                    row_idx=row_idx
                )
                self.loaded_sub_rows[mat_alias].append(row_idx)
                continue
            
            # 解析 Sub Projection: C = A @ SubMatrix(W, 1)
            # 计算 C = A @ W[:, col_idx*mlen:(col_idx+1)*mlen]
            match = re.match(r'(\w+)\s*=\s*(\w+)\s*@\s*SubMatrix\s*\(\s*(\w+)\s*,\s*(\d+)\s*\)', line, re.IGNORECASE)
            if match:
                result_alias = match.group(1)  # C
                act_alias = match.group(2)  # A
                mat_alias = match.group(3)  # W
                col_idx = int(match.group(4))  # 列索引
                
                if act_alias not in self.tensor_aliases:
                    raise ValueError(f"Tensor '{act_alias}' not found")
                if mat_alias not in self.sub_matrices:
                    raise ValueError(f"SubMatrix '{mat_alias}' not registered")
                if col_idx not in self.loaded_sub_cols.get(mat_alias, []):
                    raise ValueError(f"SubMatrix column {mat_alias}[:][{col_idx}] not loaded. Use Load_SubMatrix first.")
                
                act_name = self.tensor_aliases[act_alias]
                
                self.compiler.sub_projection(
                    act_tensor=act_name,
                    mat_name=mat_alias,
                    mat_col_idx=col_idx,
                    result_tensor=result_alias
                )
                self.tensor_aliases[result_alias] = result_alias
                continue
            
            # 解析 Sub Projection T: C = A @T SubMatrix(W, 1)
            # 计算 C = A @ W[row_idx*mlen:(row_idx+1)*mlen, :].T
            match = re.match(r'(\w+)\s*=\s*(\w+)\s*@T\s*SubMatrix\s*\(\s*(\w+)\s*,\s*(\d+)\s*\)', line, re.IGNORECASE)
            if match:
                result_alias = match.group(1)  # C
                act_alias = match.group(2)  # A
                mat_alias = match.group(3)  # W
                row_idx = int(match.group(4))  # 行索引
                
                if act_alias not in self.tensor_aliases:
                    raise ValueError(f"Tensor '{act_alias}' not found")
                if mat_alias not in self.sub_matrices:
                    raise ValueError(f"SubMatrix '{mat_alias}' not registered")
                # sub_projection_T 需要加载行子块
                if row_idx not in self.loaded_sub_rows.get(mat_alias, []):
                    raise ValueError(f"SubMatrix row {mat_alias}[{row_idx}][:] not loaded. Use Load_SubMatrix_Row first.")
                
                act_name = self.tensor_aliases[act_alias]
                
                self.compiler.sub_projection_T(
                    act_tensor=act_name,
                    mat_name=mat_alias,
                    mat_row_idx=row_idx,
                    result_tensor=result_alias
                )
                self.tensor_aliases[result_alias] = result_alias
                continue
            
            # ================================================================
            # VRAM 子矩阵解析
            # ================================================================
            
            # 解析 Register_VRAMSubMatrix: A_sub = Register_VRAMSubMatrix(A)
            match = re.match(r'(\w+)\s*=\s*Register_VRAMSubMatrix\s*\(\s*(\w+)\s*\)', line, re.IGNORECASE)
            if match:
                alias = match.group(1)  # A_sub
                source_alias = match.group(2)  # A
                
                if source_alias not in self.tensor_aliases:
                    raise ValueError(f"Source tensor '{source_alias}' not found")
                
                source_name = self.tensor_aliases[source_alias]
                
                # 注册 VRAM 子矩阵
                self.compiler.register_vram_sub_matrix(
                    name=alias,
                    source_tensor=source_name
                )
                self.vram_sub_matrices[alias] = source_name
                self.tensor_aliases[alias] = alias
                continue
            
            # 解析 Allocate_VRAMMatrix: C = Allocate_VRAMMatrix(rows, cols)
            # 分配一个大的 VRAM 矩阵，用于存储多个子块组合的结果
            match = re.match(r'(\w+)\s*=\s*Allocate_VRAMMatrix\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)', line, re.IGNORECASE)
            if match:
                mat_name = match.group(1)
                rows = int(match.group(2))
                cols = int(match.group(3))
                
                self.compiler.allocate_vram_matrix(
                    name=mat_name,
                    rows=rows,
                    cols=cols
                )
                self.tensor_aliases[mat_name] = mat_name
                
                # 记录分配的 VRAM 矩阵
                if not hasattr(self, 'allocated_vram_matrices'):
                    self.allocated_vram_matrices = {}
                self.allocated_vram_matrices[mat_name] = (rows, cols)
                continue
            
            # 解析 VRAM Sub Projection To: C[r][c] = VRAMSubMatrix(A, i) @ SubMatrix(W, j)
            # 将子块乘法结果写入到大矩阵的指定位置
            match = re.match(
                r'(\w+)\s*\[\s*(\d+)\s*\]\s*\[\s*(\d+)\s*\]\s*=\s*VRAMSubMatrix\s*\(\s*(\w+)\s*,\s*(\d+)\s*\)\s*@\s*SubMatrix\s*\(\s*(\w+)\s*,\s*(\d+)\s*\)',
                line, re.IGNORECASE
            )
            if match:
                target_mat = match.group(1)  # C
                target_row_idx = int(match.group(2))  # r
                target_col_idx = int(match.group(3))  # c
                vram_alias = match.group(4)  # A_sub
                vram_row_idx = int(match.group(5))  # VRAM 行索引
                mram_alias = match.group(6)  # W_sub
                mram_col_idx = int(match.group(7))  # MRAM 列索引
                
                if not hasattr(self, 'allocated_vram_matrices') or target_mat not in self.allocated_vram_matrices:
                    raise ValueError(f"Target matrix '{target_mat}' not allocated. Use Allocate_VRAMMatrix first.")
                if vram_alias not in self.vram_sub_matrices:
                    raise ValueError(f"VRAM SubMatrix '{vram_alias}' not registered. Use Register_VRAMSubMatrix first.")
                if mram_alias not in self.sub_matrices:
                    raise ValueError(f"MRAM SubMatrix '{mram_alias}' not registered")
                if mram_col_idx not in self.loaded_sub_cols.get(mram_alias, []):
                    raise ValueError(f"SubMatrix column {mram_alias}[:][{mram_col_idx}] not loaded. Use Load_SubMatrix first.")
                
                self.compiler.vram_sub_projection_to(
                    vram_mat_name=vram_alias,
                    vram_row_idx=vram_row_idx,
                    mram_mat_name=mram_alias,
                    mram_col_idx=mram_col_idx,
                    target_matrix=target_mat,
                    target_row_idx=target_row_idx,
                    target_col_idx=target_col_idx
                )
                continue
            
            # 解析 VRAM Sub Projection T To: C[r][c] = VRAMSubMatrix(A, i) @T SubMatrix(W, j)
            # 将转置子块乘法结果写入到大矩阵的指定位置（用于 Flash Attention 的 S = Q @ K^T）
            match = re.match(
                r'(\w+)\s*\[\s*(\d+)\s*\]\s*\[\s*(\d+)\s*\]\s*=\s*VRAMSubMatrix\s*\(\s*(\w+)\s*,\s*(\d+)\s*\)\s*@T\s*SubMatrix\s*\(\s*(\w+)\s*,\s*(\d+)\s*\)',
                line, re.IGNORECASE
            )
            if match:
                target_mat = match.group(1)  # C
                target_row_idx = int(match.group(2))  # r
                target_col_idx = int(match.group(3))  # c
                vram_alias = match.group(4)  # A_sub
                vram_row_idx = int(match.group(5))  # VRAM 行索引
                mram_alias = match.group(6)  # W_sub
                mram_row_idx = int(match.group(7))  # MRAM 行索引
                
                if not hasattr(self, 'allocated_vram_matrices') or target_mat not in self.allocated_vram_matrices:
                    raise ValueError(f"Target matrix '{target_mat}' not allocated. Use Allocate_VRAMMatrix first.")
                if vram_alias not in self.vram_sub_matrices:
                    raise ValueError(f"VRAM SubMatrix '{vram_alias}' not registered. Use Register_VRAMSubMatrix first.")
                if mram_alias not in self.sub_matrices:
                    raise ValueError(f"MRAM SubMatrix '{mram_alias}' not registered")
                if mram_row_idx not in self.loaded_sub_rows.get(mram_alias, []):
                    raise ValueError(f"SubMatrix row {mram_alias}[{mram_row_idx}][:] not loaded. Use Load_SubMatrix_Row first.")
                
                self.compiler.vram_sub_projection_T_to(
                    vram_mat_name=vram_alias,
                    vram_row_idx=vram_row_idx,
                    mram_mat_name=mram_alias,
                    mram_row_idx=mram_row_idx,
                    target_matrix=target_mat,
                    target_row_idx=target_row_idx,
                    target_col_idx=target_col_idx
                )
                continue

            # 解析 VRAM Sub Projection: C = VRAMSubMatrix(A, 1) @ SubMatrix(W, 2)
            # 计算 A[1][:] @ W[:][2] -> (mlen, mlen)
            match = re.match(
                r'(\w+)\s*=\s*VRAMSubMatrix\s*\(\s*(\w+)\s*,\s*(\d+)\s*\)\s*@\s*SubMatrix\s*\(\s*(\w+)\s*,\s*(\d+)\s*\)',
                line, re.IGNORECASE
            )
            if match:
                result_alias = match.group(1)  # C
                vram_alias = match.group(2)  # A_sub
                vram_row_idx = int(match.group(3))  # VRAM 行索引
                mram_alias = match.group(4)  # W_sub
                mram_col_idx = int(match.group(5))  # MRAM 列索引
                
                if vram_alias not in self.vram_sub_matrices:
                    raise ValueError(f"VRAM SubMatrix '{vram_alias}' not registered. Use Register_VRAMSubMatrix first.")
                if mram_alias not in self.sub_matrices:
                    raise ValueError(f"MRAM SubMatrix '{mram_alias}' not registered")
                if mram_col_idx not in self.loaded_sub_cols.get(mram_alias, []):
                    raise ValueError(f"SubMatrix column {mram_alias}[:][{mram_col_idx}] not loaded. Use Load_SubMatrix first.")
                
                self.compiler.vram_sub_projection(
                    vram_mat_name=vram_alias,
                    vram_row_idx=vram_row_idx,
                    mram_mat_name=mram_alias,
                    mram_col_idx=mram_col_idx,
                    result_tensor=result_alias
                )
                self.tensor_aliases[result_alias] = result_alias
                continue
            
            # 解析 VRAM Sub Projection T: C = VRAMSubMatrix(A, 1) @T SubMatrix(W, 1)
            # 计算 A[1][:] @ W[1][:]^T -> (mlen, mlen)
            match = re.match(
                r'(\w+)\s*=\s*VRAMSubMatrix\s*\(\s*(\w+)\s*,\s*(\d+)\s*\)\s*@T\s*SubMatrix\s*\(\s*(\w+)\s*,\s*(\d+)\s*\)',
                line, re.IGNORECASE
            )
            if match:
                result_alias = match.group(1)  # C
                vram_alias = match.group(2)  # A_sub
                vram_row_idx = int(match.group(3))  # VRAM 行索引
                mram_alias = match.group(4)  # W_sub
                mram_row_idx = int(match.group(5))  # MRAM 行索引
                
                if vram_alias not in self.vram_sub_matrices:
                    raise ValueError(f"VRAM SubMatrix '{vram_alias}' not registered. Use Register_VRAMSubMatrix first.")
                if mram_alias not in self.sub_matrices:
                    raise ValueError(f"MRAM SubMatrix '{mram_alias}' not registered")
                if mram_row_idx not in self.loaded_sub_rows.get(mram_alias, []):
                    raise ValueError(f"SubMatrix row {mram_alias}[{mram_row_idx}][:] not loaded. Use Load_SubMatrix_Row first.")
                
                self.compiler.vram_sub_projection_T(
                    vram_mat_name=vram_alias,
                    vram_row_idx=vram_row_idx,
                    mram_mat_name=mram_alias,
                    mram_row_idx=mram_row_idx,
                    result_tensor=result_alias
                )
                self.tensor_aliases[result_alias] = result_alias
                continue
            
            # ================================================================
            # Softmax 相关操作（用于 Flash Attention 展开版本）
            # ================================================================
            
            # 解析 Init: m = Init(-inf, mlen) 或 O = Init(0, rows, cols)
            match = re.match(r'(\w+)\s*=\s*Init\s*\(\s*([^,]+)\s*,\s*(\d+)\s*(?:,\s*(\d+))?\s*\)', line, re.IGNORECASE)
            if match:
                var_name = match.group(1)
                init_val = match.group(2).strip()
                dim1 = int(match.group(3))
                dim2 = match.group(4)
                
                # 解析初始值
                if init_val == '-inf':
                    init_value = float('-inf')
                else:
                    init_value = float(init_val)
                
                # 记录到 scalar_vectors（golden 计算时使用）
                if dim2 is None:
                    self.scalar_vectors[var_name] = ('vector', dim1, init_value)
                else:
                    self.scalar_vectors[var_name] = ('matrix', dim1, int(dim2), init_value)
                
                self.tensor_aliases[var_name] = var_name
                continue
            
            # 解析 RowMax: m = RowMax(S)
            match = re.match(r'(\w+)\s*=\s*RowMax\s*\(\s*(\w+)\s*\)', line, re.IGNORECASE)
            if match:
                result_name = match.group(1)
                source_name = match.group(2)
                self.scalar_vectors[result_name] = ('row_max', source_name)
                self.tensor_aliases[result_name] = result_name
                continue
            
            # 解析 RowSum: l = RowSum(S)
            match = re.match(r'(\w+)\s*=\s*RowSum\s*\(\s*(\w+)\s*\)', line, re.IGNORECASE)
            if match:
                result_name = match.group(1)
                source_name = match.group(2)
                self.scalar_vectors[result_name] = ('row_sum', source_name)
                self.tensor_aliases[result_name] = result_name
                continue
            
            # 解析 RowSoftmax: P = RowSoftmax(S)
            match = re.match(r'(\w+)\s*=\s*RowSoftmax\s*\(\s*(\w+)\s*\)', line, re.IGNORECASE)
            if match:
                result_name = match.group(1)
                source_name = match.group(2)
                
                # 检查源矩阵是否存在
                if source_name not in self.tensor_aliases:
                    raise ValueError(f"Source matrix '{source_name}' not found")
                
                # 调用 DeveloperCompiler 生成 ISA
                # 注意：这是 in-place 操作
                self.compiler.row_softmax_vram_matrix(matrix_name=source_name)
                
                # 记录别名（指向同一个 VRAM 地址，因为是 in-place）
                if result_name != source_name:
                    self.tensor_aliases[result_name] = source_name
                else:
                    self.tensor_aliases[result_name] = result_name
                
                # 记录到 scalar_vectors 用于 golden 计算
                self.scalar_vectors[result_name] = ('row_softmax', source_name)
                continue
            
            # 解析 Scale: S = Scale(S, factor)
            match = re.match(r'(\w+)\s*=\s*Scale\s*\(\s*(\w+)\s*,\s*([^)]+)\s*\)', line, re.IGNORECASE)
            if match:
                result_name = match.group(1)
                source_name = match.group(2)
                factor_str = match.group(3).strip()
                
                # 解析 scale factor
                try:
                    scale_factor = float(factor_str)
                except ValueError:
                    raise ValueError(f"Invalid scale factor: {factor_str}")
                
                # 检查源矩阵是否存在
                if source_name not in self.tensor_aliases:
                    raise ValueError(f"Source matrix '{source_name}' not found")
                
                # 调用 DeveloperCompiler 生成 ISA
                # 注意：scale_factor 需要在 fp_preload 中预加载
                self.compiler.scale_vram_matrix(
                    matrix_name=source_name,
                    scale_factor=scale_factor,
                    fp_scale_addr=1  # 假设 scale 在 FP SRAM[1]
                )
                
                # 如果 result_name != source_name，说明是赋值给新变量（但实际是 in-place）
                # 为了支持 S_scaled = Scale(S, factor)，我们需要在 symbol table 中添加别名
                if result_name != source_name:
                    # 添加别名（指向同一个 VRAM 地址）
                    self.tensor_aliases[result_name] = source_name
                else:
                    self.tensor_aliases[result_name] = result_name
                
                # 记录到 scalar_vectors 用于 golden 计算
                self.scalar_vectors[result_name] = ('scale', source_name, factor_str)
                continue
            
            # 解析 Max: m_new = Max(m_old, m_cur)
            match = re.match(r'(\w+)\s*=\s*Max\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)', line, re.IGNORECASE)
            if match:
                result_name = match.group(1)
                left_name = match.group(2)
                right_name = match.group(3)
                self.scalar_vectors[result_name] = ('max', left_name, right_name)
                self.tensor_aliases[result_name] = result_name
                continue
            
            # 解析 Exp: r = Exp(x)
            match = re.match(r'(\w+)\s*=\s*Exp\s*\(\s*(\w+)\s*\)', line, re.IGNORECASE)
            if match:
                result_name = match.group(1)
                source_name = match.group(2)
                self.scalar_vectors[result_name] = ('exp', source_name)
                self.tensor_aliases[result_name] = result_name
                continue
            
            # 解析 Sub: r = Sub(a, b)
            match = re.match(r'(\w+)\s*=\s*Sub\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)', line, re.IGNORECASE)
            if match:
                result_name = match.group(1)
                left_name = match.group(2)
                right_name = match.group(3)
                self.scalar_vectors[result_name] = ('sub', left_name, right_name)
                self.tensor_aliases[result_name] = result_name
                continue
            
            # 解析 Add: r = Add(a, b)
            match = re.match(r'(\w+)\s*=\s*Add\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)', line, re.IGNORECASE)
            if match:
                result_name = match.group(1)
                left_name = match.group(2)
                right_name = match.group(3)
                self.scalar_vectors[result_name] = ('add', left_name, right_name)
                self.tensor_aliases[result_name] = result_name
                continue
            
            # 解析 Mul: r = Mul(a, b) 或 r = a * b
            match = re.match(r'(\w+)\s*=\s*Mul\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)', line, re.IGNORECASE)
            if not match:
                match = re.match(r'(\w+)\s*=\s*(\w+)\s*\*\s*(\w+)', line)
            if match:
                result_name = match.group(1)
                left_name = match.group(2)
                right_name = match.group(3)
                self.scalar_vectors[result_name] = ('mul', left_name, right_name)
                self.tensor_aliases[result_name] = result_name
                continue
            
            # 解析 Div: r = Div(a, b) 或 r = a / b
            match = re.match(r'(\w+)\s*=\s*Div\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)', line, re.IGNORECASE)
            if not match:
                match = re.match(r'(\w+)\s*=\s*(\w+)\s*/\s*(\w+)', line)
            if match:
                result_name = match.group(1)
                left_name = match.group(2)
                right_name = match.group(3)
                self.scalar_vectors[result_name] = ('div', left_name, right_name)
                self.tensor_aliases[result_name] = result_name
                continue
            
            # 解析 Result: Result C
            match = re.match(r'Result\s+(\w+)', line, re.IGNORECASE)
            if match:
                result_tensor = match.group(1)
                if result_tensor not in self.tensor_aliases:
                    raise ValueError(f"Result tensor '{result_tensor}' not found")
                # 结果已经在 symbol table 中，这里只是标记
                continue
        
        if result_tensor is None:
            raise ValueError("No 'Result' statement found")
    
    def get_symbol_table(self):
        """获取符号表"""
        return self.compiler.symbol_table
    
    def print_symbol_table(self):
        """打印符号表"""
        self.compiler.print_symbol_table()


if __name__ == "__main__":
    # 示例用法
    compiler = SimpleCompiler()
    
    # 创建示例文件
    example_file = Path(__file__).parent / "example.txt"
    with open(example_file, 'w') as f:
        f.write("""# Input tensors
X: hbm_addr=0, hbm_size=1152, shape=(8, 128)
Y: hbm_addr=1152, hbm_size=36864, shape=(128, 256)

# Code
A = Load_Batch(X)
B = Load_Matrix(Y)
C = A @ B
Result C
""")
    
    # 解析并生成代码
    code = compiler.parse_file(str(example_file))
    print("Generated ISA Code:")
    print(code)
    print("\nSymbol Table:")
    compiler.print_symbol_table()

