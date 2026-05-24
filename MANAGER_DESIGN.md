# PLENA Manager 设计方案

> 多 kernel 编排层。统一在 **HBM bin 字节镜像** 之上工作:每个 tensor 有
> 自己的 shape 与确定的 HBM 字节地址,可被独立 seek 写入 / 读出 / 比较;
> kernel 调用时把每个 tensor 的地址 override 喂给 compiler;启动时全局
> 收集并统一分配所有 kernel 的 FPRAM 常量。

---

## 0. 背景:为什么必须自己写,tilelang 给不了

tilelang 在本栈里只是一个 **PrimFunc 生成器**:`T.Kernel` / `T.copy` /
`T.gemm` → TVM TIR,到此为止。它假设运行在 PyTorch + CUDA 环境,所有
host 端 tensor 内存由 PyTorch 托管,kernel 之间靠 torch tensor 对象传递,
**没有"地址"概念**(地址是 CUDA driver 的事)。

而我们:
- 后端是 custom 的 `tilelang_tvm_compiler`(mid_ir → HLIR → AddressAlloc →
  PreIsa → MIR → ISA),tilelang 的 GPU codegen **完全没用**。
- 运行在 PLENA emulator 上,"内存"是一个 **扁平的 HBM bin 字节镜像**
  (`hbm_for_behave_sim.bin`)+ FPRAM 常量槽位。

所以 manager 没有任何现成可继承的东西。它就是要在 "TIR 之下、emulator
之上" 这一层做编排:**地址、bin 布局、MX 量化、constant 槽位、多 kernel
串联**。

---

## 1. 核心数据结构

### 1.1 `ManagedTensor`

一个 tensor 的全部元信息。地址是一等公民。

```
@dataclass
class ManagedTensor:
    name: str                 # 全局唯一,如 "MLP_hbm" / "Q_cache"
    shape: tuple[int, ...]    # 逻辑 shape(BSHD 等),元素个数 = prod(shape)
    hbm_addr: int             # 在 HBM bin 中的起始字节偏移(关键字段)
    role: Role                # WEIGHT | ACTIVATION | SCRATCH | IO
    data: torch.Tensor | None # .pt 数据(fp32/fp16,无损);scratch 可为 None

    # 派生(由 AddressAllocConfig 计算,不手存):
    #   num_elements = prod(shape)
    #   packed_bytes = _hbm_packed_byte_size(num_elements, cfg)
    #   scale_offset = elem 区 padded 到 hbm_row_width 后的字节数
```

- `data` 是 **.pt(torch.Tensor)** 而非 .txt —— 无损,直接 `torch.save/load`。
  写入 bin 前由 manager 做 MX 量化;比较时 golden 也走同一条 MX round-trip
  (见 [[reference_hbm_testbench_mx_roundtrip]]:所有 HBM tensor 都是
  MX-E4M3,golden 必须双边 round-trip)。
- `role`:
  - `WEIGHT` —— 只读,跨 kernel/跨 layer 复用,启动时一次写入。
  - `ACTIVATION` —— kernel 产出又被下游消费,producer.addr == consumer.addr。
  - `SCRATCH` —— 临时,**只占地址不需要预置数据**(data=None)。
  - `IO` —— 整个 pipeline 的最终输入/输出。

### 1.2 `HbmLayout`(全局 HBM 地址空间)

manager 自有的扁平地址空间,**不复用** address_alloc 内部的 bump cursor
(用户明确选择独立拥有)。

```
class HbmLayout:
    cfg: AddressAllocConfig          # 提供 _hbm_packed_byte_size 的打包参数
    base: int = cfg.hbm_base
    cursor: int                      # bump 分配用
    tensors: dict[str, ManagedTensor]

    def place(name, shape, role, data=None) -> ManagedTensor
        # 用 _hbm_packed_byte_size(prod(shape), cfg) 算 packed_bytes,
        # 从 cursor bump 出 hbm_addr,cursor += packed_bytes,登记。

    def pin(name, addr)              # 手动钉死某 tensor 地址(复用场景)
    def alias(new_name, existing)    # 复用:两个名字指向同一 hbm_addr
    def overrides() -> dict[str,int] # → AddressAllocConfig.hbm_address_overrides
```

**关键不变量:** packed_bytes 必须用 `address_alloc._hbm_packed_byte_size`
([address_alloc.py:127](compiler/tilelang_tvm_compiler/address_alloc.py#L127))
算,与 compiler 的地址推进**逐字节一致**,否则 ISA 的 HBM 地址会指向错
tensor 的字节(历史上 cosine=0 的根因)。

复用("有的 tensor 没有 weight 功能可以合并"):通过 `alias` 让多个逻辑
名共享一个 hbm_addr,而不是各占一段 —— 这正是用户要的"很多 tensor 可
以复用,只留重要 weight"。

---

## 2. Manager API

### 2.1 seek 写入(bin 层,任意地址)

替代 `map_mx_data_to_hbm_for_behave_sim` 的 **append-only** 局限
([memory_map.py:114](tools/memory_mapping/memory_map.py#L114) 只能顺序
追加,删一个 tensor 就错位 —— 这是历史脆弱性的来源)。

```
def write_tensor(self, t: ManagedTensor, bin_path: str):
    """把 t.data 量化成 MX 字节,seek 到 t.hbm_addr 覆盖写入。
       单个 tensor 独立可重写,不影响相邻 tensor。"""
    # 1. quantize_tensor(t.data) -> (blocks, scales)   (rand_gen.py:133)
    # 2. 按 layout 拼字节:[elem padded→row_width][scale padded→row_width]
    #                     [total padded→64]          (与 _hbm_packed_byte_size 同布局)
    # 3. open(bin_path,'r+b'); f.seek(t.hbm_addr); f.write(payload)
```

- bin 是 **"模拟前的最终产物"**;manager 直接操作字节,先用 0 把整个
  HBM 区初始化到 cursor 上界,再逐 tensor seek 写。
- WEIGHT/IO/ACTIVATION 初值有 data → 写;SCRATCH(data=None)→ 跳过,
  只保留地址段(留零)。

> 需要新写一个 seek-based packer(复制
> `map_mx_data_to_hbm_for_behave_sim` 的字节布局逻辑,但用 `seek` 而非
> append)。这是 manager 唯一必须新增的"底层"代码。

### 2.2 读出(任意偏移提取任意 tensor)

直接复用
[check_mem.py:586 `read_hbm_bin_file_as_array`](transactional_emulator/tools/check_mem.py#L586):

```
def read_tensor(self, name, bin_path) -> np.ndarray:
    t = self.tensors[name]
    return read_hbm_bin_file_as_array(
        bin_path,
        exp_width   = cfg.hbm_elem_bits 的 E,   # E4M3 → exp=4, man=3
        man_width   = ...,
        start_byte_offset = t.hbm_addr,
        num_elements      = prod(t.shape),
        element_bytes     = cfg.hbm_elem_bits // 8,
        scale_width       = cfg.hbm_scale_bits,
        block_size        = cfg.hbm_block_size,
        scale_offset      = t.elem_region_padded_bytes,  # = packed elem 区长度
    ).reshape(t.shape)
```

这正是 view_mem 的能力(从 bin 任意位置提取),manager 把它包装成
"按名字取 tensor"。

### 2.3 .pt 比较(强制 HBM compare)

VRAM compare 已在 view_mem 关闭、强制 HBM compare(见 session 记录)。
manager 统一走:

```
def compare(self, name, bin_path, golden_pt) -> dict:
    """读出 bin 里的 tensor,与 golden(.pt)比较。
       golden 先做 MX round-trip(双边),再算 cos / NRMSE。"""
    # 复用 check_mem.compare_hbm_with_golden(...) (line 736)
    # 返回 {cosine, nrmse, snr};用户只看 cos 和 nrmse 正常即可
    #   (见 [[project_plena_kernel_results]] 的判读口径)
```

- golden 为 **.pt 无损**,不是 txt。
- SNR 已有 domain-error guard(check_mem ~846)。
- 用户口径:**只看 cos 和 nrmse 正常**,不纠结 relative error。

### 2.4 kernel 调用(地址 override 喂给 compiler)

```
def run_kernel(self, prim_func, name, *, inputs, outputs, build_dir):
    """编译 + 跑一个 kernel。inputs/outputs 是 ManagedTensor 名字列表。
       把它们的 hbm_addr 作为 override 传给 compiler,使 ISA 的 HBM
       地址精确落在 manager 规划的字节上。"""
    addr_cfg = AddressAllocConfig(
        mlen=..., blen=..., hlen=...,
        hbm_address_overrides = { n: self.tensors[n].hbm_addr
                                  for n in inputs + outputs + scratch },
        fpram_address_overrides = self.const_pool.overrides_for(name),  # 见 §3
    )
    ck = compile_kernel(prim_func, target=..., name=name,
                        addr_config_override=addr_cfg, use_v2=True)
    # ck.isa_text → 写 generated code;跑 emulator;输出读回供 compare
```

- `hbm_address_overrides` 是现成机制
  ([address_alloc.py:115](compiler/tilelang_tvm_compiler/address_alloc.py#L115)):
  名字在 dict 里就钉死,不在就 bump。manager 全程钉死,**不让 compiler
  自己 bump**,确保 producer.out_addr == consumer.in_addr。
- SCRATCH tensor 也要进 override(否则 compiler bump 出来会撞 manager 的
  地址)—— 历史教训:scratch 必须参与地址规划。

### 2.5 编排一整条 pipeline

```
def run_pipeline(self, steps: list[KernelStep], bin_path):
    # 0. 全局收集 + 分配 constant(§3),全局规划 HBM 地址(§1.2)
    # 1. 把所有 WEIGHT/IO 初值 seek 写进 bin;scratch 留零
    # 2. 写 fp_sram.bin(全局 const pool,一次)
    # 3. for step in steps: run_kernel(...);可选 compare 中间产物
    # 4. compare 最终 output 与 golden
```

中间产物可单独 compare(§2.3),定位某个 kernel 是否写坏了下游输入
(历史上 gelu 链 cosine=0.14 就是靠"读 bin 看 MLP_hbm 被写成常量"定位的)。

### 2.6 目录布局:`managerbuild/`

bin **不**放进 `build/`。manager 自己起一个 `managerbuild/`,内部按
"刷新节奏"分两部分:

```
managerbuild/
├── hbm_bin/                  # 持久,不每次自动刷新
│   ├── hbm_for_behave_sim.bin    # 模拟前的最终产物,seek 写入,跨 run 保留
│   ├── fp_sram.bin               # 全局 const pool,只在 ConstPool 变化时重写
│   └── layout.json               # {name: {shape, hbm_addr, role, scale_offset}}
│                                 #   HbmLayout 的持久记录,供下次 seek 写/读对齐
└── ir/                       # 自动刷新,每次 kernel 调用重新生成
    └── <kernel_name>/
        ├── <kernel>.isa.txt          # generated code
        ├── <kernel>.hlir.txt
        ├── <kernel>.lowir.txt        # 见 [[reference_lowir_report]]
        ├── midir_dump/               # compile_kernel(midir_dump_dir=) 落点
        └── compare_report.txt        # 该 kernel 的 cos/nrmse
```

两部分的刷新语义:

- **`hbm_bin/`(持久)**:bin 是"模拟前的最终产物"。WEIGHT 一次写入后
  跨多次 run 保留,**不每次刷新**;只有当某个 tensor 的 .pt data 变化、
  或某个 kernel 写了它,才 seek 覆盖那一段。`layout.json` 让下次启动能
  直接对齐已有地址,不必重新规划。这正是 seek-based write(§2.1)的意义
  —— 单 tensor 独立可重写,不动相邻字节。

- **`ir/`(自动刷新)**:每次 `run_kernel`(§2.4)调用都重新生成,按
  kernel 名分子目录,旧的直接覆盖。这是 `compile_kernel` 的
  `midir_dump_dir` + isa/hlir/lowir dump 的统一落点,以及该 kernel 的
  compare report。

manager 持有这两个根:

```
class Manager:
    build_root: str = "managerbuild"
    @property
    def hbm_dir(self):  return f"{self.build_root}/hbm_bin"   # 持久
    @property
    def ir_dir(self):   return f"{self.build_root}/ir"        # 每次刷新

    def ir_dir_for(self, kernel_name):
        d = f"{self.ir_dir}/{kernel_name}"
        # run_kernel 进来时清空该 kernel 的 ir 子目录再重写
        return d
```

注意:emulator 自己跑出来的临时产物(中间 .bin/状态)仍可留在它原本的
工作目录;`managerbuild/` 只收 **(a) 持久的 HBM/FPRAM 镜像 + layout**,
**(b) 每个 kernel 调用产生的 IR/report**。两者物理隔离,清理 `ir/` 不会
碰到来之不易的 `hbm_bin/`。

---

## 3. Constant 全局收集 + 统一分配(用户选定策略)

### 3.1 问题

每个 kernel 有自己 hoist 出来的 FPRAM 浮点常量
([hoist_float_constants.py:364](compiler/tilelang_tvm_compiler/frontend/passes/hoist_float_constants.py#L364):
`run(func, root_block_name=...)` 把 `{name: value}` 写进
`PrimFunc.attrs["plena.hoisted_constants"]`)。

链式 kernel 里若各自从 FPRAM_BASE 起算,kernel N 的 scratch 会盖掉 kernel
N+1 预置的 const → 全零输出(见
[[feedback_chained_fpram_scratch_const_overlap]])。

### 3.2 manager 的 ConstPool

```
class ConstPool:
    base: int = cfg.fpram_base   # FPRAM_USER_BASE = 32 (address_alloc.py:81)
    slots: dict[float|key, int]  # 去重:相同常量值共享一个槽
    per_kernel: dict[str, dict[name,int]]  # kernel → {const_name: fpram_addr}

    def collect(self, prim_funcs: dict[name, PrimFunc]):
        """启动时遍历所有 kernel,跑 hoist pass(或读已 stamp 的 attrs),
           把每个 kernel 的 hoisted_constants 汇总。"""

    def allocate(self):
        """统一从 base 往上分配。相同 value 去重共享槽,
           不同 value 各占一槽。所有 kernel 的 scratch 一律放在
           const 上界之上(固定 FPRAM_SCRATCH_BASE),避免互相覆盖。"""

    def overrides_for(self, kernel_name) -> dict[str,int]:
        """→ AddressAllocConfig.fpram_address_overrides,
           告诉 compiler 这个 kernel 的每个 const 落在哪个 FPRAM 槽。"""

    def write_fp_sram(self, path):
        """把全局 const pool 按槽写 fp_sram.bin(float16 by slot)。
           整条 pipeline 只写一次,所有 kernel 共享同一份 preload。"""
```

- **全局收集**:启动时一次性扫所有 kernel。
- **统一分配**:一个全局 FPRAM 地址空间,相同常量去重共享槽。
- **scratch 隔离**:所有 kernel 的 FPRAM scratch 统一放在 const 上界之上
  (固定 SCRATCH_BASE),从根上消除链式覆盖(对应已有的 feedback)。
- fp_sram.bin **只写一次**,全 pipeline 共享。

---

## 4. 与现有 API 的对接表

| manager 职责 | 复用的现有 API | 文件 |
|---|---|---|
| 几何/精度真相源 | 读 `[BEHAVIOR]` section(见 §6) | plena_settings.toml |
| MX 量化 | `quantize_tensor(tensor)` → (blocks, scales) | rand_gen.py:133 |
| packed 字节数 | `_hbm_packed_byte_size(n, cfg)` | address_alloc.py:127 |
| 字节布局参考 | `map_mx_data_to_hbm_for_behave_sim`(append,**仅参考布局**) | memory_map.py:114 |
| **seek 写入** | ⚠️ **需新增**(复制布局逻辑,用 seek 不 append) | manager 新代码 |
| 任意偏移读出 | `read_hbm_bin_file_as_array(...)` | check_mem.py:586 |
| HBM 比较 | `compare_hbm_with_golden(...)` | check_mem.py:736 |
| 强制 HBM compare | 已在 view_mem 关闭 VRAM compare | view_mem.py ~243 |
| kernel 编译 + 地址 override | `compile_kernel(..., addr_config_override=)` | pipeline.py:101 |
| 地址 override 字段 | `AddressAllocConfig.hbm_address_overrides / fpram_address_overrides` | address_alloc.py:115 |
| const hoist | `hoist_float_constants.run(func, root_block_name=)` → attrs | hoist_float_constants.py:364 |
| emulator 环境 | `create_sim_env(...)` / `create_mem_for_sim(...)` | create_sim_env.py / build_env.py:67 |

唯一必须**新写**的底层代码:**seek-based MX packer**(§2.1)。其余全是
对现有 API 的编排封装。

---

## 5. 迁移路径

1. **新建** `tools/manager/` 包,放 `ManagedTensor` / `HbmLayout` /
   `ConstPool` / `Manager`。不动现有 kernel 单测。产物落
   `managerbuild/`(§2.6):`hbm_bin/` 持久、`ir/` 每次刷新,**不写进
   `build/`**。
2. **第一步只做 §2.1 + §2.2**(seek 写 + 读回),用一个已 PASS 的单
   kernel(如 gelu_min)验证:manager seek 写 input → 跑 → 读回 output
   → compare,结果与现有单测一致。这是地基,先证明 bin 层正确。
3. **加 §2.4 run_kernel**(地址 override),仍单 kernel,验证 override
   路径与 manager 规划一致。
4. **加 §3 ConstPool**,验证两个 kernel 串联时 const/scratch 不互相覆盖。
5. **加 §2.5 run_pipeline**,跑到 mlp 阶段的多 kernel 链(用户的目标:
   "给我一个跑到 mlp 阶段的")。
6. 旧的 SSB 框架已删除;manager 是它的系统化替代,不复活旧代码。

### 验证口径
- 每步只看 **cos / nrmse**(用户口径)。
- golden 一律 .pt 无损 + 双边 MX round-trip。
- 中间产物逐 tensor 可 compare(定位某 kernel 是否写坏下游)。

---

## 6. 与 plena_settings.toml 挂钩(唯一真相源)

manager 的**所有**几何 + 精度参数从
[plena_settings.toml](plena_settings.toml) 读,**不硬编码**。这同时回答了
原 §6.1(HBM 是否 E4M3):是 E4M3,但从 toml 读出来,不写死。

### 6.1 读哪个 section

只读 **`[BEHAVIOR]`**,不看 `[MODE].active`。`[MODE].active` 是死配置 ——
Python(load_sizes)和 Rust 都强制走 `[BEHAVIOR]` section(见 session
历史:golden/emulator config mismatch 的根因之一就是误信 `[MODE].active`)。

### 6.2 从 toml 映射到 `AddressAllocConfig`

```
def addr_cfg_from_toml(toml_path) -> AddressAllocConfig:
    s = load(toml_path)["BEHAVIOR"]
    cfg = s["CONFIG"]
    elem = s["PRECISION"]["HBM_M_WEIGHT_TYPE"]   # 代表性 MX 类型
    return AddressAllocConfig(
        mlen           = cfg["MLEN"]["value"],           # 64
        blen           = cfg["BLEN"]["value"],           # 4
        hlen           = cfg["HLEN"]["value"],           # 16
        hbm_row_width  = cfg["HBM_WIDTH"]["value"],      # 64  ⚠️ 不是默认 512
        hbm_elem_bits  = 1 + elem["ELEM"]["exponent"]
                           + elem["ELEM"]["mantissa"],   # 1+4+3 = 8
        hbm_scale_bits = elem["SCALE"]["exponent"]
                           + elem["SCALE"]["mantissa"],  # 8+0 = 8
        hbm_block_size = elem["block"],                  # 8
    )
```

⚠️ **真实坑**:`AddressAllocConfig` 的默认 `hbm_row_width=512` 是 ANALYTIC
的值;BEHAVIOR 的 `HBM_WIDTH=64`。manager **必须**用 toml 的值覆盖,否则
`_hbm_packed_byte_size` 的 row 对齐错,所有地址错位 → 读回乱码。

### 6.3 读出 / 量化参数也从同一处取

- `read_tensor`(§2.2)的 `exp_width=4, man_width=3, scale_width=8,
  block_size=8, element_bytes=1` —— 全部来自上面同一个 `HBM_M_WEIGHT_TYPE`
  (ELEM/SCALE/block),不再各处硬编码。
- VLEN(`cfg["VLEN"]["value"]=64`)等不进 AddressAllocConfig 的几何,manager
  另存一份 `BehaviorSettings` 供 kernel 调用时构造 target / load_sizes 用,
  与 emulator 读的同一份 toml 同源,避免 golden/emulator 几何不一致。
- manager 启动时把解析出的几何写进 `layout.json`,作为 `hbm_bin/` 持久镜像
  的"生成时几何"指纹;若下次 toml 几何变了,触发 `hbm_bin/` 全量重建
  (见 §7 开放问题的增量 bin 语义)。

> 单一真相源原则:emulator 读 toml,manager 也读**同一个** toml。任何几何/
> 精度只在 toml 改一处,两边自动一致。这从根上消除历史上 golden(HLEN=128)
> vs emulator(HLEN=16)那类 mismatch。

---

## 7. 其余开放问题(实现前需确认)

1. **alias 复用的安全性**:两个 ACTIVATION alias 同一地址时,manager 是否
   需要校验生命周期不重叠(producer 写完前不被另一个 alias 覆盖)?
3. **bin 初始化**:`hbm_bin/` 是持久的(§2.6),所以**不能**每次 pipeline
   全量 zero-fill —— 那会抹掉跨 run 保留的 WEIGHT。倾向:首次创建或
   `layout.json` 变化时才全量 zero-fill 重建;之后只 seek 覆盖被写的段、
   scratch 段每次 run 前单独清零。需确认这个增量语义。
