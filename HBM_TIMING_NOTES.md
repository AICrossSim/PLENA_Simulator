# PLENA Simulator — HBM/VRAM 计时与 Prefetch 机制笔记

> 整理自 2026-05-25 的一次代码追查。回答这几个问题:
> 1. `H_PREFETCH` / `H_STORE` 的 cycle 到底怎么算的?为什么报告里说"HBM = 0 cycle"?
> 2. ramulator 在哪起作用?HBM 延迟为什么"不固定"?
> 3. tRCD / tCL / tRP 是什么,这个 build 里精确多少?
> 4. 一次 HBM→VRAM 传输的粒度(是不是只 64 字节?VRAM 一次只能一个 MLEN?)
> 5. prefetch 怎么从 ISA + C_LOOP 触发?
> 6. ramulator 要不要做依赖分析?它怎么给乱序并发的请求定序?
> 7. VRAM 侧有没有依赖处理?
> 8. flash attention 有没有软件流水(提前 prefetch 下一个 tile)?
>
> 所有代码行号、数值都是从源码 / nix store 的 ramulator2 源里查实的,不是估计。

---

## 0. 一句话总览

模拟器只有**一个虚拟时钟**(异步 `Executor`,最后 `executor.now()` = 总 latency)。
推进它有两个入口:

- **`cycle!(n)` 宏**(固定常量):matmul / vector / scalar / control 走这个。
- **`hbm.read()/write().await`**(可变):HBM 访存走这个,延迟由 **ramulator**(cycle-accurate DRAM 仿真)动态算出。

报告里"HBM DMA = 0 cycle"指的是 **`H_*` opcode 本身没有 `cycle!` 常量**;真实 HBM
延迟藏在它调用的 `transfer_mx_from_hbm → hbm.read().await` 里,由 ramulator 注入。
所以:

```
模拟器实测 latency (executor.now())  =  静态 ISA cycle (Σ cycle!)  +  ramulator 的 HBM 时延
```

→ **HBM 时延 = 实测 latency − 静态 cycle**(因为无指令级并行,两者串行累加,相减干净)。
`tools/manager/_cycle_analyze.py` 算的是**静态值**(算力侧下界,H_* 当 0),不含 HBM 时延。

---

## 1. 两个时钟入口

### 1.1 `cycle!` 宏 (transactional_emulator/src/main.rs:90)

```rust
macro_rules! cycle {
    ($cycle: expr) => {
        runtime::Executor::current().resolve_at(PERIOD * ($cycle as u32)).await;
    };
}
```

固定推进 n 个周期。各 opcode 的成本(active = analytic, DC_EN=1, MLEN=VLEN=1024):

| 类 | opcode | cost |
|---|---|---|
| matmul | M_MM / M_TMM / M_BMM / M_BTMM / M_MV / M_TMV | MLEN = 1024 |
| matmul drain / 批向量 | M_*_WO, M_BMV, M_BTMV | 1 |
| vector | V_ADD/SUB/MUL = 1;V_RECI = 2;V_RED_MAX = 4;V_RED_SUM = 8;V_EXP = 1 |
| v↔fpram | S_MAP_V_FP / S_MAP_FP_V | VLEN = 1024 |
| scalar fp / int | S_*_FP / S_*_INT(含 IntRAM LD/ST) | 1 |
| control | C_LOOP_START/END, C_SET_*_REG | 1 each |
| **HBM DMA** | **H_PREFETCH_M / H_PREFETCH_V / H_STORE_V** | **0 (无 cycle!)** ← 见下 |

SYSTOLIC_PROCESSING_OVERHEAD = 0,所以 matmul 纯 = MLEN。rd==0 的 fp/reduce 形是 no-op,成本 0。

### 1.2 HBM 访存 (main.rs 的 H_* arm,无 cycle!)

`H_PREFETCH_M`(main.rs:2066)、`H_PREFETCH_V`(:2118)、`H_STORE_V`(:2214) 这三个
arm 里**没有 `cycle!(...)`**。它们调:

```
H_PREFETCH_V → transfer_mx_from_hbm(...) → 对每个 64B chunk: hbm.read(addr).await
```

`self.hbm` 的类型(main.rs:2410):

```rust
WithStats( WithTiming( Ramulator::hbm2_preset(8), MemoryBacked ) )
```

- `MemoryBacked::read`(lib/memory/src/lib.rs:115):纯数组取下标,**0 延迟**,只管"内容是什么"。
- `WithTiming::read`(lib/memory/src/lib.rs:143):**先 `self.timing.read(addr).await`** —— 这一步走
  ramulator,**推进同一个 Executor 时钟**,延迟由 DRAM 状态机算;再读数据。

```rust
impl MemoryModel for WithTiming<T, M> {
    async fn read(&self, addr: u64) -> [u8; 64] {
        self.timing.read(addr).await;   // ← ramulator,推进时钟,可变延迟
        self.data.read(addr).await      // ← 取数据,0 延迟
    }
}
```

---

## 2. ramulator:HBM 延迟怎么算出来的(cycle-accurate)

ramulator 是逐周期(cycle-by-cycle)的真实 DRAM 仿真器(Ramulator 2.0,
arxiv 2308.11030),不是给一个固定延迟。

### 2.1 调用链

```
H_PREFETCH_V (0 cycle 本身)
 └ transfer_mx_from_hbm: 把一个 VLEN tile 拆成多个 64B chunk,并发(FuturesUnordered/join_all)发出
    └ hbm.read(addr) → Ramulator::read  (lib/ramulator/src/model.rs:164)
       └ ramulator_request(addr)  → ramulator2 C++ 内核 (pkgs/ramulator2/ramulator_capi.cc)
          └ DRAM 状态机逐拍 tick(),直到这次读完成 → 触发 callback → 唤醒那个 .await
```

- C API 封装:`ramulator_request` / `ramulator_tick` / `ramulator_period`
  (pkgs/ramulator2/ramulator_capi.cc)。真正的时序逻辑在 ramulator2 C++ 内核里(FFI)。
- Rust 侧 `read_transfer`(model.rs:122):发请求 → `fut.await` **一直挂着**,
  直到内核回调说"读完了"。`tick()`(model.rs:60)每拍 `next_instant += period`,
  `period = get_tCK()`。读完成需要多少拍 = 这次访存的 cycle。

### 2.2 决定"多少拍"的因素

一个请求落到 (channel, bank, row, col)。延迟取决于**行缓冲状态**(open-row policy):

- **行命中**:目标 row 已在 sense amplifier(行缓冲)里开着 → 只需 CAS。**最快**。
- **行空闲**:bank 没开任何行 → ACT(激活)→ CAS。
- **行冲突**:bank 开着别的行 → PRE(预充电)→ ACT → CAS。**最慢**。

外加 bank/channel 并行、inter-command 时序约束(tRRD/tFAW/tRAS/刷新 tRFC 等),
由 `FRFCFS`(First-Ready FCFS)调度器每拍挑一个时序就绪的命令发出。

**没有闭式公式** —— 它依赖访问历史(前面读了哪些行、哪些 bank 还忙),所以必须
cycle-accurate 仿真。这就是你导师说"由 ramulator 给、不固定"的本质。

---

## 3. tRCD / tCL / tRP 是什么,这个 build 精确多少

时序参数,单位 = DRAM 周期数:

| 参数 | 全称 | 物理含义 | 类比 |
|---|---|---|---|
| **tRCD** | RAS-to-CAS Delay | 激活(打开)一行 → 能读列(把整行拉进行缓冲) | 拉开抽屉 |
| **tCL** (=tCAS) | CAS Latency | 行已开 → 发列地址到数据出来 | 从开着的抽屉取一份 |
| **tRP** | Row Precharge | 关掉当前行(写回+预充电)才能开新行 | 关上抽屉 |
| tRAS | Row Active Time | 一行开着的最短时间 | |

### 3.1 这个 build 的精确数值

源(nix store 解包的 ramulator2,2025-05-07 版):
`/nix/store/jlk5h9hg4x8bphk15dgzdfy69bqfvhfc-source/src/dram/impl/HBM2.cpp:19`

preset = **`HBM2_2Gbps`**,`tCK_ps = 1000 ps = 1 ns/周期`(所以周期数 = 纳秒数):

```
name        rate  nBL nCL nRCDRD nRCDWR nRP nRAS nRC nWR ... nFAW nRFC ... tCK_ps
HBM2_2Gbps  2000   4   7    7      7     7   17   19  8  ...  15   160 ... 1000
```

| 参数 | 周期 | = 纳秒 |
|---|---|---|
| **nRCD (tRCD)** | 7 | 7 ns |
| **nCL (tCL)** | 7 | 7 ns |
| **nRP (tRP)** | 7 | 7 ns |
| nRAS | 17 | 17 ns |
| nRC(同 bank 两次 ACT 间隔) | 19 | 19 ns |
| nBL(burst 长度) | 4 | 4 ns |
| nFAW | 15 | 15 ns |
| nRFC(刷新) | 160 | 160 ns |
| **tCK** | — | **1 ns** |

### 3.2 三种读各花多少(精确)

| 情况 | 命令序列 | 周期 | 时间 |
|---|---|---|---|
| **行命中** | CAS (+nBL 传数据) | nCL = 7 (+4) | **~7–11 ns** |
| **行空闲** | ACT → CAS | nRCD + nCL = 7+7 | **~14 ns** |
| **行冲突** | PRE → ACT → CAS | nRP + nRCD + nCL = 7+7+7 | **~21 ns** |

命中 7ns vs 冲突 21ns,**差 3 倍** —— HBM 时间"不固定"的精确量化。
(注:channel 数 = 8,硬编码在 `hbm2_preset(8)`,见 §6。)

---

## 4. 一次传输的粒度

层层嵌套,别混淆:

```
一条 H_PREFETCH_V (amount = PREFETCH_V_AMOUNT)
  = 连续做 amount 次「读一个 VLEN tile → 写进 VRAM 一行」
      每个 VLEN tile (e.g. VLEN=1024 元素)
        在 transfer_mx_from_hbm 里拆成多个 64 字节 chunk
          每个 64B chunk = 一次 ramulator.read()  ← 64B 是 ramulator 的 burst 访问粒度
            多个 chunk 并发,吃 8 个 channel 的并行
```

要点:

- **VRAM 一次写 = 1 个 VLEN tile**(`vector_sram::write` 注释 "Clip to VLEN",
  lib/vector_sram/src/lib.rs:150)。**你说的"一次 VRAM 最多一个 MLEN"是对的。**
- **`PREFETCH_V_AMOUNT` 不是"一次读多宽",是"连续读几个 VLEN tile"**(批量/预取深度)。
  transfer 函数注释(main.rs:1100):`load_amount = how many reads;
  write_amount = how many sram writes; num_writes = load_amount / write_amount`。
  H_PREFETCH_V 传的是 `load_dim=VLEN, write_amount=1` → 每次写 1 个 VLEN,连写 amount 次。
- **64 不是传输总量,是 DRAM burst 颗粒**。VLEN=64 的小配置下恰好"一个 tile = 一个 64B chunk",
  所以小配置里会看到 64;大配置一个 tile 是很多个 64B chunk。

当前默认 amount(plena_settings.toml [BEHAVIOR]):
`HBM_M_Prefetch_Amount`、`HBM_V_Prefetch_Amount`、`HBM_V_Writeback_Amount`
(main.rs:63-65 读成 PREFETCH_M_AMOUNT / PREFETCH_V_AMOUNT / STORE_V_AMOUNT)。

---

## 5. toml 哪些参数影响 HBM 时间,怎么影响

分三类:

### 类 1:根本不进 ramulator —— DRAM 时序硬编码

ramulator 的时序全写死在 `lib/ramulator/src/preset.rs` 的 `hbm2_preset(8)`:

- channel 数 = **8**(硬编码实参,不是 toml)。
- DRAM 规格 = `HBM2_8Gb`,时序 = `HBM2_2Gbps`(硬编码字符串)。
- 所有 tCK/tRCD/tRP/tCL/tRAS/tRFC 在 ramulator2 上游 HBM2 表里,**toml 碰不到**。

→ **改 toml 不会改"一次行命中/冲突读各多少 ns"**。要改得改 preset.rs 硬编码,
或把 channel 数 / 时序参数改成从 toml 读。

### 类 2:toml 影响"读/写多少次、多少字节" → 间接影响总 HBM 时间

| toml 参数 | 怎么影响 |
|---|---|
| `HBM_M/V_Prefetch_Amount`, `HBM_V_Writeback_Amount` | 一条 prefetch/store 指令触发多少次 ramulator 访问 |
| `MLEN / VLEN` | 一个 tile 多大 → 拆成多少个 64B chunk |
| `HBM_WIDTH` | **只在 manager/compiler 侧**(binio.py:62 `row_unit = HBM_WIDTH//elem_bits`)决定 MX 打包对齐 → tensor 占多少字节 → 读多少 chunk。**不进 emulator 的 ramulator。** |
| `HBM_SIZE` | 只决定 alloc 内存大小(main.rs:2412),不影响时序 |

### 类 3:MX 精度 → 字节数 → chunk 数

`HBM_*_TYPE`(E4M3 / E8M0 / block=8)决定每元素多少 bit、scale 多少 → 同样元素数
占多少字节 → 读多少个 64B chunk。间接路径。

**小结**:单次访存"几拍"由硬编码 DRAM 时序决定(toml 改不动);toml 改的是
"发多少次访存、每次多大"(prefetch amount / tile 大小 / 打包 / 精度),间接决定
ramulator 被 tick 的总拍数。所以"实测 − 静态"这个差值会随 prefetch/tile/精度变,
但不会因想调 HBM 带宽/延迟而变。

---

## 6. ramulator 怎么给乱序并发请求定序(要不要依赖分析?)

**不需要软件层的依赖分析。** 模拟器把一条 prefetch 的多个 64B read 并发丢给 ramulator,
依赖在两个层面被**硬件语义自动**处理(用的是 `Generic` controller,
src/dram_controller/impl/generic_dram_controller.cpp):

### 6.1 数据依赖(RAW):write-forwarding (generic_dram_controller.cpp:131)

read 入队时,**即时扫 write_buffer 有没有同地址的未完成 write**:

```cpp
if (req.type_id == Read) {
    if (在 m_write_buffer 找到同 addr 的 write) {
        req.depart = m_clk + 1;   // 直接转发写的值,下个周期返回
        pending.push_back(req);
        return true;
    }
}
```

→ 不是事先建依赖图,是每个 read 入队时即时查表转发。保证读到正确值,与并发顺序无关。

### 6.2 资源/时序依赖:check_ready(line 346/360)

调度器从队列挑请求时,对每个候选调 `m_dram->check_ready(command, addr_vec)`:
检查这条命令在**当前 DRAM 状态**下时序是否允许(bank 是否在 ACT、行缓冲状态、
tRCD/tRP/tRAS 是否满足)。**没就绪的请求发不出去,留在队列里等。**

→ 两个落同 bank 的请求不会真并行,被 DRAM 时序模型**自然串行化**,不靠依赖图。

### 6.3 完成顺序:乱序完成 + 回调(serve_completed_reads, line 298)

每个 read 带自己的 callback,**可乱序完成**(open-row 命中的先回来),各自触发回调
唤醒 Rust 侧对应的 `.await`。Rust 侧 `transfer_mx_from_hbm` 用 `join_all` 等**全部**
chunk 回调完才算这次 transfer 结束。谁先谁后无所谓。

→ **本质:ramulator 是 cycle-accurate 硬件行为仿真,依赖是"硬件物理本来就这么串"
的副产物,不是单独算出来的依赖图。**

(我们的访问模式 prefetch 多为只读、weight/activation 跑前已写好,6.1 的 forwarding
很少触发;主要是 6.2 的 bank 时序串行 + open-row 命中决定延迟。)

---

## 7. VRAM 侧的依赖处理(和 HBM 不同,但确实有)

VRAM(vector_sram)靠 **每行一个 `Mutex<RowData>` 状态机:`Pending → Ready`**
(lib/vector_sram/src/lib.rs)。

```rust
enum RowData {
    Ready(Vec<u8>),                  // 数据已就位
    Pending(Receiver<QuantTensor>),  // 数据还在从 HBM 路上(异步通道)
}
```

### 7.1 写(prefetch 发起时):标 Pending

`write_delayed`(lib.rs:175):prefetch 一发,目标 VRAM 行立刻标成
`RowData::Pending(receiver)`,receiver 连到 HBM transfer。**prefetch 指令本身 0 cycle
立即返回**,HBM 搬运在后台跑。

### 7.2 读(后续 vector 指令消费时):读到 Pending 就 await

`read`(lib.rs:99):

```rust
if let RowData::Pending(receiver) = *guard {
    let tensor = receiver.await.unwrap();   // ★ 阻塞,直到 HBM 数据真的到了
    *guard = RowData::Ready(...);
}
```

→ **这是 VRAM 的 RAW 依赖处理**:读一个还没到的行会挂起,直到那次 HBM 传输完成。
**真正的 HBM 等待延迟推迟到第一条用这行数据的 vector 指令**,在它的 `receiver.await` 上付。

### 7.3 两层依赖分工

| | HBM (ramulator) | VRAM (vector_sram) |
|---|---|---|
| 谁做 | DRAM controller | 每行 `Mutex<RowData>` 状态机 |
| 处理什么 | bank/行/时序争用;HBM 内同地址读写转发 | prefetch(写) → 后续 vector(读) 的 RAW |
| 怎么做 | check_ready 每拍只发就绪命令 + write-forward | 读到 Pending 行就 `.await` 等数据到 |
| 延迟在哪计 | ramulator tick 推进时钟 | 等待时钟经 receiver 从 HBM transfer 传过来 |

数据流:`vector 指令 read 一行 → 行是 Pending → await receiver → receiver 由 HBM
transfer 的所有 64B chunk(经 ramulator 定序计时)join_all 完成后 fire → 行变 Ready
→ 指令拿到数据继续`。

---

## 8. prefetch 怎么从 ISA + C_LOOP 触发

prefetch 是**编译器为一次 DMA(把数据从 HBM 搬进 VRAM/MRAM)生成的循环**,每轮算地址 +
发一条 `H_PREFETCH`。真实 ISA(SSB_build_1/ir/linear_q/linear_q.isa)节选:

```
C_LOOP_START gp7, 4          ; dma_h2v 循环,变量 h2v_inner
  ... (一堆 S_LD/S_ADD/S_SLLI 算地址) ...
  H_PREFETCH_V gp11, gp10, a0, 1, 0
  S_ADDI_INT gp8, gp8, 1     ; idx += 1
C_LOOP_END gp7
```

→ prefetch **就嵌在 C_LOOP 循环体里**,这也是 `_cycle_analyze.py` 静态分析按 C_LOOP
trip 展开的依据(只不过 H_* 静态当 0,真实延迟在 §7 的 await 里)。

---

## 9. ⚠️ 关键发现:当前**没有软件流水**(prefetch 即时同步,HBM 延迟全暴露)

问题:flash attention 的 KV serial loop,每轮 DMA 一个 tile,在第一个 tile 还没算完时,
能不能提前 prefetch 第二个 tile(把 HBM 搬运和计算重叠)?

**答案:两条 flash 实现都不能,也都没做。**

### 9.1 v2 tilelang 路径 (SSB_build_1/ir/flash_attention/flash_attention.isa)

```
C_LOOP_START gp7, 8          ; KV-block serial 循环
  H_PREFETCH_M (197)         ; 取这一轮的 tile
  H_PREFETCH_M (218)
  M_BTMM (226)               ; ← 紧接着就用,算 Q@K
  M_BMM_WO (230)
  C_LOOP gp8,1024 ... V_RED_MAX/V_EXP/V_RED_SUM   ; softmax
  M_MM (521)                 ; 算 @V
C_LOOP_END gp7
```

prefetch 和它的消费者(M_BTMM)在**同一轮、相邻几行**,没有把"下一轮取数"提到
"本轮计算"之前。

### 9.2 手写 asm 模板 (compiler/asm_templates/flashattn)

同样:`qkt.py:66` 的 `H_PREFETCH_M` 紧接 `:73` 的 `M_BTMM`,同一轮相邻。
`overall.py` 主循环 `for kv_head: for k_seq: for q_seq: { qkt(含prefetch) → softmax → pv }`,
prefetch 在 body 内部。`reset_kv_prefetch`(reset.py:105)名字像预热预取,**实际只设
C_SET_SCALE_REG / C_SET_STRIDE_REG,没有任何 H_PREFETCH** —— 没有 prologue,没有流水。

### 9.3 后果

```
loop over tiles:
    H_PREFETCH 这个 tile       ← 发出,标 VRAM 行 Pending
    M_BTMM 用这个 tile         ← 立刻读 → 行还是 Pending → receiver.await 等满 HBM 延迟
    softmax / PV
```

- prefetch 在这里语义上是"**当前 tile 的同步 DMA**",不是经典"提前若干步的预取"。
- **每个 tile 的 HBM 延迟暴露并串行累加进总时间,没有被 compute 覆盖。**
- VRAM 的 Pending/Ready(§7)在这里唯一作用:同一轮里多个 64B chunk 并发吃 8-channel
  并行 + 合并 Q/K/V 几路 prefetch 的等待 —— **但不跨循环轮次**。

### 9.4 含义

- **真正的软件流水(double-buffering / prefetch-ahead)整个项目都没实现** —— 明确的未做优化。
  要做:循环体里把"取 tile i+1"挪到"算 tile i"之前,用两块 VRAM buffer 轮换,
  让 HBM 搬运和 systolic array 计算重叠。
- 对 cycle 分析:**"实测 − 静态" 的差值 ≈ 所有 tile 的 HBM 延迟之和(全暴露)**,
  因为没有任何隐藏。这个差值就是软件流水能省下来的上限。

---

## 10. 关键文件索引

| 文件 | 作用 |
|---|---|
| `transactional_emulator/src/main.rs` | cycle! 宏(:90)、H_* opcode arms(:2066+)、transfer_mx_from/to_hbm(:1088/:1324)、hbm 构造(:2410) |
| `transactional_emulator/lib/memory/src/lib.rs` | MemoryBacked(:115)、WithTiming::read(:143) |
| `transactional_emulator/lib/ramulator/src/model.rs` | Ramulator::read / read_transfer / tick(:60/:122/:164) |
| `transactional_emulator/lib/ramulator/src/preset.rs` | hbm2_preset(8) —— 硬编码 HBM2_2Gbps + 8 channel(:38) |
| `transactional_emulator/lib/vector_sram/src/lib.rs` | RowData Pending/Ready(:29)、read(:92)、write_delayed(:175) |
| `transactional_emulator/pkgs/ramulator2/ramulator_capi.cc` | C API:ramulator_request / tick / period |
| ramulator2 源(nix store):`.../src/dram/impl/HBM2.cpp:19` | HBM2_2Gbps 时序整数表(tRCD/tCL/tRP=7) |
| ramulator2 源:`.../src/dram_controller/impl/generic_dram_controller.cpp` | send/write-forward(:113/:131)、tick/check_ready(:171/:346) |
| `compiler/asm_templates/flashattn/{overall,qkt,reset}.py` | 手写 flash 模板(无软件流水) |
| `tools/manager/_cycle_analyze.py` | 静态 cycle 分析(H_* 当 0,不含 HBM 时延) |
| `KERNEL_REPORT.md` §2.12 | manager build 的 cycle 分析(静态) |

---

## 11. 待办 / 优化机会(从这次追查得出)

1. **软件流水**:对 flash 的 KV serial loop 做 double-buffering / prefetch-ahead,
   把 HBM 延迟藏进 matmul/softmax 计算时间。当前两条路径都没做(§9)。上限 = "实测−静态"差值。
2. **量化 HBM 时延**:对每个 kernel 跑实测 latency,减去静态值,得逐 kernel 的 HBM 时延表,
   补进 KERNEL_REPORT §2.12。(机制已通,只差脚本跑一遍。)
3. **HBM 配置参数化**:目前 channel 数(8)和 HBM2 时序硬编码在 preset.rs;若要扫不同
   HBM 配置,需改成从 toml 读。
