# validate MMDiT vs 真实 Open-Sora MMDiT — 完整对照

> 配置基准: Open-Sora v2, **256px / 16:9 / 129 帧**
> ([configs/diffusion/inference/256px.py](../../../../Open-Sora/configs/diffusion/inference/256px.py))

validate 验证的是**单个 block 的 kernel 链**:
- `_validate_block.py` = 1 个 **SingleStreamBlock**(真实 38 个)
- `_validate_double_block.py` = 1 个 **DoubleStreamBlock**(真实 19 个)

真实 MMDiT = **embedder + 19 double + 38 single + final_layer**,外层还有 50 步采样循环 / CFG / VAE。

差异分三类:
- **① 数学不等价** — 结果真的会不一样,要改才等于真实
- **② 结构缺失** — validate 根本没建模,做完整 MMDiT 要补(算力多 <1%)
- **③ 等价 / 有意适配** — 不影响正确性,有的是为硬件改写,**不用动**

---

## 一、数学不等价(结果会不同)

| # | 项 | 真实 Open-Sora | validate | 影响 |
|---|---|---|---|---|
| 1 | **mlp_ratio** | **4**(MLP 中间 12288) | **1**(中间 = HD) | ❗ MLP 算量只有真实 1/4;block 算量需 ×1.51 才到真实。**SSB / DSB 都缩水**(SSB 的 `Wm` 也是 HD×HD) |
| 2 | **RoPE 位置编码** | **3D 时空 [16,56,56]**,每 token 按 (t,h,w) 三轴旋转 | **1D 序列**,按 `pos=arange(S)` 单轴旋转 | ❗ 旋转角度不同 → attention 分数不同。只需换 cos/sin 表,kernel 不变 |
| 3 | **SSB 的 LayerNorm** | **无 affine**(`elementwise_affine=False`) | **误带 affine**(用了随机 `ln_sc`/`ln_bi`) | ❗ SSB 的 LN 多了仿射项(DSB 正确:喂 scale=1/bias=0) |

---

## 二、结构缺失(validate 没建,做完整 MMDiT 要补)

| # | 缺的 | 真实 | validate 现状 | 算力 |
|---|---|---|---|---|
| 4 | **入口 embedder** | img_in(64→3072)、txt_in(4096→3072)、time_in、vector_in、cond_in、pe_embedder | 完全没有 | <1% |
| 5 | **modulation 生成链** | `vec→SiLU→Linear(H→6H/3H)→chunk` 算出 shift/scale/gate(adaLN-Zero) | 直接喂**固定随机张量**当 shift/scale/gate,跳过生成 | ≈0 |
| 6 | **vec** | `time_in(timestep)+vector_in(CLIP)`,`[1,3072]` 全局条件,每步变 | 不存在(无 timestep/CLIP 概念) | ≈0 |
| 7 | **block 串联到真实 depth** | 19 double + 38 single | 各 **1 个** | ×57 |
| 8 | **double→single 衔接** | `cat(txt,img)` 并成一条 → 过 single → `img[:,txt_len:]` 切回 | 单 block,无衔接 | 0 |
| 9 | **final_layer** | adaLN(vec)→LN(无affine)→仿射→Linear(3072→64) | 没有 | <1% |
| 10 | **采样循环/CFG/VAE** | 50 步 × 3 pass + CFG 外推 + VAE 解码 | 单次,无循环 | (×150) |

---

## 三、等价 / 有意适配(不影响正确性,不用动)

| # | 项 | 真实 | validate | 性质 |
|---|---|---|---|---|
| 11 | **flash attention 数学** | 无 causal mask,`softmax(QKᵀ/√d)·V` | **完全相同** | ✅ 等价,无算力缩水 |
| 12 | **DSB joint attention** | txt/img 的 Q/K/V 沿序列轴 cat → 一次 attn → split | **精确还原** | ✅ 等价 |
| 13 | **flash 算法实现** | FA2/FA3(在线分块) | golden 朴素 softmax / kernel 在线分块 | ✅ 数学等价,累积顺序不同 |
| 14 | **RoPE 实现形式** | 复数 / 2×2 旋转(Liger) | 置换矩阵 P + cos/sgn_sin(matmul) | ✅ 代数恒等,为 PLENA 改写 |
| 15 | **RoPE 融合位置** | 在 attention **内部**(`apply_rope`) | 拆成**独立前置 kernel**(多一次 HBM round-trip) | ✅ 等价,多一次 MX 量化 |
| 16 | **fused_qkv** | config = **False**(q/k/v 分开投影) | 也分开 linear_q/k/v | ✅ **一致**(本配置就是不融合) |
| 17 | **QK-norm** | per-head RMSNorm,scale 初始化 = ones | RMSNorm,scale 随机 | ✅ 结构同,初值不同 |
| 18 | **GELU** | tanh 近似 | tanh 近似 | ✅ 一致 |
| 19 | **几何规模** | H=3072, heads=24, head_dim=128, seq=8316/joint 8828 | H=HEAD·HLEN, heads=HEAD, seq=NSB·MLEN(4096/256) | 规模缩小(非错误) |

---

## 四、按"影响"归类速查

**真正会让数值/算力错的(必须改才 = 真实):**
- mlp_ratio 1 → 4(算力大头)
- RoPE 1D → 3D(语义)
- SSB LayerNorm 去掉 affine
- seq / H / heads 调到真实规模(算力规模)

**为了完整性补、但算力 ≈ 0:**
- embedder、modulation 生成链、vec、final_layer、cat/slice

**完全不用动(已对 / 有意适配):**
- flash attention(数学 + 算力都对)、DSB joint、fused_qkv、RoPE 实现形式、GELU、QK-norm 结构

---

## 五、单 block 算力构成(印证 MLP 为何关键)

真实 single block(seq=8828, mlp_ratio=4)≈ **2.96 TFLOP**:

| 部分 | 占比 | 说明 |
|---|---|---|
| **MLP** | **45%** (1.33 TFLOP) | ← validate 缩水到 1/4,这里低估最多 |
| **attention** | **32%** (0.96 TFLOP) | ← validate 完全正确 |
| qkv + proj | 23% | |
| modulation | ≈0.004% | 就地按层算,不预存 |

→ validate 现状(seq=4096 + mlp=1)的 block ≈ 1.13 TFLOP,**只有真实的 38%**。

---

## 六、端到端算力定位(为什么重心在 MMDiT)

| 模型 | 单次 | 跑几次 | 总算力 | 占端到端 |
|---|---|---|---|---|
| **MMDiT** | 168.6 TFLOP | 150 (50步×3pass) | **25,300 TFLOP** | **99.40%** |
| VAE decode | 146 TFLOP | 1 | 146 TFLOP | 0.57% |
| T5-XXL | 4.84 TFLOP | 1 | 4.84 TFLOP | 0.019% |
| CLIP | ~0.5 TFLOP | 1 | ~0.5 TFLOP | ~0.002% |

模型出处:MMDiT = Modified from **Flux** (Black Forest Labs);VAE = Modified from **HunyuanVideo** (腾讯) + diffusers;T5/CLIP = Google/OpenAI。Open-Sora 自身贡献 = 视频生成权重 + 把这些零件组装成开源推理/训练管线。

---

## 一句话总结

flash attention 部分(含 DSB joint、fused_qkv=False、GELU、QK-norm 结构)validate 与真实**一致或等价,不用动**。真实与 validate 的**实质差异**只有少数几个:

1. **mlp_ratio 1 vs 4**(算力低估最严重)
2. **RoPE 1D vs 3D 时空 [16,56,56]**(语义,换表即可)
3. **SSB 的 LayerNorm 多带了 affine**(真实无 affine)
4. **规模缩小**(seq / H / heads)

其余都是 validate 未建模的**完整 MMDiT 外围**(embedder、modulation 生成链、vec、final_layer、19+38 串联、采样循环),算力均 <1%,补它们是为结构/数值忠实而非算力。

- **算力评估** 视角:只需还原 **mlp_ratio + seq/H**。
- **数学忠实** 视角:再加 **RoPE-3D + SSB 去 affine** 两个修正。
