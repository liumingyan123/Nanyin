# 南音生成实验日志

> 分支：主线 `exp3_window_inference` → E1 `feature/compound-events` → 当前 `feature/E2_compound_window`
> 实验线：E0 基线 → exp1（其它分支）→ exp2（训练切片）→ exp3（窗口推理 + 评估 + 可视化）→ E1（复合事件编解码）→ E2（滑窗样本 + 窗口训练）

---

## E0 基线（`006ac88`）

- 整曲训练（不做窗口切分），`use_gt_rhythm` 开启
- 验证集 = 2 首 dapu 大谱整曲（`dapu14kouhuangtian` / `dapu15wujinjiao`）
- Tier-1（整曲口径，`results/comparison_original_vs_generated.csv`）：

| 来源 | composite（2 曲）|
|---|---|
| 原始南音（GT 天花板）| 87.98 |
| BoYaTCN | 65.86 |
| MusicMamba | 67.20 |

---

## exp2 — 训练阶段：256 音符窗口切片（`5acabba`）

**唯一改动**：`dataset_nanyin.py`（训练/验证样本定义），其余训练代码、超参全部不动（控制变量）。

- 划分口径：严格按歌曲 ID 整曲划分，同曲切出的窗口不跨训练/验证集
- 训练集：窗长 256、步长 128（50% 重叠）
- 测试/验证集：窗长 256、步长 256（不重叠），尾部不足 256 音符丢弃（不 padding）
- 结果：训练样本数百个；**验证集 = 10 个窗口**（`dapu14kouhuangtian` 4 + `dapu15wujinjiao` 6）
- 全局级真值（速度/调式/风格）仍按整曲推导，仅序列级字段按窗口裁剪

**训练产物（`checkpoints/exp2_*`，best val CE 早停）**

| 模型 | 权重文件 | epoch | best val CE | val composite |
|---|---|---|---|---|
| BoYaTCN | `exp2_boyatcn/best_valloss_model.pt` | 76 | 2.3909 | 0.5398 |
| MusicMamba | `exp2_musicmamba/best_valloss_model.pt` | 24 | 2.4604 | 0.5310 |

---

## exp3 — 推理评估阶段：窗口粒度推理 + Tier-1 评估 + 降维可视化

### 3.0 核心原则

> **训练 / 推理粒度同源**：exp2 以 256 音符窗口为训练单位，exp3 全程以**窗口**为推理与评估单位，
> **禁止切换整曲推理口径**，否则与训练分布不一致、结论不可比。

### 3.1 基线校验（推理前已完成，全部通过）

| 校验项 | 结果 |
|---|---|
| `WINDOW_SLICE_ENABLED` / `WINDOW_SIZE` / `TRAIN_WINDOW_STRIDE` / `TEST_WINDOW_STRIDE` | `True` / 256 / 128 / 256 ✅ |
| BoYaTCN 权重 strict 加载 | ✅ 2.65M 参数，NaN 0 / Inf 0 |
| MusicMamba 权重 strict 加载 | ✅ 1.40M 参数，NaN 0 / Inf 0 |
| 验证集样本 | 10 个 256 音符窗口 ✅（严格 256 token 校验 10/10）|

### 3.2 推理生成（`generate_music.py`）

- 权重：`checkpoints/exp2_{boyatcn,musicmamba}/best_valloss_model.pt`（可用环境变量
  `NANYIN_CKPT_FILENAME=best_model.pt` 切回 NOEF 权重）
- 样本：验证集 10 个窗口；每窗口取前 16 个 GT token 作 prompt，自回归续写至 256 token
- 命名：`{song_id}_w{窗口起始编号}_{model}.mid`（防同曲多窗口覆盖）
- 产物：20 条 MIDI（BoYaTCN 10 + MusicMamba 10），目录
  `results/generated_midi_exp2_windows/{boyatcn,musicmamba}/`
- 索引清单：`results/generate_list_exp2_windows.csv`（曲号、窗口区间、窗口音符数、模型、MIDI 路径）

### 3.3 Tier-1 窗口级评估（仅符号层面，不做音频）

- 评估粒度：窗口级；GT = 该窗口对应的 GT 切片
- **音符排序 bug 修复**：GT 切片必须取 `dataset_nanyin.load_song_annotations()` 的
  **训练同源顺序**（onset 升序、同 onset 按音高升序）；
  `nanyin_metrics.load_ground_truth()` 用 pretty_midi 原序，在"同时发音"音符上会错位（实测 10/10 窗口全部错位）
- 生成侧：`nanyin_metrics.parse_midi_to_sequence()` 取真实 onset，统一跳过前 16 个 prompt token；
  GT 天花板为 GT vs 自身（不跳过 prompt）
- 46 维特征（评估区域 = 跳过 16 个 prompt 后的 240 音符，三类口径一致）：
  pitch-class 直方图(12) + 相邻音程直方图(12) + log-时值直方图(8) + log-IOI 直方图(8)
  + 音高统计(2) + 音符密度(1) + 调式符合度(1) + 装饰音占比(2)

**composite 汇总（10 窗口）**

| 类别 | n | mean | std | min | max |
|---|---|---|---|---|---|
| GT 天花板 | 10 | 83.55 | 1.45 | 81.51 | 86.23 |
| BoYaTCN | 10 | 65.99 | 1.26 | 64.81 | 68.79 |
| MusicMamba | 10 | **65.59** | 0.67 | 64.58 | 66.45 |

**9 项 NOEF 指标均值**

| 指标 | GT 天花板 | BoYaTCN | MusicMamba |
|---|---|---|---|
| pitch_accuracy | 100.00 | 4.88 | 4.63 |
| pitch_stability | 90.78 | 89.47 | 88.75 |
| rhythm_consistency | 63.05 | 51.42 | 51.88 |
| tempo_accuracy | 100.00 | 75.11 | 86.09 |
| ornament_score | 76.83 | 72.90 | 71.91 |
| ornament_density | 87.96 | 90.94 | 90.98 |
| mode_matching_score | 88.98 | 81.29 | 75.00 |
| fad_distance | 0（未做音频）| 0（未做音频）| 0（未做音频）|
| melody_cosine_similarity | 100.00 | 96.05 | 95.23 |

**与 E0 对照（同指标 composite）**

| 类别 | E0 整曲（2 曲）| exp3 窗口（10 窗口）| 差异 |
|---|---|---|---|
| GT 天花板 | 87.98 | 83.55 | −4.43 |
| BoYaTCN | 65.86 | 65.99 | +0.13 |
| MusicMamba | 67.20 | 65.59 | −1.61 |

解读：256 窗口训练 + 早停后的生成质量与 E0 整曲训练基本持平（差值 ≤ 1.6 分）；
GT 天花板在窗口口径下略低（窗口只覆盖局部片段，节奏一致性等指标波动更大）。
两模型差距 0.40 分（65.99 vs 65.59），落在各自 std（1.26 / 0.67）量级内，属噪声范围。

产物：`results/exp3_eval_windows.csv`（30 行 × 完整指标 + 46 维特征）、
`results/exp3_eval_windows_summary.json`（composite 与 9 指标 mean/std 汇总）。

### 3.4 降维可视化（`visualize_embedding.py --exp3-csv`）

- 数据源：直接读 `results/exp3_eval_windows.csv` 内的 46 维特征（非评估指标列）
- 三类共 30 点：窗口 GT 10 + BoYaTCN 10 + MusicMamba 10；StandardScaler 标准化后降维
- 超参（小样本必须压小）：UMAP `n_neighbors=8, min_dist=0.1`；t-SNE `perplexity=5, max_iter=2000, init=pca`；
  `random_state=42` 固定，可复现
- 图中：三类着色 + `X` 标类中心 + 细灰线连接同一窗口的 GT→BoYaTCN→MusicMamba 三点
- 产物：`results/exp3_umap_windows.png`（166 KB）、`results/exp3_tsne_windows.png`（144 KB）

**量化结果**

| 指标 | 数值 |
|---|---|
| silhouette @46 维 | 0.193 |
| silhouette @UMAP 2D | 0.717 |
| silhouette @t-SNE 2D | 0.649 |
| 类内平均距离 GT / BoYaTCN / MusicMamba | 6.071 / 4.639 / 3.540 |
| 中心距 GT–BoYaTCN | 7.287 |
| 中心距 GT–MusicMamba | **9.194** |
| 中心距 BoYaTCN–MusicMamba | 5.266 |

**结论**

1. **生成与 GT 明确分离**：两类生成样本到 GT 的中心距（7.287 / 9.194）均大于其自身类内散布
   （4.639 / 3.540），t-SNE 与 UMAP 两种降维结论一致。
2. **MusicMamba 距 GT 更远**（9.194 > 7.287），与 Tier-1 中 MusicMamba 的
   `mode_matching_score` 明显更低（75.00 vs 81.29）互相印证：MusicMamba 在调式符合度上偏离更大。
3. **两模型部分可分但差距有限**：BoYaTCN–MusicMamba 中心距 5.266 大于各自类内散布
   （4.639 / 3.540），故降维图上可分；但两者 composite 仅差 0.40 分，
   说明"可分"主要体现在统计特征分布层面，而非质量高低。
4. n=10/类，样本量小，silhouette 与簇形状仅作定性参考，**不宜写成"A 优于 B"的定量证据**。

### 3.5 未完成 / 禁止项

- 未做 Tier-2：不运行 TokenSynth 音频渲染、不做 FAD 音频评估（本轮 `fad_distance` 统一记 0，不计入结论）
- 未做整曲推理（刻意禁止，保证训练/推理窗口粒度同源）
- 两个"无参考旋律质量"辅助指标 `transition_smoothness` / `melody_coherence` 在南音素材上恒为 0（相邻音平均步长 8~10 半音，超出 0–100 映射上限），不具区分力
- `pitch_accuracy` 约 5% 属该指标固有特性（逐位与 GT 比对的严格匹配率），E0 同期同为 4~6%，不是 exp3 引入的回退

### 3.6 复现命令

```bash
# 1) 推理：20 条 MIDI + 索引清单
venv/Scripts/python.exe generate_music.py

# 2) Tier-1 窗口级评估（驱动脚本在 results/exp3_recipe/，被 .gitignore 排除，不入库）
venv/Scripts/python.exe results/exp3_recipe/_exp3_eval_windows.py

# 3) 三类降维可视化
venv/Scripts/python.exe visualize_embedding.py --exp3-csv
```

### 3.7 入库范围

- 入库（源码/文档）：`generate_music.py`、`visualize_embedding.py`、`EXPERIMENT_LOG.md`
- 不入库（`.gitignore` 规则 `results/`）：图片、CSV、JSON、权重、MIDI

---

## E1 — 复合事件编解码：pitch + duration + time-shift（`feature/compound-events` @ `ee8d23e`）

### E1.0 目标与范围

> 只做**复合事件序列 + MIDI 往返编解码**（`MIDI → (pitch, duration, time-shift) → MIDI`），
> 加上模型/数据/生成链路的**结构性改造**；不训练、不滑窗、不做损失消融、**不产出任何生成 MIDI**。

为什么要有 E1：旧链路的音符 token 只有"音高 + 时值"两种信息，onset 由时值隐式累加，
没有显式的"休止/间隔"表达——南音大量撩拍气口（onset 间隔 ≫ 音符时值）无法编码；
E1 把序列改成三路分类 token，其中 **time-shift 是显式 onset 间隔**，且约定为
**前向 IOI：shift_i = onset_{i+1} − onset_i**（onset 递推：`onset_{i+1} = onset_i + shift_i`）。
该约定天然使**序列末位的 shift 成为伪标签**（末音符之后没有"下一个音"），编码置 0，
训练时必须对该位置做 loss 掩码（已写入代码注释与自测报告）。

### E1.1 各文件改动明细（对应提交 `ee8d23e`，8 文件 +1681/−504）

**新增：`nanyin_config.py` + `configs/e1_compound_events.yaml`（配置收口）**

- 之前超参散在各脚本硬编码里，E1 起统一为 YAML + 一个 `load_config()` 加载器；
- yaml 内容：量化网格 `time_unit_sec=0.02`（20 ms）、`pitch_vocab_size=128`、
  `duration_max_steps=400`（精确上限 7.98 s）、`shift_max_steps=1000`（精确上限 19.98 s）、
  模型三头维度、采样参数、自测参数；
- max_steps 不是拍的脑袋：实测 11 首 GT 分布（最大时值 6.89 s、最大 onset 间隔 19.40 s）
  后定标，使真实数据**不落入溢出档**，往返误差限制在 ±半格（≤10 ms）。

**修改：`dataset_nanyin.py`（数据侧核心改动）**

- 新增 `CompoundEventCodec` 类：`encode_notes()` 把音符列表编成三路等长 ID 序列，
  `events_to_notes()` 由 shift **累加**重建 onset；
  超上限落"溢出档"（类别数 = max_steps + 2），解码只能还原下界，且其后所有 onset 被前移（误差累积，已知局限）；
- time-shift 编码方式有讲究：**先把每个音符的绝对 onset 量化到网格、再相邻作差**
  （最初按"每个原始间隔独立量化"实现，各处 ±半格舍入在解码累加时形成单向漂移，
  实测千音符曲目累计 onset 误差达数秒——dapu06 均值 4.19 s；改为绝对位置作差后不随长度累积）；
- 排序约定：编码前按 `(start_time, note)` 排序（训练同源顺序），负 gap（多轨合并重叠）截断为 0 并计数；
- `build_target_dict()` 新增产出 `target_pitch_ids / target_duration_ids / target_shift_ids`，
  旧 `target_tokens` 保留（向后兼容）；`collate` 补齐新键 dtype 对齐。

**修改：`models.py`（三路输入 + 三分类头）**

- `NanyinBoYaTCN` / `NanyinMusicMamba` 共同改造：
  新增 `pitch_embed` + `duration_embed` + `shift_embed` 三个 Embedding，**三者相加**作为输入表示；
  输出端新增 `pitch_head` / `duration_head` / `shift_head` 三个线性分类头
  （128 / 402 / 1002 类）；
- 保留旧的单 token 接口（只传 pitch_ids 时走旧 `pred_tokens` 链路），旧权重仍可加载推理；
- 顺带修掉一个存量 bug：缺 `import torch.utils.checkpoint`，MusicMamba 训练态前向必 AttributeError。

**修改：`generate_music.py`（推理/写出链路重写）**

- 三属性**同时采样**（每步从三个头各采一个 token），不再只采音高；
- onset 不再由时值隐式累加，而是 `onset += index_to_shift(shift)` 显式推进（含休止表达）；
- `notes_to_midi` 改为**绝对时间事件流**写 MIDI：先按 (tick, 事件类型) 排序全部 note_on/note_off，
  解决旧版"边写边配对"在同时发音/同音重叠时的 (channel, pitch) 配对歧义
  （历史 bug：1025 音符被解析成 792）；同音高重叠时把前一音截断到后一音起点（保数量、牺牲时值）；
- 无复合事件权重时明确跳过生成，不再产出残缺 MIDI。

**新增：`tests/test_compound_event_roundtrip.py`（往返自测脚本）**

- 三层测试：① 编解码器单元（精确档/溢出档/PAD 语义 5 项）；
  ② 11 首 GT 真实曲目 `MIDI → 编码 → 解码 → 写 MIDI → 重读` 全链路逐音比较；
  ③ 5 个合成边界用例（超长休止/同时发音 shift=0/超长音符/同音反复重叠/负间隔乱序输入）；
- 产出自测报告 markdown + 指标 JSON。

**新增：`reports/E1_selftest_compound_events.md`（自测报告，入库）**

**修改：`.gitignore`**

- `tests/*` 忽略但保留 `!tests/*.py`（自测脚本要入库）。

### E1.2 自测结果（详见 `reports/E1_selftest_compound_events.md`）

| 判定项 | 标准 | 实测 | 结论 |
|---|---|---|---|
| 音高无损 | pitch mismatch = 0 | 0 / 11 首 | PASS |
| 音符数守恒（写回 MIDI 后重读） | count_dec = count_reread | 11/11 | PASS |
| 末尾 shift 为伪标签 | 末位 = 0 | 11/11 | PASS |
| 编解码器单元检查 | 5 项 | 5/5 | PASS |
| 时值往返误差 | ≤ 20 ms（1 格）| 100%，max 10.0 ms，均值 ≈5 ms | PASS |
| onset 往返误差 | ≤ 20 ms | 100%，max 20.0 ms，均值 5.1~9.8 ms | PASS |

调试过程中修过的两个坑（都记录在报告里）：
1) 比较排序口径——量化把亚毫秒级 onset 差拉平后，两侧排序键不一致会造成"假性 pitch 不一致"，
file-level 比较必须两侧同键 `(start, note)` 排序；
2) 末音符提前——曾按"后向 IOI"（shift_i = onset_i − onset_{i−1}）把最后一个真实间隔当伪标签清零，
导致末音符 onset 提前，改为前向 IOI 约定后消除。

已如实记录的残余局限：同音高重叠截断（每首 20~311 处，换音符数守恒）、
`max_notes=4096` 截断（dapu02/04/05/12/13）、为避开溢出把 shift 词表拉到 1002 类导致长尾稀疏
（后续可改对数分箱）、time-shift 溢出档误差累积。

### E1.3 冒烟验证（未训练，仅结构验证）

- 两模型复合事件前向 + 三头 CE loss 反向传播梯度正常；
- 旧单 token 链路 forward 形状不变（旧权重兼容）；
- 验证集 `__getitem__` 产出三路 ID 序列，末位 shift = 0（伪标签就位）。

---

## E2 — 复合事件滑窗样本 + BoYaTCN 窗口训练（`feature/E2_compound_window`）

提交链：`c6cbc72`（配置）→ `21d6327`（数据）→ `3530286`（模型）→ `fca09ec`（自测）→ `90a84f4`（训练）→ `（本报告与日志补记）`
详细报告：`reports/E2_window_training.md`

### E2.0 目标与范围

在 E1 编解码基础上做**滑窗样本构建 + BoYaTCN 窗口训练**；损失只有
`total = 1.0*pitch_CE + 0.7*duration_CE + 0.7*shift_CE`，
**不做 L0-L5 损失消融**（留给 E3）、不生成 MIDI、不做评估。

### E2.1 各文件改动明细

**新增 `configs/e2_compound_window.yaml` + 修改 `nanyin_config.py`（配置分层）**

- E2 yaml 是**覆盖层**：`load_config(path)` 先读基础层 `e1_compound_events.yaml` 再做深合并，
  未声明字段回退基础层 → codec/model 等跨阶段参数只有一份真源，避免双份维护漂移；
- 新增 `window:`（256 事件窗口 / 训练步长 128 / 验证步长 256 / drop_tail）、
  `training:`（200 轮上限、bs16、lr1e-3、wd1e-5、grad_clip1.0、early_stop_patience20、
  损失权重 1.0/0.7/0.7、词表日志每 10 轮）、`outputs:`（产物路径）三节；
- 基础层 codec 新增 `num_special_tokens: 2`（BOS/EOS）。

**修改 `dataset_nanyin.py`（窗口切分核心）**

- 新增 `encode_song_events()`（整曲→三路 ID 序列）、`slice_event_windows()`（等长窗口、不足丢弃）、
  `build_window_sample()`（BOS/EOS/attention_mask/掩码）、
  `CompoundEventWindowDataset`、`get_compound_window_datasets()`、`collate_compound_batch()`、
  `compute_vocab_class_stats()`（duration/shift 词表类别频次与长尾统计）；
- 布局：输入 `[BOS] + 256 事件`，目标 `256 事件 + [EOS]`，长度 257；
  **目标末两位（最后事件位 + EOS 位）shift 置 `IGNORE_INDEX=-100`**，
  且输入侧窗口末 shift 也置 0（未定义），保证"伪标签不参与反向传播"；
- 曲目 ID 隔离：训练 9 首 / 验证 2 首无交集，窗口只在单曲内切；
- 四条 E1 硬性约束落地：① 末尾 shift ignore_index 掩码；② 词表长尾频次统计；
  ③ 被 `max_notes=4096` 截断的曲目清单如实打印；④ **GT 噪声来源写入文件头注释**
  （GT=BasicPitch 转录，同音重叠，写回 MIDI 时前一音时值被截断 → 局部时值噪声，论文须记录）。

**修改 `models.py`（词表扩展 BOS/EOS）**

- 三路 embedding / 分类头大小 = 真实类别数 + `num_special_tokens`：
  pitch 130、duration 404、shift 1004；BoYaTCN 与 MusicMamba 口径统一。

**修改 `train_boyatcn.py`（新增 E2 训练模式）**

- 新增 `--mode {noef, compound}`，**原 E0/exp 主线 NOEF 整曲训练路径（`train()`）完全保留**；
- `train_compound()`：三路加权 CE（shift 用 ignore_index）、Adam+StepLR、梯度裁剪、
  **早停监控验证集 total loss（patience 20）**、保存 `best_valloss_model.pt`（best-val）；
- 训练前必打：窗口统计清单（写 CSV）、max_notes 截断曲目清单、词表长尾统计（写 JSON）；
  训练中每 10 轮复打长尾摘要；逐轮 train/val 分项 loss + val token 准确率写入日志与 CSV。

**新增 `tests/test_window_slice.py`（窗口切分单元自测）**

- T1 窗口数量公式与边界（7 个子用例）、T2 无 padding、T3 曲目 ID 隔离、
  T4 BOS/EOS/attention_mask、T5 伪标签掩码（位置 + 损失等价 + 梯度为 0 三层验证）、
  T6 切片一致性与 onset 单调、T7 真实数据统计；实测 **22/22 PASS**。

### E2.2 实测结果（真实运行，非挑选）

| 项 | 实测 |
|---|---|
| 窗口数 | 训练 248（9 首，步长 128）/ 验证 10（2 首，非重叠）；尾部丢弃 358 事件 |
| 被 max_notes 截断 | 6 首（dapu02/04/05/06/12/13，合计丢弃 7282 音符，均未参与训练）|
| 单元自测 | 22/22 PASS；掩码三层验证：损失等价差 0.00e+00，不屏蔽时 7.4636 vs 7.4647，屏蔽位梯度 0.0e+00 |
| 训练 | 200 轮上限，**早停于 epoch 27**（val 连续 20 轮未下降），耗时 4.6 分钟，cuda |
| best-val | **epoch 7，val_total = 6.8788**（`checkpoints/e2_boyatcn/best_valloss_model.pt`）|
| best 轮验证 token 准确率 | pitch 36.07% / duration 7.00% / shift 23.22% |
| 末轮（27）| train 1.8382 vs val 10.1936；准确率 34.67% / 5.84% / 11.14% |
| 词表长尾 | duration 185/404 类出现（219 类零样本，长尾 62 类占 0.25%）；shift 175/1004 类出现（**829 类零样本**，长尾 46 类占 0.17%）|

**不如预期的地方（如实记录，详见报告第 7 节）**：

1. **严重过拟合**：train loss 9.34→1.84 单调下降，val loss 却在第 5~7 轮见底（6.88）后升到 10.19；
   248 窗口 / 9 首喂 3.44M 参数，best-val 落在 epoch 7。
2. **duration 头几乎没学到**：验证准确率 epoch1 7.12% → best 7.00% → 末轮 5.84%，
   低于多数类基线（≈10.8%），404 类细档 + 分布集中是主因。
3. **shift 验证准确率随训练下降**（24.98% → 23.22% → 11.14%），与过拟合一致。
4. **验证集过小**：仅 2 首 10 窗口，val loss 抖动大、早停判定敏感。
5. **伪标签掩码的代价**：中段窗口最后 1 个真实可得的 shift 监督被一并屏蔽（248 个/轮），
   选择一致处理是为与"第 256 事件后接 EOS"的生成语义一致；E3 若要回收需区分整曲末尾/中段窗口。

---

## 4. 分支与提交历史

主线：`exp3_window_inference`（exp1 消融实验在其它分支）；E1 起新分支 `feature/compound-events`（自 main `006ac88` 切出）。

| 提交 | 分支 | 内容 |
|---|---|---|
| `006ac88` | 各分支公共基点 | E0 基线：原始版本，整曲训练，`use_gt_rhythm` 开启 |
| `5acabba` | exp3_window_inference | exp2：仅改 `dataset_nanyin.py`，256 音符窗口切片，其余不动（控制变量）|
| `ffff985` | exp3_window_inference | exp3：窗口粒度推理 + Tier-1 评估 + UMAP/t-SNE 可视化 |
| `dde7a3a` | exp3_window_inference | 本日志补充分支历史与资产台账 |
| `ee8d23e` | feature/compound-events | E1：复合事件（pitch+duration+time-shift）编解码 + 三路分类头 + MIDI 绝对时间写出 + 往返自测 |
| `ecf8293` | feature/compound-events | 日志补记 E1 阶段（逐文件改动明细 + 自测结果 + 资产台账）|
| `c6cbc72` → `90a84f4` | feature/E2_compound_window | E2：滑窗与训练配置、256 事件窗口切分（BOS/EOS+掩码+GT 噪声注释）、词表扩展 BOS/EOS、窗口切分自测、三输出头窗口训练（当前 HEAD）|

> 线路调整记录：2026-09-17 曾做过一版"以 10 窗口指标 CSV 为数据源的作图"及对应脚本改动
> （提交 `8109fd0` / `d23efb5`），经确认口径有歧义后整体废弃，分支硬回退到 `5acabba` 并清理全部
> 临时产物后重做，即本日志所记录的现行 exp3 线路。

---

## 5. 实验资产台账（文件 → 作用 → 位置 → 是否入库）

约定：✅ = 已入库（git 跟踪）；🏠 = 仅本地，被 `.gitignore` 排除（规则 `results/`、`checkpoints/` 等）。

### 5.1 源码（仓库根目录，✅ 全部入库）

| 文件 | 作用 |
|---|---|
| `dataset_nanyin.py` | 数据集定义。exp2 起 `WINDOW_SLICE_ENABLED=True`：训练窗 256/步长 128，测试窗 256/步长 256；`load_song_annotations()` 提供训练同源顺序音符流（exp3 GT 切片的唯一正确来源）。E1 起新增 `CompoundEventCodec`（音符列表 ↔ 三路 ID 序列，time-shift 为**前向 IOI**：shift_i = onset_{i+1} − onset_i，末位伪标签 0；绝对 onset 先量化再作差防漂移；溢出档/负 gap 截断计数），`build_target_dict()` 产出三路 ID 序列（旧 `target_tokens` 保留兼容）|
| `models.py` | `NanyinBoYaTCN` / `NanyinMusicMamba` 模型定义。E1 起新增 `pitch/duration/shift` 三个输入 Embedding（相加）与 `pitch_head/duration_head/shift_head` 三个分类头（128/402/1002 类）；保留旧单 token 链路（旧权重兼容）；顺带修复缺失的 `torch.utils.checkpoint` 导入 |
| `train_boyatcn.py` / `train_musicmamba.py` | 两套模型训练脚本（exp2 用它们训练出 `exp2_*` 权重）|
| `loss_nanyin.py` | NOEF 风格多任务损失 |
| `generate_music.py` | 推理脚本。exp3 版：加载 `checkpoints/exp2_*` best-val-CE 权重，对 10 个验证窗口做窗口粒度续写（前 16 GT token 作 prompt → 256 token），输出 20 条 MIDI + 清单。E1 起改造为复合事件采样：每步从三头各采一 token，`onset += index_to_shift(shift)` 显式推进（含休止）；`notes_to_midi` 改**绝对时间事件流**写出（全事件按 (tick, 类型) 排序后再配对，杜绝 (channel,pitch) 错配丢音符），同音高重叠截断保数量；无复合事件权重时明确跳过、不产出残缺 MIDI |
| `nanyin_metrics.py` | Tier-1 指标库：`evaluate_single_song`、`parse_midi_to_sequence`、`compute_composite_score`、9 项 NOEF 指标 |
| `feature_extractor.py` | 46 维符号统计特征（pitch-class/音程/时值/IOI 直方图 + 统计量）|
| `visualize_embedding.py` | 降维作图。默认：E0 四类口径 → `embedding_symbolic_*.png`；`--exp3-csv`：读 `exp3_eval_windows.csv` 的 46 维特征画 exp3 三类图 → `exp3_{umap,tsne}_windows.png`（三类配色 绿/蓝/红，X=类中心，灰线=同窗口三点连线）|
| `compare_original_vs_generated.py` | E0 整曲口径 Tier-1 评估 → `comparison_original_vs_generated.csv` |
| `compare_openstyle_ablation.py` | 早期风格损失消融评估 → `comparison_openstyle_ablation.csv` |
| `nanyin_config.py` | E1 新增：`load_config()` 统一加载 `configs/*.yaml`（量化网格/词表/模型头/采样/自测参数收口，替代脚本内硬编码）|
| `tests/test_compound_event_roundtrip.py` | E1 新增：往返一致性自测（编码器单元 5 项 + 11 首 GT 全链路逐音比较 + 5 个合成边界用例），产出报告与指标 JSON |
| `basic_pitch_runner.py` / `run_amt_pipeline.py` / `make_gt_midi.py` / `wav_to_midi_json.py` / `model_midi.py` | 数据管线：音频→BasicPitch→GT MIDI→标注 JSON |
| `EXPERIMENT_LOG.md` | 本日志 |

### 5.1b E1 配置与报告（`configs/`、`reports/`、`tests/`，✅ 入库）

| 文件 | 说明 |
|---|---|
| `configs/e1_compound_events.yaml` | E1 唯一参数源：`time_unit_sec=0.02`、`duration_max_steps=400`（上限 7.98 s）、`shift_max_steps=1000`（上限 19.98 s）、三头词表、采样与自测参数；max_steps 按 11 首 GT 实测分布定标（最大时值 6.89 s / 最大 onset 间隔 19.40 s），使真实数据不落溢出档 |
| `reports/E1_selftest_compound_events.md` | E1 自测报告（入库）：配置、结论表、单元/曲目/边界用例明细、已知局限 7 条、复现方式 |
| `tests/`（含 `.gitignore` 规则 `tests/*` + `!tests/*.py`）| 自测脚本目录，只入库 `.py` |

### 5.1c E2 配置与报告（`configs/`、`reports/`、`tests/`，✅ 入库）

| 文件 | 说明 |
|---|---|
| `configs/e2_compound_window.yaml` | E2 **覆盖层**配置：未声明字段回退 E1 基础层。`window:`（256 事件窗口 / 训练步长 128 / 验证步长 256 / drop_tail）、`training:`（200 轮上限、bs16、lr1e-3、wd1e-5、clip1.0、patience20、权重 1.0/0.7/0.7、词表日志每 10 轮）、`outputs:`（产物路径）|
| `reports/E2_window_training.md` | E2 训练报告（入库）：窗口统计清单、超参、27 轮完整 loss 表与早停节点、词表长尾统计、7 条已知问题（含过拟合与 duration 头未学到东西）、损失曲线图、复现方式 |
| `reports/E2_loss_curve.png` | E2 损失曲线图（入库）：左=train/val 总损失+best-val 标注（epoch 7 见底后回升），右=三路分项 CE（train 实线/val 虚线，duration 的 val CE 不降反升）|
| `plot_loss_curve.py` | E2 新增作图脚本：读逐轮损失 CSV → 双面板损失曲线 PNG（中文字体处理与 visualize_embedding.py 一致）|
| `tests/test_window_slice.py` | E2 新增：窗口切分与掩码自测 T1-T7（22 项），产出自测明细 JSON |

### 5.2 权重与工具（`checkpoints/`，🏠 不入库）

| 路径 | 说明 |
|---|---|
| `exp2_boyatcn/best_valloss_model.pt`（14.1 MB）| **exp3 推理用**：BoYaTCN best val CE（epoch 76，CE 2.3909）|
| `exp2_musicmamba/best_valloss_model.pt`（9.3 MB）| **exp3 推理用**：MusicMamba best val CE（epoch 24，CE 2.4604）|
| `exp2_boyatcn/best_model.pt` / `exp2_musicmamba/best_model.pt` | exp2 的 NOEF 最优权重（备用）|
| `boyatcn/best_model.pt`、`musicmamba/best_model.pt` 及各 `checkpoint_epoch_*.pt` | E0 整曲训练产物（历史对照用）|
| `grid_boyatcn_*` / `grid_musicmamba_*` | 早期五项风格损失网格/消融权重（历史实验，与 E0~exp3 主线无关）|
| `fluidsynth.exe`、`*.sf2` | 音频渲染工具（Tier-2 备用，本轮未使用）|

### 5.3 exp3 产物（`results/`，🏠 不入库）

| 路径 | 说明 |
|---|---|
| `generated_midi_exp2_windows/boyatcn/`（10 个 .mid）| BoYaTCN 窗口生成：`{song_id}_w{起始编号}_boyatcn.mid` |
| `generated_midi_exp2_windows/musicmamba/`（10 个 .mid）| MusicMamba 窗口生成，命名同上 |
| `generate_list_exp2_windows.csv` | 生成索引清单：曲号、窗口区间、窗口音符数、模型、MIDI 路径 |
| `exp3_eval_windows.csv` | Tier-1 明细：30 行（10 窗口 × 3 类）×（9 项 NOEF + composite + 46 维特征 `feat_00~45`）|
| `exp3_eval_windows_summary.json` | composite 与 9 指标的 mean/std/min/max 汇总 |
| `exp3_tsne_windows.png` / `exp3_umap_windows.png` | exp3 三类降维图（本日志 3.4 节的图）|
| `exp3_recipe/_exp3_eval_windows.py` | Tier-1 窗口级评估驱动脚本（不入库，复现命令见 3.6）|

### 5.3b E1 产物（`results/`，🏠 不入库）

| 路径 | 说明 |
|---|---|
| `e1_roundtrip_metrics.json` | E1 自测指标：unit_checks + 11 首逐曲误差统计（pitch/duration/onset 均值与 max、容差内比例、写回重读计数）+ 5 个边界用例明细 |
| `e1_roundtrip_tmp/*.mid` | E1 往返临时 MIDI（11 首 GT 往返 + 边界用例 + 重复对照，可随时由自测脚本重建）|

### 5.3c E2 产物（`results/`，🏠 不入库）

| 路径 | 说明 |
|---|---|
| `e2_window_stats.csv` | 窗口数量统计清单：逐曲 划分/事件数/窗口数/尾部丢弃事件/max_notes 截断音符 |
| `e2_boyatcn_train.log` | 完整训练日志：配置、窗口统计、截断清单、词表频次统计、27 轮逐轮 loss 与早停节点 |
| `e2_boyatcn_train_log.csv` | 逐轮 train/val 分项 loss + val token 准确率 + lr + is_best（27 行）|
| `e2_vocab_stats.json` | duration / time-shift 词表各类别频次、零样本类别数、长尾类别与占比、Top10 类别 |
| `e2_window_slice_test.json` | 窗口切分自测明细（22 项判定 + 各用例实测值）|
| `checkpoints/e2_boyatcn/best_valloss_model.pt` | E2 best-val 权重（epoch 7，val_total 6.8788）|

### 5.4 E0 / 历史产物（`results/`，🏠 不入库，保留作对照）

| 路径 | 说明 |
|---|---|
| `generated_midi/`、`generated_midi_old/` | E0 整曲生成 MIDI（新旧两版权重）|
| `generate_list.csv` | E0 生成清单 |
| `comparison_original_vs_generated.csv` | E0 整曲 Tier-1 结果（exp3 的对照基线，见 3.3 表）|
| `embedding_symbolic_umap.png` / `embedding_symbolic_tsne.png` | E0 四类降维图（`visualize_embedding.py` 无参数运行）|
| `exp2_train_*.log`、`exp2_train_summary_*.json`、`experiment_record_exp2_*.csv`、`exp2_recipe/` | exp2 训练日志/汇总/超参快照 |
| `comparison_openstyle_ablation.csv`、`generated_ensemble*`、`gt_*`、各 `*.log` | 更早的风格消融/合奏/音频实验产物（历史线，与 E0~exp3 主线无关）|

---

## 6. E3 损失消融实验（分支 `experiment/E3_loss_ablation`）

分支来源：`feature/E2_window_training`（基准 commit `ec4bd97`）→ 新建 `experiment/E3_loss_ablation`。
分支纪律：旧预实验分支 `fix/evaluation-umap`、`exp2`、`exp3_window_inference` 只读；
`feature/E2_window_training` 冻结，E2（`ec4bd97`）作为唯一对照基线；E3 不改动 E2 代码与产物。

### 6.0 前置背景：E2 基线现象（E3 所有指标均与之对照）

| 现象 | E2 实测（`ec4bd97`） |
|---|---|
| 样本量 | 248 训练窗口 / 10 验证窗口（9 训练曲 + 2 验证曲）——小样本 |
| 过拟合 | train 总损失 9.34 → 1.84；val 总损失 epoch 7 见底 **6.8788** → 10.19（早停 epoch 27，patience 20） |
| duration 头 | 验证 token 准确率仅 **7.00%**（低于多数类基线 ≈10.8%），404 细档长尾稀疏（219 类零样本） |
| 其余头 | val token 准确率 pitch 36.07% / shift 23.22% |

E3 目标：主干、窗口切分、数据、超参全部固定，只改损失组合，加入**基于生成头 logits** 的南音分布约束
（音级 / 调式 / 时值 / IOI），观察过拟合程度与 duration 准确率的变化。

### 6.1 代码改动

| 文件 | 改动 |
|---|---|
| `models.py` | `NanyinBoYaTCN` / `NanyinMusicMamba` 新增 `use_features_head` / `use_style_head`（**默认 False**）：独立辅助头 features_head（MFCC 回归）、style_head（随机投影风格嵌入）默认不创建、不参与前向、forward 返回 None。→ 所有南音分布约束只能作用于 pitch_head / duration_head / shift_head 的最终 logits。noef 老链路（`loss_nanyin.nanyin_total_loss` 与两个训练脚本）做 None 兜底（风格损失记 0），不报错。 |
| `loss_nanyin.py` | 新增第九节"E3 可微分布损失"：`build_pitch_class_matrix`（128→12 音级）、`build_value_bin_matrix`（类别秒值→直方图 bin）、`build_mode_distribution_matrix`（南音四调五声音级先验）、公共内核 `soft_histogram_kl`（softmax→软直方图→KL，全程可微）、`pitch_class_distribution_loss`、`mode_distribution_loss`、`duration_distribution_loss`、`shift_ioi_distribution_loss`、`ornament_density_distribution_loss`（L5 框架）。 |
| `dataset_nanyin.py` | 新增 `resolve_window_mode_index()`，窗口样本新增 `target_mode_index`（窗口音级直方图匹配南音四调 → 回退曲目级 → 回退五空管）；`collate_compound_batch` 同步 stack；新增 `mode_index_counts()` 供训练日志记录调式真值分布。 |
| `train_boyatcn.py` | 新增 `--config`（覆盖层 YAML）；`build_distribution_matrices()` 构造常量聚合矩阵；`compute_compound_loss` / `evaluate_compound` 支持 4 项分布损失 + L5 装饰损失权重；**早停/最优判定改为监控"三路加权 CE"**（与 E2 同口径，跨组可比，分布损失只参与优化）；逐轮 CSV 新增 `train/val_pc_kl`、`train/val_mode_kl`、`train/val_dur_dist_kl`、`train/val_ioi_dist_kl`、`train/val_ornament_kl`、`train/val_ce_total` 列；输出路径支持 `{seed}` 模板；训练结束自动调用 `plot_loss_curve.py` 出图。 |
| `configs/e3_L0.yaml` ~ `configs/e3_L5.yaml` | 每组一份覆盖层配置（超参/窗口/数据逐位继承 E2；四项分布损失权重统一取 1.0 或 0.0，未做逐组调参；含 `distribution` 分箱真源与 `{seed}` 产物路径）。 |
| `tests/test_loss_gradient.py` | E3 梯度单元测试（G0 辅助头停用 + G1~G5 五项损失梯度 + G6 掩码/调式链路）。 |

> 说明：L0 与 E2 的损失口径一致，但**不是** bit-identical 复现——E3 移除了两个辅助头，参数总量与初始化
> RNG 流发生变化。跨阶段对照以"指标量级与趋势"为准。

### 6.2 梯度单元测试（`tests/test_loss_gradient.py`）

- 分支：`experiment/E3_loss_ablation`；commit：`da8c9b2`；运行日期：2026-09-22；设备：cuda（RTX 3060 Laptop）
- 结果：**18/18 项 PASS**，无一项梯度为 0 → 允许启动 L0-L4 消融训练
- 明细产物：`results/e3_gradient_test.json`（git 排除），标准输出留档 `results/e3_gradient_test_stdout.txt`

| 损失项 | 期望受约束的头 | loss 初值 | pitch_head | duration_head | shift_head | 判定 |
|---|---|---|---|---|---|---|
| G1 音级(pitch-class)分布 KL | pitch_head | 0.6384 | **0.5522** | 0.0 | 0.0 | PASS |
| G2 时值(duration)分布 KL | duration_head | 2.5582 | 0.0 | **1.1561** | 0.0 | PASS |
| G3 IOI(time-shift)分布 KL | shift_head | 3.2681 | 0.0 | 0.0 | **1.1385** | PASS |
| G4 调式(mode)符合度 KL | pitch_head | 0.8486 | **0.6036** | 0.0 | 0.0 | PASS |
| G5 装饰音密度（L5 框架） | duration_head | 0.0718 | 0.0 | **0.0047** | 0.0 | PASS |

附加校验：G0 五个辅助头停用项 PASS；G6 shift 目标含 4 个 IGNORE_INDEX 掩码位（2 窗口 × 2：末尾伪标签 + EOS），
去掉掩码后 IOI 损失由 3.317358 → 3.327659（证明掩码确实生效）；调式真值解析（五空管音级 → 0）PASS。

### 6.3 L0 组结果（基础三路事件 CE，无分布损失）

- 配置：`configs/e3_L0.yaml`；命令：`train_boyatcn.py --mode compound --config configs/e3_L0.yaml --seed {2026,2027,2028}`
- 超参/窗口/掩码完全继承 E2：bs=16, lr=1e-3, StepLR(50, 0.5), patience=20, 上限 200 epoch，窗口 256 事件 / 训练步长 128 / 验证步长 256，shift 末尾伪标签仍用 ignore_index 屏蔽
- 模型：`NanyinBoYaTCN`（3,426,717 可训练参数；features_head / style_head 已停用）

| seed | 总轮数 | best epoch | best val 三路加权 CE | best 轮 train total | val token acc pitch / duration / shift (%) | 末轮 val CE | 末轮 train total |
|---|---|---|---|---|---|---|---|
| 2026 | 27 | 7 | **6.8788** | 5.9980 | 36.07 / 7.00 / 23.22 | 10.1936 | 1.8382 |
| 2027 | 26 | 6 | 6.8957 | 6.1677 | 35.29 / 6.15 / 23.65 | 10.0053 | 1.9213 |
| 2028 | 27 | 7 | 6.8795 | 5.9971 | 35.99 / 7.63 / 23.53 | 10.2115 | 1.7851 |
| **均值** | 26.7 | 6.7 | **6.8813** | 6.0543 | 35.78 / 6.93 / 23.47 | 10.1368 | 1.8482 |

**Sanity check（重要）**：L0 seed=2026 与 E2 基线（`ec4bd97`）**逐位复现** —— best val 损失同为 **6.8788**，三路 token 准确率同为 36.07 / 7.00 / 23.22。
原因：features_head / style_head 在 `models.py` 中位于三个生成头之后创建且不参与前向，停用不改变三个生成头的初始化随机数流，前向计算也未受牵连。→ L0 是 E2 的严格复现，后续跨组对照可信。

**过拟合现象（三籽一致）**：train total 5.998 → 1.838（降约 3.3×），val CE 6.879 → 10.194（升约 48%），best-val 全部出现在 epoch 6~7。
**duration 头依然没学到**：best 轮 token 准确率 6.15~7.63%（多数类基线约 10.8%），与 E2 的 7.00% 同水平 → 不是随机波动，而是结构性问题（404 细档长尾稀疏，219 类零样本）。

产物：`results/e3/L0_seed{2026,2027,2028}_train.log` 与 `_train_log.csv`、`checkpoints/e3/L0_seed*/best_valloss_model.pt`、`reports/e3/L0_seed*_loss_curve.png`。

### 6.4 L1 组结果（基础 + 音级 pitch-class 分布 KL，权重 1.0）

- 配置：`configs/e3_L1.yaml`；命令：`train_boyatcn.py --mode compound --config configs/e3_L1.yaml --seed {2026,2027,2028}`
- 与 L0 唯一差异：`loss_weight_pitch_class = 1.0`（作用于 pitch_head logits，12 维音级直方图 KL）

| seed | 总轮数 | best epoch | best val 三路加权 CE | best 轮 train total | best 轮 val pc_KL | val token acc pitch / duration / shift (%) | 末轮 val CE | 末轮 train total |
|---|---|---|---|---|---|---|---|---|
| 2026 | 27 | 7 | **6.8637** | 6.0668 | 0.0293 | 35.37 / 7.20 / 23.88 | 10.0572 | 1.8864 |
| 2027 | 26 | 6 | 6.8839 | 6.2422 | 0.0305 | 34.79 / 6.15 / 23.69 | 9.8120 | 2.0154 |
| 2028 | 27 | 7 | 6.8823 | 6.0815 | 0.0380 | 35.72 / 6.77 / 23.22 | 9.9663 | 1.8708 |
| **均值** | 26.7 | 6.7 | **6.8766** | 6.1302 | 0.0326 | 35.29 / 6.71 / 23.60 | 9.9452 | 1.9242 |

**与 L0 对照（均值）**：best val CE 6.8813 → 6.8766（**−0.0047，约 −0.07%，在种子波动范围内**）；
duration acc 6.93% → 6.71%（略降）、pitch acc 35.78% → 35.29%（略降）、shift acc 23.47% → 23.60%（略升）—— 三项变化均小于种子间离散度（duration 种子极差 L0 1.48pt / L1 1.05pt），**不能认为有实质改善**。
**过拟合**：末轮 val CE 10.1368 → 9.9452（略缓，约 −1.9%），train total 末值 1.8482 → 1.9242；best-val 仍集中在 epoch 6~7，早停轮数 26/27/27 与 L0 几乎相同 → 音级分布约束**没有改变过拟合的进程**。
诊断：验证集音级 KL 在 best 轮已降到 0.03 量级 —— 模型本身很快就匹配了窗口的音级直方图，该项约束的实际作用域很小（CE 目标已隐含相同信息）。

### 6.5 L2 组结果（基础 + 音级 KL + 调式 mode 符合度 KL，权重各 1.0）

- 配置：`configs/e3_L2.yaml`；命令：`train_boyatcn.py --mode compound --config configs/e3_L2.yaml --seed {2026,2027,2028}`
- 与 L1 唯一差异：`loss_weight_mode = 1.0`（同样作用于 pitch_head logits；目标为窗口所属南音调式的五声音级均匀先验）
- 调式真值分布（训练窗口 248 / 验证窗口 10，**全部落入"五空管"**）：见日志"[调式真值分布]"行 —— 数据集层面四调不平衡，mode 损失实际只向单一调式先验施加约束（如实记录为方法局限）

| seed | 总轮数 | best epoch | best val 三路加权 CE | best 轮 train total | best 轮 val pc_KL / mode_KL | val token acc pitch / duration / shift (%) | 末轮 val CE | 末轮 train total |
|---|---|---|---|---|---|---|---|---|
| 2026 | 27 | 7 | 6.8769 | 6.3797 | 0.0335 / 0.1604 | 34.75 / 7.39 / 24.08 | 9.8894 | 2.2360 |
| 2027 | 26 | 6 | 6.8821 | 6.5144 | 0.0337 / 0.1658 | 34.20 / 5.76 / 24.12 | 9.5514 | 2.3861 |
| 2028 | 27 | 7 | **6.8604** | 6.3890 | 0.0387 / 0.1620 | 35.10 / 7.59 / 24.16 | 9.8048 | 2.2167 |
| **均值** | 26.7 | 6.7 | **6.8731** | 6.4277 | 0.0353 / 0.1627 | 34.68 / 6.91 / 24.12 | 9.7485 | 2.2796 |

**与 L0 对照（均值）**：best val CE 6.8813 → 6.8731（**−0.0082，约 −0.12%**，仍在种子波动内）；
pitch acc 35.78% → 34.68%（降 1.1pt，三籽一致下降，是**负向**副作用）、duration 6.93% → 6.91%（持平）、shift 23.47% → 24.12%（升 0.65pt，三籽高度一致：24.08/24.12/24.16）。
**过拟合**：末轮 val CE 10.1368 → 9.7485（降约 3.8%，为三组中最缓和），但 train total 末值反而更高（2.28 vs 1.85），说明优化目标被分布项抬高而 CE 泛化略好；best-val 仍集中在 epoch 6~7。
诊断：mode_kl 停在 0.16 附近不再下降 —— 目标"五声音级均匀先验"与真实旋律音级分布本就不完全吻合（存在非五声音级的经过/装饰音），该项存在不可消的残差下限，持续施压会把 pitch 分布往先验拉偏，这与 pitch acc 下降一致。

### 6.6 L3 组结果（基础 + 时值分布 KL + IOI 分布 KL，权重各 1.0）

- 配置：`configs/e3_L3.yaml`；命令：`train_boyatcn.py --mode compound --config configs/e3_L3.yaml --seed {2026,2027,2028}`
- 与 L0 差异：`loss_weight_duration_dist = 1.0`（duration_head logits，402 类 → 14 个秒级 bin）、`loss_weight_ioi_dist = 1.0`（shift_head logits，1002 类 → 15 个 IOI 秒级 bin；末尾伪标签位继续 ignore）
- 设计意图：直接给 E2 最薄弱的 duration 头施加"粗粒度时值/节奏分布"约束，绕开 404 细档长尾稀疏

| seed | 总轮数 | best epoch | best val 三路加权 CE | best 轮 train total | best 轮 val dur_KL / IOI_KL | val token acc pitch / duration / shift (%) | 末轮 val CE | 末轮 train total |
|---|---|---|---|---|---|---|---|---|
| 2026 | 28 | 8 | 6.8635 | 6.0427 | 0.0563 / 0.0729 | 35.56 / 6.85 / 24.35 | 9.6284 | 2.0880 |
| 2027 | 29 | 9 | 6.8657 | 5.8607 | 0.0580 / 0.0724 | 36.93 / 5.84 / 23.65 | 9.8713 | 1.9694 |
| 2028 | 27 | 7 | **6.8418** | 6.2067 | 0.0560 / 0.0644 | 34.63 / 6.89 / 24.20 | 9.2951 | 2.2578 |
| **均值** | 28.0 | 8.0 | **6.8570** | 6.0367 | 0.0568 / 0.0699 | 35.71 / 6.53 / 24.07 | 9.5983 | 2.1051 |

**与 L0 对照（均值）**：
- best val CE 6.8813 → **6.8570（−0.0243，约 −0.35%）**，为目前四组中降幅最大，但三籽极差 0.0239（6.8418~6.8657）与组间差值同量级 → **尚不足以判定显著**；
- 末轮 val CE 10.1368 → **9.5983（−5.3%）**，且早停轮数由 26.7 → 28.0 —— 分布约束让训练多撑了 1~2 轮，**过拟合速度确实减缓**（这是迄今唯一较稳定的正向观测）；
- duration acc 6.93% → 6.53%（**反而下降 0.4pt**，种子极差 1.05pt）→ **分布约束没能改善 duration token 级准确率**，原假设（粗粒度 bin 能救长尾细档）未被证实；
- pitch acc 35.78% → 35.71%（持平）、shift acc 23.47% → 24.07%（升 0.6pt，与 L2 一致的顺序）。
诊断：dur_KL（0.056）与 IOI_KL（0.07）在早期就被压到很低 —— 秒级 bin 聚合后的 GT 直方图本身很集中，模型用粗粒度分布几乎"免费满足"，因此只起到轻微正则作用，没能传递细档判别信息。


### 6.7 L4 组结果（基础 + 音级 KL + 调式 KL + 时值分布 KL + IOI 分布 KL，四项全开）

- 配置：`configs/e3_L4.yaml`；命令：`train_boyatcn.py --mode compound --config configs/e3_L4.yaml --seed {2026,2027,2028}`

| seed | 总轮数 | best epoch | best val 三路加权 CE | best 轮 train total | best 轮 val pc / mode / dur / IOI KL | val token acc pitch / duration / shift (%) | 末轮 val CE | 末轮 train total |
|---|---|---|---|---|---|---|---|---|
| 2026 | 27 | 7 | 6.8624 | 6.5657 | 0.0404 / 0.1638 / 0.0593 / 0.0672 | 34.36 / 6.73 / 24.71 | 9.0828 | 2.7056 |
| 2027 | 29 | 9 | **6.8610** | 6.2304 | 0.0364 / 0.1602 / 0.0657 / 0.0694 | 37.35 / 5.88 / 23.96 | 9.3381 | 2.4238 |
| 2028 | 27 | 7 | 6.8758 | 6.5887 | 0.0458 / 0.1586 / 0.0556 / 0.0723 | 34.16 / 7.47 / 24.39 | 9.0258 | 2.7588 |
| **均值** | 27.7 | 7.7 | **6.8664** | 6.4616 | 0.0409 / 0.1609 / 0.0602 / 0.0696 | 35.29 / 6.69 / 24.35 | 9.1489 | 2.6294 |

**与 L0 对照（均值）**：best val CE 6.8813 → 6.8664（−0.0149，约 −0.22%，低于 L3 的降幅）；
pitch acc 35.78% → 35.29%（降 0.5pt）、duration 6.93% → 6.69%（降 0.24pt）、shift 23.47% → 24.35%（**升 0.88pt，四组中最高**）。
**过拟合**：末轮 val CE 10.1368 → **9.1489（−9.7%，四组中降幅最大）**，早停轮数 26.7 → 27.7 → 四项分布损失叠加起来确实延缓了验证损失的恶化速度。
诊断：四项叠加并未产生"1+1>1"的效果 —— best val CE 的最好单组是 **L3（6.8570，仅时值+IOI）**，L4 反而略高；
mode_KL 稳定停在 0.16 附近（先验与真实音级分布存在不可消残差），它与音级/调式约束一起把 pitch 往"均匀五声先验"拉偏，抵消了 L3 在时值上的收益。

### 6.8 L5（装饰相关损失）：仅代码框架，不参与正式结果

- 框架实现：`loss_nanyin.ornament_density_distribution_loss`（duration_head softmax 在"短音类别"上的概率质量 vs GT 短音比例，MSE）；配置 `configs/e3_L5.yaml`（`loss_weight_ornament = 1.0`，短音阈值 0.15s）
- 梯度单元测试 G5 已 PASS（duration_head 梯度范数 0.0047 ≠ 0）
- **未执行训练、不纳入上述任何对照表**：装饰音真值来自 `dataset_nanyin.build_target_dict` 的"短音 + 邻音跳进"启发式（BasicPitch AMT 转录派生），**未经人工核验**；按实验规程，只有完成小规模人工核验后 L5 才能作为正式结果。

### 6.9 E3 阶段小结（对比 E2 基线）

**跑完的组合**：L0 / L1 / L2 / L3 / L4 × seed {2026, 2027, 2028} = 15 次训练（每次 ~4.5 分钟，GPU RTX 3060 Laptop）。
**L5** 仅实现代码框架 + 梯度测试，**未训练，不进正式结果**（装饰音真值无人工核验）。

三籽均值汇总（val CE = 验证集三路加权 CE，与 E2 同口径）：

| 组 | 分布损失 | best val CE | 早停轮数 | pitch / duration / shift acc (%) | 末轮 val CE |
|---|---|---|---|---|---|
| E2 基线 `ec4bd97`（籽 2026） | 无 | 6.8788 | 27 | 36.07 / 7.00 / 23.22 | 10.19 |
| L0 | 无 | 6.8813 | 26.7 | 35.78 / 6.93 / 23.47 | 10.1368 |
| L1 | 音级 KL | 6.8766 | 26.7 | 35.29 / 6.71 / 23.60 | 9.9452 |
| L2 | +调式 KL | 6.8731 | 26.7 | 34.68 / 6.91 / 24.12 | 9.7485 |
| L3 | 时值+IOI 分布 KL | **6.8570** | 28.0 | 35.71 / 6.53 / 24.07 | 9.5983 |
| L4 | 四项全开 | 6.8664 | 27.7 | 35.29 / 6.69 / **24.35** | **9.1489** |

**结论（对比 E2）**

1. **没有一组在最优验证损失上取得实质改善**：相对 L0 降幅 0.07%~0.35%，均小于组内三籽极差（0.015~0.024），落在种子波动内。最好的 L3 也只到 6.8570 vs E2 的 6.8788。
2. **唯一较稳定的正向效应是"过拟合被延缓"**：末轮 val CE 由 10.14（L0，与 E2 的 10.19 一致）单调降到 9.15（L4，−9.7%），早停轮数 26.7 → 28.0，best epoch 6.7 → 8.0。收益是"更耐训"，而非"泛化上限更高"。
3. **duration 头依然没救**（本阶段最重要的负向结果）：各组 duration acc 6.53~6.93%，**没有任何一组超过 L0**，与 E2 的 7.00% 同水平；专门设计的粗粒度秒级 bin 时值/IOI 分布 KL（L3）也未提升。根因是 404 细档长尾（219 类零样本）+ 任务本身对齐问题，不是缺分布监督。
4. **shift acc 有小幅重复提升**（23.47 → 24.35，L2/L3/L4 三籽高度一致），幅度约 +0.6~0.9pt。
5. **含调式先验的组 pitch acc 略降**（35.78 → 34.68/35.29）：mode_KL 停在 0.16 不降，均匀五声先验与真实旋律不吻合，存在不可消残差。
6. **L0 籽 2026 逐位复现 E2**（6.8788 / 36.07 / 7.00 / 23.22），证明停用 features_head、style_head 未扰动主链路，本阶段跨阶段对照可信。

**执行/依赖确认**：训练前 `tests/test_loss_gradient.py` 18/18 PASS；支路 `feature/E2_window_training`（`ec4bd97`）与三条旧预实验分支未被改动；
早停/最优判定统一监控"三路加权 CE"以保跨组可比；全部窗口/掩码/超参继承 E2。

**commit 记录**：`da8c9b2`（代码+配置+梯度测试脚本）→ `a00c2d2`（梯度测试日志）→ `a81edef`（L0）→ `7191866`（L1）→ `ebf9d23`（L2）→ `06e1d00`（L3）→ `20c6d8f`（L4+L5）。

**下一步（留给 E4 及以后）**：① duration 词表粗化/对数分箱或改连续时长回归；② 扩大验证集（≥3~4 首）降低 best-val 方差；③ mode 先验改平滑分布；④ 若要继续用分布损失，先做权重量级搜索（1/5/10）。

## 7. E4：固定权重 + temperature / top-p 采样超参网格搜索（分支 `experiment/E4_sampling_search`）

### 7.0 前置背景与约束

- **E4 不做任何训练**。模型权重全部固定取自 **E3-L3 三组随机种子（2026/2027/2028）的 best-val 早停 checkpoint**
  （早停口径 = 验证集三路加权 CE，best epoch 8/9/7，val CE 6.8635 / 6.8657 / 6.8418）。
- **只改推理采样超参**（temperature / top-p）。模型结构、词表、窗口切分、数据集划分、掩码规则全部继承 E2/E3。
- **重要约束（已知不可通过采样修复）**：E2/E3 记录的底层缺陷 —— duration / time-shift 词表长尾、细档 token 识别准确率低
  （duration head 验证集 token acc 仅 6.53%），**采样策略只能改变生成多样性与非法事件比例，无法提升细档判别能力**。
- 禁止加载非 best-val 的过拟合权重；禁止训练/finetune/改权重；评估禁止开任何 MIDI 规则后处理或时长钳位。

### 7.1 代码改动（commit `待补`）

| 文件 | 改动 |
|---|---|
| `symbolic_features.py` | **新增**。46 维符号统计特征（pitch-class 12 + 有向音程 12 + log-时值 8 + log-IOI 8 + 音高统计 2 + 音符密度 1 + 调式符合度 1 + 装饰音占比 2），维数分块与 exp3 记录一致；另有 `pairwise_distance_matrix()` 供多样性度量。注：原 `feature_extractor.py` 实为**音频**特征（librosa F0/MFCC），不含该 46 维符号特征；exp3 的临时脚本未入库，故按文档口径重建。 |
| `configs/e4_sampling_search.yaml` | **新增**。网格 T∈{0.6,0.8,1.0,1.2} × top-p∈{0.7,0.8,0.9,0.95}；权重模板 `checkpoints/e3/L3_seed{seed}/best_valloss_model.pt`；窗口继承 E2/E3（256 事件、val 步长 256 → 10 窗口）；prompt 16 → 总长 256；`postprocess.*` 四开关全部 False（仅预留）；非法事件/重复率/多样性/产物路径配置。 |
| `generate_music.py` | `_filter_and_sample` 新增 `generator` 参数；`generate_compound_autoregressive` 新增 `temperature` / `top_p` / `top_k` / `generator` / `prepend_bos`（默认沿用 e1 基线 → 旧行为逐位不变）；新增 E4 小节：`build_midi_name()`、`_rng_seed()`、`load_val_windows()`、`load_e4_model()`、`run_sampling_search()`；新增 `--mode sampling_search` CLI（保留原 main 不动）。输入序列 Prepend BOS（`[BOS] + 事件序列`），与训练样本输入格式一致。 |
| `tests/test_sampling_consistency.py` | **新增**。T0 配置约束 / T1 可复现 / T2 随机流生效 / T3 采样参数生效 / T4 prompt 对齐 / T5 窗口口径 / T6 命名规范 / T7 46 维特征，共 23 项。 |

### 7.2 单元测试结果（**23/23 PASS**，设备 cuda，commit `待补`）

明细：`results/e4/e4_sampling_consistency_test.json`

- **T1 可复现**：同权重 + 同 (T=0.8, p=0.9) + 同 rng_seed=1849095758 跑两次 → 三路 256 事件序列**逐位相同**。
- **T2 随机流生效**：换 rng_seed → 序列不同（证明可复现不是"退化成贪心"）。
- **T3 采样参数生效**：T 0.8→1.2、top-p 0.9→0.7 在同一 rng 下序列均改变。
- **T4 prompt 对齐**：生成序列前 16 个事件 == 窗口 GT 前 16 个事件（pitch/duration/shift 三路逐位）；总长 = 256 事件。
- **T5 窗口口径**：10 个验证窗口（dapu14kouhuangtian 4 + dapu15wujinjiao 6），窗口长度 256，编号/起始事件号与 `CompoundEventWindowDataset` records 一致，窗口数与 E3 `e3_window_stats.csv` 一致。
- **T0 约束**：postprocess 四开关全 False、prompt=16、target=256、prepend_bos=True、评估 skip_prompt=16、top-k 全关、权重模板指向 E3-L3 best_val。
- 每条序列的随机数流由独立 `torch.Generator` 决定，种子公式 `rng_seed = base + model_seed·97³ + grid_index·97² + window_index`（base=20260）。

### 7.3 批量生成与评估实现

| 文件 | 说明 |
|---|---|
| `evaluate_e4_sampling.py` | **新增**。E4 Tier-1 评估驱动：读 `e4_events.jsonl` 的原始三路 token（**不经 MIDI 往返**，从而保留非法 token），对每条序列调用 `nanyin_metrics.evaluate_single_song`（窗口 GT 由训练同源顺序构造，skip_prompt=16）+ `compute_composite_score`；统计非法事件率（pitch≥128 / duration≥402 或 =0 / shift≥1002 或 =0，窗口末 shift 伪标签不计）、4-gram 重复率、46 维特征；计算两类多样性（同窗口换采样参数 `interDiv`、同配置换训练种子 `seedDiv`）与到 GT 的 46 维距离；输出明细/聚合 CSV+JSON 与三张热力图。 |
| `generate_music.py`（补充） | 原始三路事件序列落盘 `results/e4/e4_events.jsonl`；按已存在组合断点续跑（避免 MIDI 与事件流水不一致）。 |

生成侧：16 组 × 3 种子 × 10 窗口 = **480 条序列 / 480 个 MIDI**，完整网格重跑一次耗时 **32.7 分钟**（GPU RTX 3060 Laptop）。
（注：首次运行中断后在重跑时因事件流水被重建，最终以前后一致的一次完整网格为准，480/480 齐全。）

### 7.4 结果与最优配置

- **composite**（每组 30 样本均值 ± std，GT 天花板 82.76 ± 1.60）：
  最低 58.05（T=0.8/p=0.8），最高 **67.39（T=1.2/p=0.95）**；67.21（1.2/0.9）、66.81（1.0/0.95）紧随。
  极差 9.34 分；最好的采样配置仍比 GT 低 15.4 分。
  提升几乎全部来自 `tempo_accuracy`（52.6~56.3 → 79.98~85.18），rhythm_consistency / pitch_stability / mode_matching 基本不动。
  组内 std 随采样熵上升而变小：7.07（0.6/0.7）→ 2.78（1.2/0.95）。
- **非法事件率：全部 16 组 = 0.0000**（480 条 × 240 事件 = 115,200 token 零非法）。
  原因：p ≤ 0.95 的 nucleus 截断把长尾（PAD / 溢出档 / 特殊 token）一并裁掉。
  ⚠ 不可外推到 T > 1.2 或 p > 0.95（未测试）。
- **4-gram 重复率**随熵单调下降 0.865 → 0.063：**11/16 组重复率 > 0.5（循环退化）**，集中在 T ≤ 0.8 且 p ≤ 0.8 的区域
  （另含 0.6/0.9、0.6/0.95、0.8/0.9、1.0/0.7、1.0/0.8、1.2/0.7）。
- **局部结构反向崩塌**：(transition_smoothness + melody_coherence)/2 由 45.9（0.8/0.7）降到 1.8（1.2/0.95）。
  这两项**不在 composite 公式内** → "composite 最高"恰是"局部旋律最碎"，因此不能做"越高越好"的单向结论。
- **多样性两个口径结论相反**（方法学陷阱）：`interDiv` 与 `seedDiv` 都在**低熵**配置最高（1.18 / 1.38），
  因为退化循环各不相同；高熵时反而收敛（0.34）。真正有信息量的是到 GT 的 46 维距离，随熵单调下降 1.62 → 0.40。
- **多准则 winner 不一致**：只看 composite → 1.2/0.95；composite+局部结构（或再加多样性）→ **0.6/0.8**（重复率 0.835 的退化配置，不可采纳）。
- **逐种子稳健性**：三个种子的配置排序 Spearman ρ = 0.803 / 0.868 / 0.803；跨种子 composite std 在高熵区最小
  （1.2/0.95 = 0.41、1.0/0.95 = 0.78），低熵区可达 6.45~7.07。

**选定综合最优：`temperature = 1.0, top-p = 0.95`**（composite 66.81 ± 2.62）
理由：① 距最高的 67.39 仅差 0.58，小于各自 std（2.62/2.78）→ 统计上并列第一梯队；
② 跨种子 std 0.78（全网格第 2 稳）→ 不依赖某一份权重；③ 重复率 0.165，非退化；
④ top-3 中局部结构最好（struct 5.90 vs 3.55 / 1.77）；⑤ T=1.0 不对模型分布做额外形变，方法论最中性；
⑥ 到 GT 的 46 维距离 0.512，接近最优 0.395。
备选：`1.2/0.9`（只求分布贴合/节奏自然，重复率 0.150）；`0.8/0.95`（需要局部连贯时，struct 13.40 最高，composite 63.47）。
不建议使用：低熵退化区（重复率 > 0.5）；多样性过低：1.2/0.95 与 1.2/0.9（seedDiv 0.336 / 0.388 为全网格最低两档）。

### 7.5 E4 阶段小结

1. 在固定 E3-L3 权重、固定 prompt（16 GT 事件 → 256 总长）、零后处理的前提下，
   **采样超参对 Tier-1 指标的影响是可测且稳健的**（Δcomposite 达 9.34 分，跨种子排序 ρ ≈ 0.8~0.87）。
2. **真正的权衡不是"质量 vs 非法率"，而是"循环退化 vs 局部结构崩塌"**：
   低熵（T ≤ 0.8 且 p ≤ 0.8）→ 高重复率退化；高熵（T ≥ 1.2 且 p ≥ 0.9）→ composite 最高但旋律结构散掉。
   本网格 16 组**没有任何一组**同时满足"高 composite + 低重复 + 高 local 结构"。
3. 非法率在本网格内恒为 0，**没有观察到**"某些参数产生大量非法时值/shift 事件"的证据（范围外未验证）。
4. 采样仍然救不了训练缺陷：duration 细档 token acc 仍 6.53%，生成与 GT 的 composite 差距仍达 15.4 分。
   **下一步（E5 及以后）应回到词表 / 训练侧**：duration 词表粗化或对数分箱、粗粒度 IOI 词表、扩大验证窗口数。

commit 记录：`9f8ce4f`（46 维特征 + E4 配置 + 采样参数化生成入口 + 一致性测试 23/23 PASS）。

---

## 7. E5 阶段：模型公平对比 + 自回归误差累积漂移（分支 `experiment/E5_model_compare_drift`）

> **分支来源**：`experiment/E4_sampling_search` HEAD 新建。E2 / E3 / E4 分支与产物全部冻结只读。
> **E5 允许新增文件**：`configs/e5_model_drift.yaml`、`tests/test_e5_infer_consistency.py`、`reports/E5_model_compare_drift_summary.md`（其余均为既有文件的定向扩展）。

### 7.0 前置背景与约束（写入报告头部）

- E5 全程**不训练、不微调、不修改权重**，推理权重全部取自 best-val 早停 checkpoint；
  采样超参**固定为 E4 结论配置**：`temperature=1.0`、`top-p=0.95`、`top-k=0`。
- 本实验属于**推理层面评估**，无法修复训练阶段的底层缺陷：duration / time-shift 词表长尾、
  细档 token 识别准确率低（E2 记录 404 类中 219 类零样本；E3 五组消融下 duration token acc
  稳定在 6.5~7.0%，无一超过 L0 基线）。所有观测现象如实记录，不做美化。

### 7.1 阻塞点：仓库缺少"复合事件口径"的 MusicMamba 权重（已获用户授权补训）

任务A 要求"完全对等条件下的公平对比"，但开工核查时发现：

| 现有 MusicMamba 权重 | 状态 | 能否用于 E5 |
|---|---|---|
| `checkpoints/musicmamba/`、`exp2_musicmamba/`、`grid_musicmamba_*` | E1 之前的**旧单 token 表达**（vocab 128）产物 | ❌ 事件表示空间不兼容，指标不可比 |
| `train_musicmamba.py` | 旧 NOEF 单 token 管线，无 `--mode compound` | ❌ 无复合事件训练路径 |
| `models.NanyinMusicMamba` | 已具备三路输出头（`pitch/duration/shiftlogits`），与 `NanyinBoYaTCN` 输出结构一致 | ✅ 可直接替换骨干 |

结论：**缺的只是"同等条件下的训练权重"**。经向用户请示并获明确授权（"授权前置训练 MusicMamba"），
E5 增设一个**前置补训子阶段**，之后再进入纯推理评估。这是获得公平对比结果的唯一可行路径。

### 7.2 代码改动（前置补训）

`train_boyatcn.py`（**定向扩展，未整文件重写**，E2/E3 的 `train()` 与 `train_compound()` 主逻辑不动）：

| 位置 | 改动 | 目的 |
|---|---|---|
| import | `from models import NanyinBoYaTCN, NanyinMusicMamba` | 支持骨干切换 |
| `train_compound()` | 新增形参 `model_name: str = "boyatcn"`，默认值使 E2/E3 行为**逐位不变** | 引入唯一自变量（骨干） |
| `_fmt()` 旁新增 `_fmt_out()` | 输出文件名追加 `_musicmamba` 后缀 | **防止覆盖 E3 既有日志 / CSV / PNG**（`results/e3/*_musicmamba.log` 为新增文件，E3 原文件未被改写） |
| 模型构造 | `NanyinBoYaTCN() if boyatcn else NanyinMusicMamba()` | 公平替换 |
| CLI | 新增 `--model {boyatcn,musicmamba}` | 命令行选择 |

训练命令（**数据与超参与 E3-L3 逐位一致**：同一 `configs/e3_L3.yaml`、同一窗口切分、同一三路加权 CE
1.0/0.7/0.7 + 时值分布 KL 1.0 + IOI 分布 KL 1.0，同一个 val-CE 早停 patience=20）：

```
venv/Scripts/python.exe train_boyatcn.py --mode compound --model musicmamba \
    --config configs/e3_L3.yaml --checkpoint-dir checkpoints/e5/musicmamba/L3_seed{seed} --seed 2026|2027|2028
```

### 7.3 前置补训结果（MusicMamba，三籽）

| seed | 总轮数 | best epoch | best val CE | 该轮 pitch / duration / shift acc (%) | 末轮 val CE |
|---|---|---|---|---|---|
| 2026 | 26 | 6 | 6.7406 | 40.70 / 6.96 / 24.24 | 9.2740 |
| 2027 | 27 | 7 | 6.7637 | 42.06 / 7.28 / 22.16 | 9.4574 |
| 2028 | 27 | 7 | **6.6820** | 42.22 / 7.39 / 24.12 | 9.4855 |
| **均值** | 26.7 | 6.7 | **6.7288** | 41.66 / 7.21 / 23.51 | 9.4056 |

对照 E3-L3 的 BoYaTCN 三籽均值：best val CE **6.8570**、acc 35.71 / 6.53 / 24.07。
→ 在**完全相同的数据与损失**下，MusicMamba 的验证 CE 低 0.128、pitch token acc 高约 6 个百分点
（2.18M 参数 vs BoYaTCN 3.43M）。**注意：这是 teacher-forcing 下的下一 token 预测能力，
不等于自回归生成质量**，后者由任务 A/B 的实测给出。

坑记录：seed=2028 首次训练时被 IDE 会话回收中断（日志停在 epoch 11），已重跑到早停完成；
最终 best epoch=7 与中断前一致，ce 值相同，结果未受影响。

产物：`checkpoints/e5/musicmamba/L3_seed{2026,2027,2028}/best_valloss_model.pt`、
`results/e3/L3_seed{...}_train_musicmamba.log` + `_train_log_musicmamba.csv`、
`reports/e3/L3_seed{...}_loss_curve_musicmamba.png`。

### 7.4 E5 代码改动（推理与评估）

| 文件 | 改动 |
|---|---|
| `configs/e5_model_drift.yaml`（新增） | 两组模型权重模板（各 3 籽 best-val）；prompt=16；`target_len_events_list: [32,64,128,256]`（**总长度口径**，含 prompt）；采样固定 T=1.0/p=0.95/k=0；`postprocess.*` 全 false；评估 `skip_prompt=16`；46 维降维超参（StandardScaler + UMAP n_neighbors=8/min_dist=0.1/seed=42 + t-SNE perplexity=5/max_iter=2000/init=pca/seed=42，与 exp3 完全一致） |
| `generate_music.py` | 新增 `E5_DEFAULT_CONFIG`、`build_drift_midi_name()`、`resolve_e5_models()`、`load_e5_model()`（两种骨干）、`run_model_drift_generation()`；文件名 `{song_id}_w{w}_{model}_seed{s}_len{L}.mid`；每条序列独立 `torch.Generator`（`_rng_seed` 确定性公式）；MIDI 与原始三路 token 双写（评估直读 token，不经过 MIDI 往返 → 保留非法 token）；新增 CLI `--mode model_drift --models --lengths` |
| `evaluate_e4_sampling.py` | 新增 `--stage {e4,e5}`；**E4 的 `evaluate_all` / `_heatmap` / 主流程一行未改**，e5 走新增分支：`evaluate_all_e5()`（Tier-1 逐条评估 → 明细 CSV → 按 (模型 × 长度) 聚合 mean±std + 跨籽多样性）、`_plot_drift_lines()`（任务B 四面板折线图）、`_cluster_info()` + `_plot_reduction()`（任务C UMAP/t-SNE + 簇中心到 GT 簇中心距离）。gtDist 采用**与生成片段同起点、等长**的 GT 区域，保证不同续写长度可比 |
| `tests/test_e5_infer_consistency.py`（新增） | T0 配置与权重约束自检、T1 可复现、T2 prompt 对齐、T3 续写长度、T4 换种子序列变化（排除贪心退化）、T5 命名规范 |

### 7.5 单元测试结果（**28/28 PASS**）

判定明细：`results/e5/e5_infer_consistency_test.json`

| 编号 | 校验项 | 结果 |
|---|---|---|
| T0-1 | `postprocess.*` 全部关闭（评估使用原生输出） | PASS |
| T0-2 | 采样固定 T=1.0 / top_p=0.95 / top_k=0 | PASS |
| T0-3 | prompt_len=16，窗口 256 / 步长 256（沿用 E2/E3/E4） | PASS |
| T0-4 | 参与对比模型 = boyatcn + musicmamba | PASS |
| T0-5 | 两组模型 best-val 早停权重齐备（各 3 籽，共 6 个） | PASS |
| T0-6 | 验证窗口 = 10（dapu14×4 + dapu15×6），窗口长 256 | PASS |
| T1-boyatcn / T1-musicmamba | 同 (权重, rng, 采样参数) 两次推理三路序列**逐位一致** | PASS |
| T4-boyatcn / T4-musicmamba | 换 rng 种子后序列不同（未退化成贪心） | PASS |
| T2-<model>-w0/w5/w9 | 生成序列前 16 事件 == 对应窗口 GT 前 16 事件（三路逐位） | PASS ×6 |
| T2-<model>-region | 评估区域 = 续写片段（如 64 − 16 = 48 事件） | PASS ×2 |
| T3-<model>-32/64/128/256 | target_len 正确输出 32 / 64 / 128 / 256 长度（三路一致） | PASS ×8 |
| T5-1 / T5-2 | 文件名规范 `{song_id}_w{w}_{model}_seed{s}_len{L}.mid`，组合不重复 | PASS ×2 |

→ **结论：全部 PASS，允许启动批量生成。**


### 7.6 批量生成结果（任务A/B 数据）

命令：`venv/Scripts/python.exe generate_music.py --mode model_drift --config configs/e5_model_drift.yaml`
耗时 7.3 分钟，**240 条序列全部生成，无缺失**：

- MIDI：`results/e5/generated_midi/{song_id}_w{w}_{model}_seed{s}_len{L}.mid`（240 个）
- 清单：`results/e5/e5_generate_list.csv`（240 行；字段：曲号/窗口号/窗口起止事件/模型名/训练种子/checkpoint/采样参数/rng_seed/prompt_len/target_len/MIDI 路径）
- 原始三路 token：`results/e5/e5_events.jsonl`（评估直读 token，保留非法 token，不经 MIDI 往返）

### 7.7 Tier-1 批量评估结果

GT 天花板（GT vs 自身，不跳过 prompt）：**composite = 82.76 ± 1.60**
（mode 86.88、rhythm 60.29、tempo 100.00、ornament_score 76.14、ornament_density 87.96）

#### 任务A（len=256，每模型 30 样本 = 3 籽 × 10 窗口）

| 指标 | BoYaTCN | MusicMamba | Δ |
|---|---|---|---|
| composite | 66.87 ± 2.87 | 65.88 ± 1.94 | +0.99 |
| mode_matching_score | 83.96 ± 9.94 | **86.32 ± 8.78** | −2.36 |
| rhythm_consistency | 56.92 ± 2.45 | 56.99 ± 1.74 | −0.07 |
| tempo_accuracy | **84.69 ± 11.39** | 74.96 ± 10.26 | +9.73 |
| ornament_score | 73.29 ± 1.59 | 72.85 ± 0.96 | +0.44 |
| ornament_density | **77.93 ± 7.57** | 74.71 ± 3.61 | +3.23 |
| pitch_accuracy | 8.49 ± 1.99 | 7.74 ± 2.05 | +0.75 |
| gtDist（46 维，同起点等长 GT） | **0.527 ± 0.131** | 0.605 ± 0.136 | −0.079 |
| 4-gram 重复率 | 0.190 | 0.216 | −0.026 |
| 非法 token 占比 | 0.0000 | 0.0000 | 0 |

配对差（同窗口 + 同训练种子）：30 对，均值 **+0.99**，标准差 2.90，BoYaTCN 胜 **18/30** 对
→ **差距落在噪声范围内，判定为"基本持平、BoYaTCN 略占优"，不构成显著差异。**

**重要负向观察（如实记录）**：MusicMamba 的 teacher-forcing 指标明显更好
（val CE 6.7288 vs 6.8570；验证 pitch token acc 41.7% vs 35.7%；参数量少 36%），
但自回归生成时的 Tier-1 质量并未占优（composite 略低、重复率更高、跨种子多样性更低）
→ **teacher-forcing 能力 ≠ 自回归生成质量**。

#### 任务B（自回归漂移）

| 指标 | 模型 | 32 | 64 | 128 | 256 | △(256−32) |
|---|---|---|---|---|---|---|
| composite | BoYaTCN | 67.74 | 66.45 | 66.70 | 66.87 | −0.87 |
| | MusicMamba | 66.98 | 66.80 | 64.43 | 65.88 | −1.11 |
| gtDist | BoYaTCN | 1.106 | 0.705 | 0.657 | 0.527 | −0.579 |
| | MusicMamba | 1.147 | 0.739 | 0.764 | 0.605 | −0.542 |
| 4-gram 重复率 | BoYaTCN | 0.041 | 0.117 | 0.160 | **0.190** | **+0.149** |
| | MusicMamba | 0.054 | 0.122 | 0.224 | **0.216** | **+0.162** |
| pitch_accuracy | BoYaTCN | 14.58 | 11.11 | 9.08 | 8.49 | −6.10 |
| | MusicMamba | 13.54 | 10.07 | 8.78 | 7.74 | −5.81 |
| 跨种子多样性 | BoYaTCN | 1.121 | 0.723 | 0.607 | 0.508 | −0.613 |
| | MusicMamba | 1.153 | 0.751 | 0.693 | 0.418 | −0.735 |
| 非法 token 占比 | 两组 | 0 | 0 | 0 | 0 | 0 |

要点：
1. **漂移的真实证据 = 重复率单调上升（~3.6×）+ 逐位一致率单调下降**；二者互为印证。
2. **composite 对漂移不敏感**（−0.9 / −1.1），不能单独判断长序列质量。
3. **gtDist 随长度"下降"是小样本统计假象**：len=32 评估区仅 16 个事件，
   46 维中 40 维直方图分量方差极大 → 距离被系统性高估。跨长度不可比，同长度内才有效。
4. **非法率恒 0 的原因**：top-p=0.95 把尾部（含特殊 token）整体截断，
   240 条序列续写片段中 pitch≥128 出现 0 次、**EOS(129) 出现 0 次**
   → 副作用：生成**从不自然终止**，长度完全由人为 `target_len` 截断；
   且该维度在本实验**没有分辨力**（0 值不代表模型不会输出非法 token）。
5. **长序列退化更显著的是 MusicMamba**：重复率全程更高（256 处 0.216 vs 0.190，128 处差最大）、
   跨种子多样性掉更快（0.418 vs 0.508）、len=128 出现 mode 谷值 78.83（σ=15.87，全表最差）
   与 composite 全表最低 64.43（256 处回升，波动大于 BoYaTCN）。

### 7.8 任务C：46 维 UMAP / t-SNE

- 样本：**280 点** = 10 窗口 GT（按各长度同起点等长截取，GT-len32/64/128/256 各 10 点）
  + 两组模型 × 3 籽 × 4 长度 × 10 窗口；
- 预处理 StandardScaler；UMAP(n_neighbors=8, min_dist=0.1, seed=42)；
  t-SNE(perplexity=5, max_iter=2000, init=pca, seed=42)，**与 exp3 完全一致**；
- 特征**只取续写片段**（跳过 16 个 prompt），与评估口径严格对齐；
- **未混入 exp3 旧 NOEF 单 token 样本**（两套事件表示特征空间不兼容）。

簇中心到同长度 GT 簇中心的距离：

| 降维 | 模型 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|
| UMAP | BoYaTCN | 1.394 | 3.059 | 3.532 | 3.104 |
| UMAP | MusicMamba | 1.753 | 3.907 | 4.104 | 4.390 |
| t-SNE | BoYaTCN | 21.11 | 48.24 | 52.73 | 48.71 |
| t-SNE | MusicMamba | 27.45 | 55.60 | 68.25 | 68.95 |

定性解读：两种方法方向一致 —— 生成点随续写长度远离 GT 簇，MusicMamba 在每个长度上都更远；
len=128 的 MusicMamba 最散最远，与 §7.7 该长度 mode/composite 谷值吻合。
⚠ len=32 距离小有一部分是**小样本效应**；受样本量限制，**降维图仅作定性辅助，不用于定量论断**。

### 7.9 E5 结论

1. 同等条件下 MusicMamba teacher-forcing 更强（CE 更低、pitch token acc 高 6pt），
   但自回归 Tier-1 生成质量并未更好（composite 略低 0.99、重复率更高、多样性更低）。
2. 任务A 判为"基本持平、BoYaTCN 略优"，优势主要来自 tempo_accuracy(+9.7)，
   配对 18/30 胜出 → 未达显著。
3. 自回归漂移确实存在，最清晰证据是重复率上升与逐位一致率下降；composite 对漂移不敏感。
4. 长序列下 MusicMamba 退化更显著（重复、掉多样性、中段 mode 崩塌）。
5. 本阶段为纯推理评估，无法修复训练侧词表长尾缺陷；仅 Tier-1 符号指标，未做 Tier-2 音频/听测。

### 7.10 E5 产物清单

**代码/配置**：`configs/e5_model_drift.yaml`（新增）、`generate_music.py`（E5 生成分支）、
`evaluate_e4_sampling.py`（`--stage e5` 分支，E4 路径逐行未改）、`train_boyatcn.py`（`--model` 切换骨干）、
`tests/test_e5_infer_consistency.py`（新增）

**权重**：`checkpoints/e5/musicmamba/L3_seed{2026,2027,2028}/best_valloss_model.pt`（补训）
+ 引用 `checkpoints/e3/L3_seed*/best_valloss_model.pt`（E3 只读）

**训练产物**：`results/e3/L3_seed*_train_musicmamba.log`、`L3_seed*_train_log_musicmamba.csv`、
`reports/e3/L3_seed*_loss_curve_musicmamba.png`、`results/e5/mm_*.out`

**生成/评估产物**：`results/e5/generated_midi/*.mid`(240)、`e5_generate_list.csv`(240 行)、
`e5_events.jsonl`(240 条)、`e5_eval_rows.csv`(240 行含 46 维特征)、`e5_eval_summary.csv`(8 组聚合)、
`e5_eval_summary.json`（含 GT 天花板 + 降维簇中心距离）、`e5_infer_consistency_test.json`(28/28)

**图表**：`reports/e5/e5_drift_lines.png`、`reports/e5/e5_umap_46d.png`、`reports/e5/e5_tsne_46d.png`

**报告**：`reports/E5_model_compare_drift_summary.md`

---

## 8. E5 补充校验：整曲粒度 46 维特征降维可视化（分支 feature/e5-fullsong-vis_check）

**分支用途**：仅新增后处理校验可视化。无训练、无微调、无重新推理生成；
不修改任何既有业务脚本（generate_music.py / evaluate_e4_sampling.py / visualize_embedding.py 均保持原样）；
不改动 E5 主实验图 e5_umap_46d.png / e5_tsne_46d.png；E5 原有全部实验结论保持不变。
**不合并回 experiment/E5_model_compare_drift 主线。**

### 8.1 数据与口径（全部只读）

- 生成样本：只读 `results/e5/e5_events.jsonl`，仅取 target_len=256 记录（60 条 =
  2 模型 × 3 种子 × 10 验证窗口），丢弃前 16 个 prompt 事件，对剩余 240 个续写事件
  做全局聚合统计（1 条记录 = 图上 1 点）。不使用 32/64/128（短序列统计噪声过高）。
- GT 真值（完整乐曲，1 首 = 1 条 46 维特征 = 图上 1 点）：
  南音训练集 9 首（output/BasicPitch/dapu01,02,03,04,05,06,10,12,13）、
  南音验证集 2 首（dapu14, dapu15）、外部非南音 2 首（`古典曲/` 目录 MIDI）。
  南音走 `dataset_nanyin.encode_song_events`，外部走 pretty_midi 解析；
  两者均经 `CompoundEventCodec.encode_notes` 编码为复合事件序列，特征空间与生成样本一致。
- 46 维特征全部复用 `symbolic_features.extract_symbolic_features`；
  运行时校验 pitch-class 直方图分量求和 = 1.000000（概率归一化，消除事件总数带来的计数偏差）。
- 调式分量：曲目级 `estimate_mode_from_pitch → resolve_mode_pcs`（GT 与生成样本同口径）。

### 8.2 降维规则（fit 仅用 GT）

- StandardScaler / UMAP / t-SNE **仅用 13 条完整乐曲 GT 特征向量 fit**；
  生成样本 60 条用 transform 投射进已拟合空间，禁止全体混在一起重新 fit。
- 超参严格沿用 configs/e5_model_drift.yaml：UMAP(n_neighbors=8, min_dist=0.1, seed=42)、
  t-SNE(perplexity=5, max_iter=2000, init=pca, seed=42)。
- 实现说明：sklearn TSNE 不提供 out-of-sample transform，t-SNE 侧改用 openTSNE
  （TSNEEmbedding.transform，openTSNE 1.0.4，本分支新装依赖）；UMAP 侧为 umap-learn 原生 transform。
  ⚠ GT 仅 13 个点 < perplexity=5×3 邻居需求，openTSNE 自动将有效 perplexity 降至 4.0（有运行时警告）。

### 8.3 客观数值（原始 46 维空间 · 到南音 GT 质心欧氏距离；仅记录，不做主观解读）

| 类别 | mean | min | max | n |
|---|---|---|---|---|
| GT 南音训练集 | 0.2363 | 0.1816 | 0.3657 | 9 |
| GT 南音验证集 | 0.3270 | 0.2869 | 0.3670 | 2 |
| GT 外部非南音 | 1.1134 | 0.9452 | 1.2816 | 2 |
| BoYaTCN 续写(240事件) | 0.4204 | 0.2586 | 0.8046 | 30 |
| MusicMamba 续写(240事件) | 0.4221 | 0.2164 | 0.6153 | 30 |

### 8.4 输出图（视觉编码：颜色 = 数据类别/模型；形状 = GT(圆) vs BoYaTCN(三角) vs MusicMamba(方)）

- `reports/e5/e5_fullsong_umap_check.png`
- `reports/e5/e5_fullsong_tsne_check.png`

**图注（客观局限性，必须随图引用）**：生成样本为"单 prompt 引出的 240 事件续写片段"全局特征，
并非真正完整原曲；GT 为完整乐曲全局特征，二者原始序列长度不一致；GT 仅 13 个 fit 点，
降维几何不稳定；本图仅作定性辅助校验，**不能替代 E5 窗口粒度定量评估**。
本分支仅做后处理可视化，E5 原有实验结论保持不变。图像解读交由实验人员根据实际结果完成，
脚本不预设"GT 聚集 / 生成靠近 GT"等任何预期。

### 8.5 本分支产物清单

- 新增脚本：`visualize_fullsong_reduction.py`（唯一新增代码文件，可独立运行：
  `venv\Scripts\python.exe visualize_fullsong_reduction.py`）
- 新增图片：`reports/e5/e5_fullsong_umap_check.png`、`reports/e5/e5_fullsong_tsne_check.png`
- 新装依赖：openTSNE 1.0.4（仅本分支 t-SNE transform 需要，不影响 E5 主线复现）

### 8.6 版本更新：GT 改为 256 事件非重叠滑窗 + 多 target_len 循环绘图（同分支二次提交）

**改动内容**（仅修改本分支新增脚本 visualize_fullsong_reduction.py，其他仓库文件零改动）：

1. GT 样本：取消"整曲单点"策略，更换为 **256 事件非重叠滑窗**（window_size=256、
   stride=256，与 E5 主实验 / 训练阶段滑窗规则一致，不足 256 的尾段直接丢弃）。
   南音训练集、验证集、外部非南音曲目**全部**滑窗，每个窗口独立计算 46 维特征，
   **1 个滑窗窗口 = 1 个 GT 点**。得 GT 点 152 个：
   南音训练集 128 窗 / 南音验证集 10 窗（与 E5 主实验 10 个验证窗口口径吻合）/
   外部非南音 14 窗。
2. 生成样本聚合逻辑**保持不变**：丢弃前 16 个 prompt 事件，整条续写全部事件聚合为
   一组 46 维特征；一次续写只输出 1 个点，续写内不分段。
3. 降维拟合源更新：StandardScaler / UMAP / t-SNE 在**全部 152 个滑窗 GT 样本上 fit**；
   生成样本 transform 投射，不参与拟合。超参维持 E5 配置不变（UMAP n_neighbors=8,
   min_dist=0.1, seed=42；t-SNE perplexity=5, max_iter=2000, init=pca, seed=42，
   仍经 openTSNE 实现 out-of-sample transform）。
4. 多长度循环绘图：target_len ∈ [32, 64, 128, 256]，每个长度独立一对 UMAP / t-SNE 图，
   四个长度**复用同一套**已 fit 好的 scaler 与降维模型。
   图片：`reports/e5/e5_sliding_umap_len{32,64,128,256}_check.png`、
   `reports/e5/e5_sliding_tsne_len{32,64,128,256}_check.png`（共 8 张）。
   上一版"整曲单点"口径的 e5_fullsong_*_check.png 两张图已从本分支删除（被本版取代）。

**客观数值**（原始 46 维空间 · 到南音滑窗 GT 质心欧氏距离均值；仅记录，不做主观解读）：

| target_len（续写事件数） | BoYaTCN mean | MusicMamba mean |
|---|---|---|
| 32（16 事件） | 0.8410 | 0.8351 |
| 64（48 事件） | 0.6143 | 0.5947 |
| 128（112 事件） | 0.5255 | 0.5922 |
| 256（240 事件） | 0.4185 | 0.3999 |

**口径说明与局限性（随图引用）**：GT = 256 事件非重叠滑窗窗口全局统计；
生成样本 = 单 prompt 引出的一条续写整体聚合统计（续写内不分段，1 条 = 1 点）。
各 target_len 下续写事件数（16/48/112/240）与 GT 窗口 256 事件不等长，
且短长度统计噪声高；本图仅供定性辅助校验，不能替代 E5 窗口粒度定量评估。
仍为仅后处理：无训练、无重新推理，jsonl 只读；不合并回 E5 主线。
图像解读交由实验人员根据实际结果完成，脚本不预设任何预期结果。
