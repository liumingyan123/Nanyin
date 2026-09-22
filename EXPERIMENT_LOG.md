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
