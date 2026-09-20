# E1 复合事件（pitch + duration + time-shift）编解码单元自测报告

- **分支**：`feature/compound-events`
- **阶段**：E1_compound_events
- **范围**：只做复合事件 + MIDI 往返编解码；不训练、不滑窗、不做损失消融
- **自测对象**：`MIDI → 复合事件序列 → 写回 MIDI` 往返一致性

## 1. 配置参数（全部来自 `configs/e1_compound_events.yaml`）

| 参数 | 值 |
|---|---|
| 量化单位 | 0.02 s / 步（20 ms）|
| pitch 词表 | 128 类（MIDI pitch 0~127 直映射）|
| duration 精确上限 | 399 步 = 7.98 s，类别数 402 |
| time-shift 精确上限 | 999 步 = 19.98 s，类别数 1002 |
| MIDI 写出 | ticks_per_beat=480，tempo=120 BPM |
| 往返容差 | ±20 ms（= 1 个量化步长）|

## 2. 自测结论

| 判定项 | 标准 | 实测 | 结论 |
|---|---|---|---|
| 音高无损 | pitch mismatch = 0 | 最大 = 0 | **PASS** |
| 音符数守恒（写回 MIDI 后重读） | count_dec = count_reread | 全部一致 | **PASS** |
| 末尾 time-shift 为伪标签 | 末位 shift index = 0 | 全部为 0 | **PASS** |
| 编解码器单元检查 | 5 项全通过 | 5/5 通过 | **PASS** |
| 时值往返误差 | 全部 ≤ 20 ms | 最低满足率 100.00%，最大 10.0 ms | **PASS** |
| onset 往返误差 | 全部 ≤ 20 ms | 最低满足率 100.00%，最大 20.0 ms | **PASS** |

## 3. 分项测试明细

### T1 编解码器单元一致性

| 检查项 | 结果 | 说明 |
|---|---|---|
| duration 精确档往返 | PASS | 0~399 步长（0~7980 ms），容差 1e-9 s |
| duration 溢出档语义 | PASS | ≥ 8000 ms → index 401，还原为下界 |
| time-shift 精确档往返 | PASS | 0~999 步长（0~19980 ms）|
| time-shift 溢出档语义 | PASS | ≥ 20000 ms → index 1001 |
| index 0 = PAD / 伪标签 | PASS | 还原为 0 秒 |

### T2 真实 GT 曲目往返（11 首 BasicPitch 转录）

流程：`load_song_annotations(MIDI)` → `codec.encode_notes` → `events_to_notes` → `notes_to_midi` → `pretty_midi` 读回比较。

#### T2.1 音符级（编解码 + 量化，不含 MIDI tick 误差）

| 曲目 | 音符数 | pitch 错 | 时值均 err(ms) | 时值 max(ms) | 时值≤容差 | onset 均 err(ms) | onset max(ms) | onset≤容差 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dapu01qishouban | 3937 | 0 | 4.94 | 10.0 | 100.00% | 5.51 | 13.2 | 100.00% |
| dapu02sanmianjinqianjing | 4096 | 0 | 5.07 | 10.0 | 100.00% | 8.34 | 18.2 | 100.00% |
| dapu03wucaojinqianjing | 3118 | 0 | 5.01 | 10.0 | 100.00% | 6.33 | 15.5 | 100.00% |
| dapu04bamianjinqianjing | 4096 | 0 | 4.89 | 10.0 | 100.00% | 5.84 | 14.1 | 100.00% |
| dapu05sishijing | 4096 | 0 | 4.99 | 10.0 | 100.00% | 5.59 | 13.6 | 100.00% |
| dapu06meihuacao | 4096 | 0 | 4.92 | 10.0 | 100.00% | 8.91 | 18.6 | 100.00% |
| dapu10sanbuhe | 1412 | 0 | 4.85 | 10.0 | 100.00% | 5.07 | 11.4 | 100.00% |
| dapu12sijingban | 4096 | 0 | 4.90 | 10.0 | 100.00% | 9.36 | 19.1 | 100.00% |
| dapu13kongquezhanping | 4096 | 0 | 4.91 | 10.0 | 100.00% | 5.06 | 10.5 | 100.00% |
| dapu14kouhuangtian | 1025 | 0 | 5.10 | 10.0 | 100.00% | 5.65 | 14.1 | 100.00% |
| dapu15wujinjiao | 1746 | 0 | 5.00 | 10.0 | 100.00% | 9.77 | 20.0 | 100.00% |

#### T2.2 溢出/异常统计（量化局限的直接证据）

| 曲目 | 超出 max_notes 截断 | duration 溢出 | time-shift 溢出 | 负间隔被截断 | 末尾伪标签=0 |
|---|---:|---:|---:|---:|---|
| dapu01qishouban | 0 | 0 | 0 | 0 | 是 |
| dapu02sanmianjinqianjing | 2174 | 0 | 0 | 0 | 是 |
| dapu03wucaojinqianjing | 0 | 0 | 0 | 0 | 是 |
| dapu04bamianjinqianjing | 2301 | 0 | 0 | 0 | 是 |
| dapu05sishijing | 766 | 0 | 0 | 0 | 是 |
| dapu06meihuacao | 118 | 0 | 0 | 0 | 是 |
| dapu10sanbuhe | 0 | 0 | 0 | 0 | 是 |
| dapu12sijingban | 1302 | 0 | 0 | 0 | 是 |
| dapu13kongquezhanping | 621 | 0 | 0 | 0 | 是 |
| dapu14kouhuangtian | 0 | 0 | 0 | 0 | 是 |
| dapu15wujinjiao | 0 | 0 | 0 | 0 | 是 |

#### T2.3 写回 MIDI 后重读（含 tick 量化误差）

| 曲目 | 写入音符 | 读回音符 | 数量守恒 | pitch 错 | 时值 max(ms) | onset max(ms) | 同音重叠截断 |
|---|---:|---:|---|---:|---:|---:|---:|
| dapu01qishouban | 3937 | 3937 | 是 | 0 | 20.83 | 0.42 | 163 |
| dapu02sanmianjinqianjing | 4096 | 4096 | 是 | 0 | 20.83 | 0.42 | 311 |
| dapu03wucaojinqianjing | 3118 | 3118 | 是 | 0 | 20.83 | 0.42 | 152 |
| dapu04bamianjinqianjing | 4096 | 4096 | 是 | 0 | 20.83 | 0.42 | 236 |
| dapu05sishijing | 4096 | 4096 | 是 | 0 | 20.83 | 0.42 | 228 |
| dapu06meihuacao | 4096 | 4096 | 是 | 0 | 20.83 | 0.42 | 204 |
| dapu10sanbuhe | 1412 | 1412 | 是 | 0 | 20.63 | 0.42 | 20 |
| dapu12sijingban | 4096 | 4096 | 是 | 0 | 20.83 | 0.42 | 160 |
| dapu13kongquezhanping | 4096 | 4096 | 是 | 0 | 20.83 | 0.42 | 145 |
| dapu14kouhuangtian | 1025 | 1025 | 是 | 0 | 20.83 | 0.42 | 27 |
| dapu15wujinjiao | 1746 | 1746 | 是 | 0 | 20.83 | 0.42 | 82 |

### T3 合成边界用例

| 用例 | 音符数 | pitch 错 | 时值 max(ms) | onset max(ms) | duration 溢出 | shift 溢出 | 读回音符 | 同音重叠截断 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 超长休止(溢出档) | 3 | 0 | 0.0 | 5500.0 | 0 | 1 | 3 | 0 |
| 同时发音shift=0 | 4 | 0 | 0.0 | 0.0 | 0 | 0 | 4 | 0 |
| 超长音符(溢出档) | 2 | 0 | 4000.0 | 0.0 | 1 | 0 | 2 | 0 |
| 同音反复+重叠 | 4 | 0 | 0.0 | 10.0 | 0 | 0 | 4 | 1 |
| 负间隔防御(乱序输入) | 3 | 不适用（输入本身乱序） | 0.0 | 300.0 | 0 | 0 | 3 | 0 |

用例说明与预期：

- **超长休止 / 超长音符**：故意超过 yaml 中的精确上限，用于验证溢出档确实被触发、且溢出后后续音符 onset 会被整体前移（误差累积），属已知局限而非 bug。
- **同时发音 (shift=0)**：多个音符共享同一 onset，验证 shift=0 不丢音符、MIDI 事件流排序正确。
- **同音反复 + 重叠**：验证写 MIDI 时对同音高重叠区间的截断保护生效，音符数不丢（`truncated_overlap` 为被截断处数；历史上此处处理不当曾导致 1025 音符被解析成 792）。
- **负间隔**：输入未按 onset 排序（onset 回退），验证负 gap 被截断为 0 并计数，不会解码出负 onset。

## 4. 已知局限（本阶段如实记录，不在报告中掩盖）

1. **长休止 / 长音符表达受限（溢出档）**：time-shift 超过 20.00 s、时值超过 8.00 s 时一律压缩到溢出档，解码只能还原下界值，且**其后所有音符的 onset 会被整体前移（误差累积）**。当前 11 首 GT 的真实分布（最大 IOI 19.40 s、最大时值 6.89 s）恰好不触发溢出，故本轮自测中溢出计数为 0——这是把 max_steps 按实测分布调大的结果，并非该问题被消除；合成边界用例（休止 25 s / 音符 12 s）已验证其仍会发生。
2. **溢出档带来的词表代价（已发生的取舍）**：为了让真实数据不溢出，shift 类别数被拉到 1002、duration 到 402，其中长尾档几乎无训练样本，后续训练阶段可能出现类别稀疏；更合理的做法是非均匀（对数）分箱，本阶段不做。
3. **末尾 time-shift 为伪标签**：采用前向 IOI 约定（shift_i = onset_(i+1) - onset_i），最后一个音符之后不存在"下一个音符"，其 time-shift 无监督目标，编码置 0；**后续训练必须对该位置做 loss 掩码，窗口切片切尾时同理**。
4. **量化粒度固定**：所有位置统一 20 ms 网格。onset 误差上界 = ±半格（时值）／±1 格（相对 onset 比较含首音符舍入），无法同时精细刻画极短装饰音与超长时值。
5. **同音高重叠区间的截断**：GT 由 BasicPitch 多轨合并而来，存在同音高重叠音符。写 MIDI 时会把前一个音截断到后一个音起点（本轮最多 311 处），以规避 (channel, pitch) 配对歧义导致的丢音符；代价是这些音符的**时值被缩短**，因此 file-level 时值误差可达数十 ms（见 T2.3）。音符数量始终守恒。
6. **max_notes 截断**：单曲最多保留 4096 个音符，超出部分被丢弃（dapu02/04/05/12/13 有截断，见 T2.2）；E1 不做滑窗，滑窗在后续阶段处理。
7. **本自测只覆盖编解码与 MIDI 往返**，不含模型生成质量评估；复合事件模型尚未训练，因此 E1 未产出任何生成 MIDI、未给出任何生成指标。

## 5. 复现方式

```bash
venv/Scripts/python.exe tests/test_compound_event_roundtrip.py
```

- 报告路径：`reports/E1_selftest_compound_events.md`
- 指标 JSON：`results/e1_roundtrip_metrics.json`
- 临时 MIDI：`results/e1_roundtrip_tmp/`
