# Nanyin
作图，模型
## 对 BoYaTCN 和 MusicMamba 改了什么
改造前快照在 backup_pre_causal/，对比后可归为 3 个文件、2 类改动。

### 改动 1：models.py（模型结构因果化）
共同/共享层（OctaveConv1d，被 BoYaTCN 使用）

卷积 padding 由 1 改为 0，改为前向左侧填充 F.pad(x, (kernel-1, 0))，保证位置 i 只看到 ≤i
低频下采样 avg_pool1d(kernel=2) → 取偶数位置 [..., 0::2]（原来池化窗口 [2j, 2j+1] 会混入未来 token x[2j+1]）
上采样保留 F.interpolate(mode='nearest')（位置 i 取 floor(i/2)，天然因果）
NanyinBoYaTCN.forward（BoYaTCN 主体）

自注意力加因果掩码：torch.triu(full(-inf), diagonal=1) 传给 MultiheadAttention
BiLSTM → 单向 LSTM（bidirectional=False，隐层 d_model//2 → d_model 保持输出维度）
低频分支下采样同样由 avg_pool1d 改为 0::2
SelectiveSSM（Mamba 的核心层，两个模型共享）

训练时改用 梯度检查点 torch.utils.checkpoint.checkpoint(selective_scan_parallel, ...)：显存从 >6GB 降到可训（RTX 3060 Laptop），结构与训练目标不变
NanyinMusicMamba本身

结构无需改：选择性扫描 h_t = exp(ΔA)h_{t-1} + ΔB·x_t 是逐时间步递推，天然严格因果
只受益于上面的 SSM 梯度检查点（显存优化）
 ### 改动 2：train_boyatcn.py 和 train_musicmamba.py（训练目标对齐）
两个脚本改动一致：

新增 shift_seq_targets()：序列级 target（tokens/pitch/duration/ornament）右移一位 [:, 1:]
前向改为 model(input_tokens[:, :-1])，使 pred[i] 对应真值 x[i+1]，变成真正的"用前缀预测下一个 token"
EARLY_STOP_PATIENCE 由 60 → 160（验证集 NOEF 很早饱和，耐心太小会提前截断训练导致欠拟合）
