import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# =================================================================
# 通用工具：正弦位置编码
# 文献来源：Attention Is All You Need (Vaswani et al., 2017)
# =================================================================

class SinusoidalPositionalEncoding(nn.Module):
    """
    正弦位置编码，为输入token序列添加时序位置信息。

    文献来源：
        MelodyGLM (https://arxiv.org/pdf/2309.10738v1)
        该文在旋律生成Transformer中使用正弦位置编码。
    """
    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.shape[1], :].to(x.device)
        return self.dropout(x)


# =================================================================
# 专业版 MusicMamba 架构 (基于 Selective State Space Model)
# 真实 Mamba：选择性扫描 (Selective Scan) 忠实实现
# 文献：Mamba: Linear-Time Sequence Modeling with Selective State Spaces
# (Gu & Dao, 2023, arXiv:2312.00752)，Algorithm 1/2
# =================================================================


def selective_scan_ref(u, delta, A, B, C, D):
    """Mamba 选择性扫描 (Algorithm 1/2) 的纯 PyTorch 参考实现。

    核心递归（ZOH 零阶保持离散化后的状态累积）：
        h_t = exp(delta_t * A) * h_{t-1} + delta_t * B_t * u_t
        y_t = C_t @ h_t + D * u_t

    参数:
        u:     (B, L, D) 输入序列（SSM 内部特征）
        delta: (B, L, D) 步长（选择性参数，softplus 后非负）
        A:     (D, N)    状态转移矩阵（负对角，固定指数衰减）
        B, C:  (B, L, N) 选择性输入/输出投影
        D:     (D,)      跳跃连接（skip）
    返回:
        y:     (B, L, D) 扫描输出

    说明:
        与 mamba_ssm 官方 CUDA 内核 (SelectiveScanFn) 算法一致；
        此处用逐时间步张量运算实现，无需 CUDA 编译，可直接训练。
    """
    Bsz, L, Dd = u.shape
    N = A.shape[1]
    # ZOH 离散化：dA = exp(delta*A)，dB = delta*B
    dA = torch.exp(delta.unsqueeze(-1) * A)        # (B, L, D, N)
    dB = delta.unsqueeze(-1) * B.unsqueeze(2)      # (B, L, D, N)
    h = torch.zeros(Bsz, Dd, N, device=u.device, dtype=u.dtype)
    ys = []
    for t in range(L):
        h = dA[:, t] * h + dB[:, t] * u[:, t].unsqueeze(-1)   # 状态累积
        ys.append((h * C[:, t].unsqueeze(1)).sum(-1) + D * u[:, t])
    return torch.stack(ys, dim=1)                  # (B, L, D)


def selective_scan_parallel(u, delta, A, B, C, D):
    """Mamba 选择性扫描的 GPU 并行实现（Hillis–Steele 前缀扫描）。

    与 selective_scan_ref 数值一致：组合运算精确模拟同一递推
        h_t = dA_t * h_{t-1} + dB_t * u_t   （dA = exp(delta*A) ∈ (0,1)）
    但用并行前缀扫描替代逐时间步 Python 循环，训练提速数十倍。
    （Mamba 官方 CUDA 内核同样采用并行 scan，而非逐步串行。）

    递推视角：T_t(h) = a_t h + b_t，组合 T_j ∘ T_i 为
        (a_j * a_i,  a_j * b_i + b_j)
    Hillis–Steele 扫描 O(log L) 轮、每轮全并行，得到各时刻的组合映射；
    因初始状态 h_0 = 0，y_t = C_t @ b_t + D * u_t。
    """
    Bsz, L, Dd = u.shape
    N = A.shape[1]
    # ZOH 离散化：dA = exp(delta*A)，dB = delta*B
    a = torch.exp(delta.unsqueeze(-1) * A)          # (B, L, D, N)  递推系数 a_t
    b = delta.unsqueeze(-1) * B.unsqueeze(2)        # (B, L, D, N)  dB
    b = b * u.unsqueeze(-1)                          # (B, L, D, N)  递推输入 b_t

    # Hillis–Steele 并行前缀扫描（inclusive scan, 1-based 组合）
    n_steps = L.bit_length()
    for k in range(n_steps):
        offset = 1 << k
        if offset >= L:
            break
        # 右移 offset，头部补恒等映射 (a=1, b=0)，使 t<offset 的位置保持不变
        a_shift = torch.cat([torch.ones_like(a[:, :offset]), a[:, :-offset]], dim=1)
        b_shift = torch.cat([torch.zeros_like(b[:, :offset]), b[:, :-offset]], dim=1)
        # 组合：(a[t]*a[t-o], a[t]*b[t-o] + b[t])，注意用更新前的 a 计算 b
        b = a * b_shift + b
        a = a * a_shift

    # 初始状态 h_0 = 0，故 h_t = b_t；y_t = C_t @ h_t + D * u_t
    y = (b * C.unsqueeze(2)).sum(-1) + D * u        # (B, L, D)
    return y


class SelectiveSSM(nn.Module):
    """
    真实选择性状态空间模型 (Mamba) 核心单元。

    完整 Mamba 块结构（对齐 arXiv:2312.00752）：
      in_proj(x) -> x, z
      x -> SiLU -> causal depthwise conv1d -> x_proj(Δ,B,C) -> dt_proj(softplus)
      -> selective_scan(x, Δ, A, B, C) -> y
      y = (y * SiLU(z)) -> out_proj

    与旧版（简化门控）的根本区别：
      - 旧版：y = x * delta（把 Δ 当注意力权重，无状态空间）
      - 新版：真实选择性扫描，A/B/C/Δ 逐元素状态累积（论文 Algorithm 1/2），
              状态维度 N 上的历史信息被真实保留与选择性更新
    """
    def __init__(self, d_model, d_state=16, expand=2, conv_kernel=3, scan_mode='parallel'):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand
        self.conv_kernel = conv_kernel
        # 扫描实现：'parallel' = Hillis–Steele 并行前缀扫描（训练默认，快数十倍）
        #           'ref'      = 逐时间步参考实现（用于数值一致性验证）
        self.scan_mode = scan_mode

        # 输入投影：x 与 z（残差门控分支）
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # 因果深度卷积（Mamba 标配，增强局部上下文，groups=d_inner 为 depthwise）
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=conv_kernel,
                                groups=self.d_inner, bias=False)

        # 选择性参数投影：Δ(1) + B(d_state) + C(d_state)
        self.x_proj = nn.Linear(self.d_inner, self.d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(self.d_state * 2 + 1, self.d_inner, bias=True)

        # 状态矩阵 A（S4 初始化：正整数 -> 取负指数衰减），与论文一致
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))

        # 跳跃连接 D（论文中 y_t = C@h_t + D*x_t）
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # 输出投影
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.activation = nn.SiLU()

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        # 1. 输入投影 -> x 与 z
        x_and_res = self.in_proj(x)
        x, z = x_and_res.chunk(2, dim=-1)
        x = self.activation(x)

        # 2. 因果深度卷积（左填充保持因果性，输出长度不变）
        x = x.transpose(1, 2)                                  # (B, D, L)
        x = F.pad(x, (self.conv_kernel - 1, 0))                # 左侧 pad k-1
        x = self.conv1d(x)
        x = x.transpose(1, 2)                                  # (B, L, D)

        # 3. 选择性参数 Δ, B, C
        # Mamba 原版：dt_proj 输入完整 x_dbl(2N+1)，B/C 直接 split
        x_dbl = self.x_proj(x)                                 # (B, L, 2N+1)
        delta, B_, C_ = x_dbl.split([1, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(x_dbl))                # (B, L, D) 非负步长

        # 4. 真实选择性扫描（A 取负指数，作为衰减状态转移）
        A = -torch.exp(self.A_log)                             # (D, N)
        if self.scan_mode == 'ref':
            y = selective_scan_ref(x, delta, A, B_, C_, self.D)
        elif self.training:
            # 训练时用梯度检查点：selective_scan_parallel 的并行前缀扫描在 backward
            # 需缓存每轮循环的多个 (B,L,D,N) 中间张量，L≈4096 时显存峰值超过 6GB
            # （RTX 3060 Laptop）。梯度检查点以重算换显存，模型结构与训练目标不变。
            y = torch.utils.checkpoint.checkpoint(
                selective_scan_parallel, x, delta, A, B_, C_, self.D,
                use_reentrant=False,
            )                                                  # (B, L, D)
        else:
            y = selective_scan_parallel(x, delta, A, B_, C_, self.D)  # (B, L, D)

        # 5. 残差门控 + 输出
        y = y * self.activation(z)
        return self.out_proj(y)

class MusicMambaProfessional(nn.Module):
    """
    专业级 MusicMamba 分类器
    结合了 Mamba 的长序列建模能力
    """
    def __init__(self, input_dim=128, hidden_dim=256, num_classes=3, n_layers=2):
        super().__init__()
        # 音频特征投影 (Mel -> Hidden)
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        # 堆叠 Mamba 层
        self.layers = nn.ModuleList([
            SelectiveSSM(d_model=hidden_dim) for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch, n_mels, time) -> 转置为 (batch, time, n_mels)
        x = x.transpose(1, 2)
        
        x = self.embedding(x)
        
        # 依次通过 Mamba 层
        for layer in self.layers:
            x = x + layer(x) # 残差连接
            
        x = self.norm(x)
        
        # 全局池化 (取时间轴平均)
        x = x.mean(dim=1)
        
        return self.classifier(x)


# =================================================================
# 专业版 BoYaTCN 架构 (基于 Octave Conv + Attention)
# 模仿 BoYaTCN ，专门针对中国传统音乐设计
# =================================================================

class OctaveConv1d(nn.Module):
    """
    八度卷积 (Octave Convolution) —— 因果版本
    将特征分为高频和低频分量进行处理，适合捕捉音乐中的基频与泛音。
    因果性保证：所有卷积改用左侧填充（left pad），下采样只取偶数位置，
    使位置 i 的输出仅依赖输入位置 <= i，杜绝未来信息泄漏——
    这是"用前缀预测下一个 token"的自回归训练目标所必须的。
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, alpha_in=0.5, alpha_out=0.5, stride=1, padding=0):
        super().__init__()
        self.alpha_in = alpha_in
        self.alpha_out = alpha_out
        self.kernel_size = kernel_size
        
        # 计算高低频通道数
        in_h = int(in_channels * (1 - alpha_in))
        in_l = in_channels - in_h
        out_h = int(out_channels * (1 - alpha_out))
        out_l = out_channels - out_h
        
        # 全部 padding=0，因果性由 forward 中的左侧填充保证
        # 高频 -> 高频
        self.h2h = nn.Conv1d(in_h, out_h, kernel_size, stride, padding=0)
        # 高频 -> 低频 (下采样)
        self.h2l = nn.Conv1d(in_h, out_l, kernel_size, stride, padding=0)
        # 低频 -> 高频 (上采样)
        self.l2h = nn.Conv1d(in_l, out_h, kernel_size, stride, padding=0)
        # 低频 -> 低频
        self.l2l = nn.Conv1d(in_l, out_l, kernel_size, stride, padding=0)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.downsample = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, x_h, x_l):
        # 因果填充：左侧补 (kernel_size-1) 个零，卷积后输出长度不变
        pad = self.kernel_size - 1
        x_h_p = F.pad(x_h, (pad, 0))
        x_l_p = F.pad(x_l, (pad, 0))

        # 高频 -> 高频（长度 L）
        h2h = self.h2h(x_h_p)
        # 低频 -> 高频：nearest 上采样，位置 i 取低频 floor(i/2)，只含原位置 <=i 的信息
        l2h = F.interpolate(self.l2h(x_l_p), size=h2h.size(-1), mode='nearest')
        out_h = h2h + l2h

        # 低频 -> 低频（长度 L_l）
        l2l = self.l2l(x_l_p)
        # 高频 -> 低频：因果下采样（取偶数位置 0,2,4,...，位置 j 只含原位置 2j 及之前），
        # 取代原来的 interpolate 下采样/avg_pool，避免混入未来位置的信息
        h2l = self.h2l(x_h_p)[:, :, 0::2]
        if h2l.size(-1) != l2l.size(-1):
            h2l = F.interpolate(h2l, size=l2l.size(-1), mode='nearest')
        out_l = l2l + h2l

        return out_h, out_l

class BoYaTCNProfessional(nn.Module):
    """
    专业版 BoYaTCN
    融合了 Octave Conv, TCN 和 Self-Attention
    """
    def __init__(self, input_dim=128, hidden_dim=256, num_classes=3):
        super().__init__()
        # 初始投影层，将 Mel 频谱分为高低频
        self.in_h = int(hidden_dim * 0.5)
        self.in_l = hidden_dim - self.in_h
        
        self.first_conv = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        
        # Octave TCN 块
        self.octave_conv = OctaveConv1d(hidden_dim, hidden_dim, kernel_size=3)
        
        # Attention 层
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, batch_first=True)
        
        # BiLSTM 层，增强时序建模
        self.bilstm = nn.LSTM(hidden_dim, hidden_dim // 2, num_layers=1, bidirectional=True, batch_first=True)
        
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        # x shape: (batch, n_mels, time)
        x = self.first_conv(x)
        
        # 模拟 Octave 分解
        x_h = x[:, :self.in_h, :]
        x_l = F.avg_pool1d(x[:, self.in_h:, :], kernel_size=2, stride=2)
        
        # 八度卷积
        x_h, x_l = self.octave_conv(x_h, x_l)
        
        # 合并高低频 (低频上采样)
        x_l_up = F.interpolate(x_l, size=x_h.size(-1), mode='nearest')
        x = torch.cat([x_h, x_l_up], dim=1)
        
        # Attention (需要转置为 batch, seq, dim)
        x = x.transpose(1, 2)
        attn_out, _ = self.attention(x, x, x)
        x = x + attn_out
        
        # BiLSTM
        x, _ = self.bilstm(x)
        
        # 全局池化并分类
        x = x.mean(dim=1)
        return self.classifier(x)


# =================================================================
# TCN（时间卷积网络）：BoYaTCN 论文核心组件之一
# 文献：An Empirical Evaluation of Generic Convolutional and Recurrent
# Networks for Sequence Modeling (Bai et al., 2018)；
# BoYaTCN (MDPI Applied Sciences 2022) 在 BiLSTM 后叠加膨胀因果卷积
# =================================================================

class TemporalBlock(nn.Module):
    """TCN 残差块：两层膨胀因果卷积 + 残差连接。
    因果性通过左侧填充 (kernel-1)*dilation 并裁掉尾部实现。"""
    def __init__(self, n_channels, kernel_size=3, dilation=1, dropout=0.2):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(
            n_channels, n_channels, kernel_size,
            padding=self.padding, dilation=dilation))
        self.conv2 = nn.utils.weight_norm(nn.Conv1d(
            n_channels, n_channels, kernel_size,
            padding=self.padding, dilation=dilation))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, C, L)
        out = F.relu(self.conv1(x))
        if self.padding > 0:
            out = out[:, :, :-self.padding]      # 裁尾保因果
        out = self.dropout(out)
        out = F.relu(self.conv2(out))
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        out = self.dropout(out)
        return x + out                            # 残差


class TemporalConvNet(nn.Module):
    """膨胀率递增的 TCN 堆叠（感受野指数扩大）"""
    def __init__(self, n_channels, levels=(1, 2, 4, 8), kernel_size=3, dropout=0.2):
        super().__init__()
        self.blocks = nn.ModuleList([
            TemporalBlock(n_channels, kernel_size, d, dropout) for d in levels
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


# =================================================================
# 南音生成版 BoYaTCN：多输出头适配 NOEF 五维评估体系
# =================================================================

class NanyinBoYaTCN(nn.Module):
    """
    南音符号旋律生成模型，基于 BoYaTCN 架构扩展多输出头。

    核心架构（完整 BoYaTCN 论文结构）：
      Token Embedding → 位置编码 → OctaveConv1d → MultiheadAttention(因果掩码)
      → 单向LSTM → TCN(膨胀因果卷积) → [序列级输出头 + 全局级输出头]

      因果性说明：本模型所有层均为严格因果（位置 i 的输出仅依赖输入位置 <=i），
      配合训练脚本的 shift 对齐（输入 x[:-1]，target x[1:]），实现真正的
      "用前缀预测下一个 token" 自回归训练目标。

    输出头设计（完全对应 loss_nanyin.py 中 nanyin_total_loss() 入参）：
      - 序列级 (per-timestep): pred_tokens, pred_pitch, pred_duration, pred_ornament
      - 全局级 (pooled):      pred_tempo_dist, pred_mode_logits, pred_features, pred_style

    文献来源：
      MelodyGLM         https://arxiv.org/pdf/2309.10738v1
      Controllable Symbolic Music Generation https://www.preprints.org/manuscript/202604.0984
      TCSinger           https://arxiv.org/pdf/2409.15977
      NanyinHGNN         arXiv:2510.03617
      KAD                arXiv:2502.15602
    """

    def __init__(
        self,
        vocab_size: int = 128,
        d_model: int = 256,
        num_heads: int = 8,
        num_tempo_bins: int = 20,
        num_modes: int = 4,
        feat_dim: int = 13,
        style_dim: int = 32,
        max_seq_len: int = 4096,
        dropout: float = 0.2,
    ):
        """
        参数:
            vocab_size:      词汇表大小（MIDI token范围0~127 + 特殊token）
            d_model:         隐藏维度
            num_heads:       注意力头数
            num_tempo_bins:  速度分布分箱数（与 loss_nanyin.py 对齐）
            num_modes:       调式类别数（南音4种专属调式）
            feat_dim:        MFCC特征维度
            style_dim:       风格嵌入维度
            max_seq_len:     最大序列长度（默认8192，覆盖最长南音曲目~4643音符）
            dropout:         Dropout概率
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        # ---- Token嵌入 + 位置编码 ----
        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)

        # ---- OctaveConv 高低频分离参数 ----
        self.in_h = int(d_model * 0.5)
        self.in_l = d_model - self.in_h
        self.octave_conv = OctaveConv1d(d_model, d_model, kernel_size=3)

        # ---- 自注意力层 ----
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, batch_first=True, dropout=dropout
        )

        # ---- 单向 LSTM 时序建模（因果：位置 i 的输出只依赖 <=i 的输入）----
        # 原双向 LSTM 会让位置 i 的表示包含未来 token x[i+1]（即训练 target 本身），
        # 使训练退化为"完形填空/自复制"而非"预测下一个 token"，与自回归生成错位
        # （pitch_acc 低、生成卡死重复的根因）。改为单向保证严格因果。
        # 单层LSTM不使用dropout（nn.LSTM仅在num_layers>=2时应用dropout）
        self.lstm = nn.LSTM(
            d_model, d_model, num_layers=1,
            bidirectional=False, batch_first=True
        )

        # ---- TCN 膨胀因果卷积（BoYaTCN 论文核心组件，dilation 指数扩张） ----
        self.tcn = TemporalConvNet(d_model, levels=(1, 2, 4, 8), kernel_size=3, dropout=dropout)

        # ---- LayerNorm 稳定训练 ----
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # ---- Dropout ----
        self.dropout = nn.Dropout(dropout)

        # ========== 输出头 ==========

        # 序列级输出头（per-timestep）
        self.token_head = nn.Linear(d_model, vocab_size)        # token分类logits
        self.pitch_head = nn.Linear(d_model, 1)                  # 音高回归
        self.duration_head = nn.Linear(d_model, 1)              # 时值回归
        self.ornament_head = nn.Sequential(                      # 装饰音二分类
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )

        # 全局级输出头（序列均值池化后）
        self.tempo_head = nn.Linear(d_model, num_tempo_bins)    # 速度分布
        self.mode_head = nn.Linear(d_model, num_modes)           # 调式logits
        self.features_head = nn.Linear(d_model, feat_dim)        # MFCC特征
        self.style_head = nn.Linear(d_model, style_dim)          # 风格嵌入

    def forward(self, input_tokens: torch.Tensor) -> dict:
        """
        前向传播。

        参数:
            input_tokens: token ID序列, shape=(batch, seq_len), dtype=long
        返回:
            字典，包含所有预测输出：
              - "pred_tokens":    (batch, seq_len, vocab_size)
              - "pred_pitch":     (batch, seq_len)
              - "pred_duration":  (batch, seq_len)
              - "pred_ornament":  (batch, seq_len)
              - "pred_tempo_dist": (batch, num_tempo_bins)
              - "pred_mode_logits": (batch, num_modes)
              - "pred_features":   (batch, feat_dim)
              - "pred_style":      (batch, style_dim)
        """
        # ---- 1. Token Embedding ---- (batch, seq_len) → (batch, seq_len, d_model)
        x = self.token_embed(input_tokens)  # (B, L, d_model)
        x = x * math.sqrt(self.d_model)     # 缩放嵌入（Transformer标准做法）

        # ---- 2. 位置编码 ----
        x = self.pos_encoding(x)            # (B, L, d_model)

        # ---- 3. OctaveConv 处理 (Conv1d 模式) ----
        # 转置: (B, L, d_model) → (B, d_model, L) 适配Conv1d
        x_conv = x.transpose(1, 2)          # (B, d_model, L)
        x_h = x_conv[:, :self.in_h, :]
        # 因果下采样：取偶数位置（位置 j = 原位置 2j）。
        # avg_pool 窗口 [2j,2j+1] 会混入未来 token x[2j+1]，造成信息泄漏。
        x_l = x_conv[:, self.in_h:, 0::2]

        # 八度卷积
        x_h, x_l = self.octave_conv(x_h, x_l)

        # 合并高低频
        x_l_up = F.interpolate(x_l, size=x_h.size(-1), mode='nearest')
        x_conv = torch.cat([x_h, x_l_up], dim=1)  # (B, d_model, L)
        x = x_conv.transpose(1, 2)                # (B, L, d_model)
        x = self.norm1(x)

        # ---- 4. 自注意力（因果掩码：位置 i 只能 attend 到 <=i，杜绝未来泄漏）----
        seq_len = x.shape[1]
        attn_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=x.device, dtype=x.dtype),
            diagonal=1,
        )
        attn_out, _ = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(attn_out)            # 残差连接
        x = self.norm2(x)

        # ---- 5. 单向 LSTM（因果）----
        x, _ = self.lstm(x)                       # (B, L, d_model)

        # ---- 5.5 TCN 膨胀因果卷积（BoYaTCN 论文核心，扩张感受野） ----
        x = self.tcn(x.transpose(1, 2)).transpose(1, 2)   # (B, L, d_model)
        x = self.norm3(x)

        # ---- 6. 序列级输出 ----
        pred_tokens = self.token_head(x)                   # (B, L, vocab_size)
        pred_pitch = self.pitch_head(x).squeeze(-1)        # (B, L)
        pred_duration = F.softplus(self.duration_head(x).squeeze(-1))  # (B, L) 正值约束
        pred_ornament = self.ornament_head(x).squeeze(-1)  # (B, L) in [0,1]

        # ---- 7. 全局级输出（序列均值池化）----
        x_global = x.mean(dim=1)                           # (B, d_model)
        pred_tempo_dist = self.tempo_head(x_global)         # (B, num_tempo_bins)
        pred_mode_logits = self.mode_head(x_global)         # (B, 4)
        pred_features = self.features_head(x_global)        # (B, feat_dim)
        pred_style = F.normalize(self.style_head(x_global), p=2, dim=-1)  # L2归一化

        return {
            "pred_tokens":      pred_tokens,
            "pred_pitch":       pred_pitch,
            "pred_duration":    pred_duration,
            "pred_ornament":    pred_ornament,
            "pred_tempo_dist":  pred_tempo_dist,
            "pred_mode_logits": pred_mode_logits,
            "pred_features":    pred_features,
            "pred_style":       pred_style,
        }


# =================================================================
# 南音生成版 MusicMamba：多输出头适配 NOEF 五维评估体系
# =================================================================

class NanyinMusicMamba(nn.Module):
    """
    南音符号旋律生成模型，基于 MusicMamba (SelectiveSSM) 架构扩展多输出头。

    核心架构（真实 Mamba，arXiv:2312.00752 Algorithm 1/2）：
      Token Embedding → 位置编码 → [SelectiveSSM × n_layers] → LayerNorm
      → [序列级输出头 + 全局级输出头]
    其中 SelectiveSSM 为真实选择性扫描：
      因果卷积 → x_proj 生成 Δ/B/C → ZOH 离散化
      → 状态累积 h_t = exp(ΔA)·h_{t-1} + ΔB·x_t → y_t = C·h_t + D·x_t

    与 NanyinBoYaTCN 的区别：
      - BoYaTCN 使用 OctaveConv + MultiheadAttention + BiLSTM + TCN 时序建模
      - MusicMamba 使用 Selective State Space Model (Mamba) 进行长序列建模，
        理论优势在于 O(N) 线性复杂度，适合南音曲目中的长时依赖关系捕捉

    输出头设计（完全对应 loss_nanyin.py 中 nanyin_total_loss() 入参）：
      - 序列级 (per-timestep): pred_tokens, pred_pitch, pred_duration, pred_ornament
      - 全局级 (pooled):      pred_tempo_dist, pred_mode_logits, pred_features, pred_style

    文献来源：
      MelodyGLM         https://arxiv.org/pdf/2309.10738v1
      Controllable Symbolic Music Generation https://www.preprints.org/manuscript/202604.0984
      TCSinger           https://arxiv.org/pdf/2409.15977
      NanyinHGNN         arXiv:2510.03617
      KAD                arXiv:2502.15602
        其中 KAD 文献采用 State Space Model 进行音乐序列建模，
        为 MusicMamba 架构的选择提供了参考依据。
    """

    def __init__(
        self,
        vocab_size: int = 128,
        d_model: int = 256,
        num_layers: int = 3,
        d_state: int = 16,
        ssm_expand: int = 2,
        num_tempo_bins: int = 20,
        num_modes: int = 4,
        feat_dim: int = 13,
        style_dim: int = 32,
        max_seq_len: int = 4096,
        dropout: float = 0.2,
    ):
        """
        参数:
            vocab_size:      词汇表大小（MIDI token范围0~127 + 特殊token）
            d_model:         隐藏维度
            num_layers:      SelectiveSSM 层数
            d_state:         SSM 状态维度
            ssm_expand:      SSM 内部扩展因子
            num_tempo_bins:  速度分布分箱数（与 loss_nanyin.py 对齐）
            num_modes:       调式类别数（南音4种专属调式）
            feat_dim:        MFCC特征维度
            style_dim:       风格嵌入维度
            max_seq_len:     最大序列长度（默认8192，覆盖最长南音曲目~4643音符）
            dropout:         Dropout概率
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        # ---- Token嵌入 + 位置编码 ----
        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)

        # ---- 堆叠 SelectiveSSM 层 ----
        # Mamba 的线性 O(N) 复杂度适合南音长曲目的序列建模（文献：KAD arXiv:2502.15602）
        self.ssm_layers = nn.ModuleList([
            SelectiveSSM(d_model=d_model, d_state=d_state, expand=ssm_expand)
            for _ in range(num_layers)
        ])

        # ---- LayerNorm 稳定训练 ----
        self.norm = nn.LayerNorm(d_model)

        # ---- Dropout ----
        self.dropout = nn.Dropout(dropout)

        # ========== 输出头（与 NanyinBoYaTCN 完全一致） ==========

        # 序列级输出头（per-timestep）
        self.token_head = nn.Linear(d_model, vocab_size)        # token分类logits
        self.pitch_head = nn.Linear(d_model, 1)                  # 音高回归
        self.duration_head = nn.Linear(d_model, 1)              # 时值回归
        self.ornament_head = nn.Sequential(                      # 装饰音二分类
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )

        # 全局级输出头（序列均值池化后）
        self.tempo_head = nn.Linear(d_model, num_tempo_bins)    # 速度分布
        self.mode_head = nn.Linear(d_model, num_modes)           # 调式logits
        self.features_head = nn.Linear(d_model, feat_dim)        # MFCC特征
        self.style_head = nn.Linear(d_model, style_dim)          # 风格嵌入

    def forward(self, input_tokens: torch.Tensor) -> dict:
        """
        前向传播。

        参数:
            input_tokens: token ID序列, shape=(batch, seq_len), dtype=long
        返回:
            字典，包含所有预测输出（与 NanyinBoYaTCN.forward() 输出结构完全一致）：
              - "pred_tokens":     (batch, seq_len, vocab_size)
              - "pred_pitch":      (batch, seq_len)
              - "pred_duration":   (batch, seq_len)
              - "pred_ornament":   (batch, seq_len)
              - "pred_tempo_dist":  (batch, num_tempo_bins)
              - "pred_mode_logits": (batch, num_modes)
              - "pred_features":    (batch, feat_dim)
              - "pred_style":       (batch, style_dim)
        """
        # ---- 1. Token Embedding ---- (batch, seq_len) → (batch, seq_len, d_model)
        x = self.token_embed(input_tokens)  # (B, L, d_model)
        x = x * math.sqrt(self.d_model)     # 缩放嵌入

        # ---- 2. 位置编码 ----
        x = self.pos_encoding(x)            # (B, L, d_model)

        # ---- 3. 依次通过 SelectiveSSM 层（残差连接） ----
        for layer in self.ssm_layers:
            residual = x
            x = layer(x)                     # (B, L, d_model)
            x = residual + self.dropout(x)   # 残差连接 + dropout
            x = self.norm(x)                 # 层归一化

        # ---- 4. 序列级输出 ----
        pred_tokens = self.token_head(x)                   # (B, L, vocab_size)
        pred_pitch = self.pitch_head(x).squeeze(-1)        # (B, L)
        pred_duration = F.softplus(self.duration_head(x).squeeze(-1))  # (B, L) 正值约束
        pred_ornament = self.ornament_head(x).squeeze(-1)  # (B, L) in [0,1]

        # ---- 5. 全局级输出（序列均值池化）----
        x_global = x.mean(dim=1)                           # (B, d_model)
        pred_tempo_dist = self.tempo_head(x_global)         # (B, num_tempo_bins)
        pred_mode_logits = self.mode_head(x_global)         # (B, 4)
        pred_features = self.features_head(x_global)        # (B, feat_dim)
        pred_style = F.normalize(self.style_head(x_global), p=2, dim=-1)  # L2归一化

        return {
            "pred_tokens":      pred_tokens,
            "pred_pitch":       pred_pitch,
            "pred_duration":    pred_duration,
            "pred_ornament":    pred_ornament,
            "pred_tempo_dist":  pred_tempo_dist,
            "pred_mode_logits": pred_mode_logits,
            "pred_features":    pred_features,
            "pred_style":       pred_style,
        }
