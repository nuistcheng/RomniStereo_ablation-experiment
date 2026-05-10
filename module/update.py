import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DepthHead(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128, output_dim=1):
        super(DepthHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, output_dim, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class ConvGRU(nn.Module):
    def __init__(self, hidden_dim, input_dim, kernel_size=3):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size//2)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size//2)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size//2)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q
        return h


class MotionEncoder(nn.Module):
    def __init__(self, cor_planes, c1_planes=64, c2_planes=64,
                 d1_planes=64, d2_planes=64, out_planes=128):
        super(MotionEncoder, self).__init__()
        self.convc1 = nn.Conv2d(cor_planes, c1_planes, 1, padding=0)
        self.convc2 = nn.Conv2d(c1_planes, c2_planes, 3, padding=1)
        self.convd1 = nn.Conv2d(1, d1_planes, 7, padding=3)
        self.convd2 = nn.Conv2d(d1_planes, d2_planes, 3, padding=1)
        self.conv = nn.Conv2d(c2_planes+d2_planes, out_planes-1, 3, padding=1)

    def forward(self, corr, invdepth):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        dep = F.relu(self.convd1(invdepth))
        dep = F.relu(self.convd2(dep))
        cor_dep = torch.cat([cor, dep], dim=1)
        out = F.relu(self.conv(cor_dep))
        return torch.cat([out, invdepth], dim=1)


class PrevDepthEncoder(nn.Module):
    def __init__(self, out_dim=32):
        super(PrevDepthEncoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_dim, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, prev_depth):
        return self.conv(prev_depth)


class PositionToAngleEncoding(nn.Module):
    """
    动态计算球面角度编码，无需预先指定尺寸。
    第一次 forward 时根据实际输入尺寸生成，之后缓存复用。

    输出 4 通道：[sin(φ), cos(φ), sin(θ), cos(θ)]
      φ ∈ [-π/2, π/2]  仰角，由行决定
      θ ∈ [-π,   π  ]  方位角，由列决定
    """
    def __init__(self):
        super().__init__()
        self._cached_size = (0, 0)
        self._cached_enc  = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        # 尺寸变化时重新计算，否则直接复用缓存
        if (H, W) != self._cached_size:
            # 角度编码为纯确定性位置函数，无可学习参数，
            # 用 no_grad 避免 autograd 追踪中间张量，减少显存占用
            with torch.no_grad():
                phi   = torch.linspace(-math.pi / 2, math.pi / 2, H, device=x.device)
                theta = torch.linspace(-math.pi,     math.pi,     W, device=x.device)

                sin_phi   = phi.sin().view(1, 1, -1, 1).expand(1, 1, H, W)
                cos_phi   = phi.cos().view(1, 1, -1, 1).expand(1, 1, H, W)
                sin_theta = theta.sin().view(1, 1, 1, -1).expand(1, 1, H, W)
                cos_theta = theta.cos().view(1, 1, 1, -1).expand(1, 1, H, W)

                self._cached_enc = torch.cat(
                    [sin_phi, cos_phi, sin_theta, cos_theta], dim=1
                ).contiguous()  # requires_grad=False
                self._cached_size = (H, W)

        # expand 返回共享内存的视图，无额外拷贝；detach 确保不进入反向图
        return self._cached_enc.expand(B, -1, -1, -1).detach()


class AngleAwareChannelAttention(nn.Module):
    """
    通道注意力 + 球面角度偏置：
      weight = σ( MLP(AvgPool(x)) + MLP(MaxPool(x)) + AngleBias(AvgPool(angle)) )
    """
    def __init__(self, in_channels: int, angle_dim: int = 4, reduction: int = 16):
        super().__init__()
        mid = max(in_channels // reduction, 8)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, in_channels, 1, bias=False),
        )
        self.angle_bias = nn.Sequential(
            nn.Conv2d(angle_dim, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, in_channels, 1, bias=False),
        )
        # 零初始化最终层：使初始 sigmoid 输入趋近 0，配合残差连接实现近似 identity 初始化
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.angle_bias[-1].weight)

    def forward(self, x: torch.Tensor, angle_enc: torch.Tensor) -> torch.Tensor:
        avg_w   = self.mlp(self.avg_pool(x))
        max_w   = self.mlp(self.max_pool(x))
        angle_w = self.angle_bias(self.avg_pool(angle_enc))
        weight  = torch.sigmoid(avg_w + max_w + angle_w)
        return x * weight


class SpatialAttention(nn.Module):
    """标准 CBAM 空间注意力"""
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size,
                              padding=kernel_size // 2, bias=False)
        # 零初始化：配合残差连接，初始空间权重 sigmoid(0)=0.5，不影响主干梯度传播
        nn.init.zeros_(self.conv.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_map = x.mean(dim=1, keepdim=True)
        max_map, _ = x.max(dim=1, keepdim=True)
        weight = torch.sigmoid(
            self.conv(torch.cat([avg_map, max_map], dim=1))
        )
        return x * weight


class AngleAwareAttentionFusion(nn.Module):
    """
    三路特征（inp + motion_feat + prev_feat）拼接后，
    经角度感知通道注意力 + 空间注意力精炼。
    输出维度与原始 cat 完全一致，GRU 无需改动。
    """
    def __init__(self, inp_dim: int, motion_dim: int, prev_dim: int = 32,
                 angle_dim: int = 4, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        fused_dim = inp_dim + motion_dim + prev_dim
        self.channel_attn = AngleAwareChannelAttention(fused_dim, angle_dim, reduction)
        self.spatial_attn = SpatialAttention(spatial_kernel)

    def forward(self, inp: torch.Tensor,
                motion_feat: torch.Tensor,
                prev_feat: torch.Tensor,
                angle_enc: torch.Tensor) -> torch.Tensor:
        x = torch.cat([inp, motion_feat, prev_feat], dim=1)
        # 残差连接：x_attn = x * w_c * w_s，初始约 0.25×；
        # 加上原始 x 后初始输出 ≈ 1.25×，配合零初始化后梯度可从第一步有效传播
        x_attn = self.channel_attn(x, angle_enc)
        x_attn = self.spatial_attn(x_attn)
        return x + x_attn


class UpdateBlock(nn.Module):
    def __init__(self, opts, hidden_dim: int, input_dim: int,
                 use_sae: bool = True, use_attn: bool = True, use_ihde: bool = True):
        super(UpdateBlock, self).__init__()
        self.opts = opts
        encoder_output_dim = 128
        self.prev_depth_dim = 32

        # 消融实验开关
        self.use_sae  = use_sae   # 球面角度编码
        self.use_attn = use_attn  # 角度感知通道注意力 + 空间注意力
        self.use_ihde = use_ihde  # 迭代历史深度编码器

        self.encoder = MotionEncoder(opts.corr_levels * (2 * opts.corr_radius + 1))

        # IHDE：迭代历史深度编码器（关闭时始终使用全零占位）
        if self.use_ihde:
            self.prev_depth_enc = PrevDepthEncoder(out_dim=self.prev_depth_dim)

        # SAE：球面角度编码（关闭时向通道注意力传入全零编码）
        if self.use_sae:
            self.angle_encoding = PositionToAngleEncoding()

        # Attn：角度感知通道注意力 + 空间注意力（关闭时直接拼接原始特征）
        if self.use_attn:
            self.attn_fusion = AngleAwareAttentionFusion(
                inp_dim        = input_dim,
                motion_dim     = encoder_output_dim,
                prev_dim       = self.prev_depth_dim,
                angle_dim      = 4,
                reduction      = 16,
                spatial_kernel = 7,
            )

        # GRU 输入维度 = input_dim + encoder_output_dim + prev_depth_dim（与原始保持一致）
        self.gru = ConvGRU(
            hidden_dim,
            encoder_output_dim + input_dim + self.prev_depth_dim
        )
        self.depth_head = DepthHead(hidden_dim, hidden_dim=128, output_dim=1)

        factor = 2 ** self.opts.num_downsample
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, (factor ** 2) * 9, 1, padding=0),
        )

        # 预分配零张量缓存（use_ihde=False 时复用，避免每次 forward 重新申请显存）
        self._zeros_prev_feat: torch.Tensor = None
        self._zeros_prev_shape: tuple = None

    def _get_zeros_prev_feat(self, inv_depth: torch.Tensor) -> torch.Tensor:
        """返回缓存的零张量（形状不变时复用，避免重复申请显存）"""
        B, _, H, W = inv_depth.shape
        shape = (B, self.prev_depth_dim, H, W)
        if self._zeros_prev_feat is None or self._zeros_prev_shape != shape:
            self._zeros_prev_feat = torch.zeros(
                shape, device=inv_depth.device, dtype=inv_depth.dtype
            )
            self._zeros_prev_shape = shape
        return self._zeros_prev_feat

    def forward(self, net, inp, corr=None, inv_depth=None,
                prev_depth=None, no_upsample=False):
        motion_feat = self.encoder(corr, inv_depth)

        # IHDE：开启时编码前一帧深度；关闭时复用预分配的零张量，避免重复申请显存
        if self.use_ihde and prev_depth is not None:
            prev_feat = self.prev_depth_enc(prev_depth)
        else:
            prev_feat = self._get_zeros_prev_feat(inv_depth)

        # Attn 和 SAE 合并处理：
        #   - use_attn=False 时跳过 angle 分配，直接拼接原始特征送入 GRU
        #   - use_attn=True  时才计算 angle（use_sae=False 则传全零编码）
        #   - 训练阶段使用梯度检查点，以重计算换显存（不保留 attn 中间激活）
        if self.use_attn:
            if self.use_sae:
                angle = self.angle_encoding(inp)   # requires_grad=False（已 detach）
            else:
                # use_sae=False：向通道注意力传入全零角度编码
                B, _, H, W = inp.shape
                angle = torch.zeros(B, 4, H, W, device=inp.device, dtype=inp.dtype)

            # 外层已有迭代级梯度检查点（network.py），此处直接调用即可，
            # 嵌套双层检查点会额外保存一份输入张量，反而增加显存开销
            inp_fused = self.attn_fusion(inp, motion_feat, prev_feat, angle)
        else:
            # use_attn=False：直接拼接，不分配 angle 张量
            inp_fused = torch.cat([inp, motion_feat, prev_feat], dim=1)

        net = self.gru(net, inp_fused)
        delta_inv_depth = self.depth_head(net)

        if no_upsample:
            return net, delta_inv_depth, None

        mask = .25 * self.mask(net)
        return net, delta_inv_depth, mask