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
            phi   = torch.linspace(-math.pi / 2, math.pi / 2, H, device=x.device)
            theta = torch.linspace(-math.pi,     math.pi,     W, device=x.device)

            sin_phi   = phi.sin().view(1, 1, -1, 1).expand(1, 1, H, W)
            cos_phi   = phi.cos().view(1, 1, -1, 1).expand(1, 1, H, W)
            sin_theta = theta.sin().view(1, 1, 1, -1).expand(1, 1, H, W)
            cos_theta = theta.cos().view(1, 1, 1, -1).expand(1, 1, H, W)

            self._cached_enc = torch.cat(
                [sin_phi, cos_phi, sin_theta, cos_theta], dim=1
            ).contiguous()
            self._cached_size = (H, W)

        return self._cached_enc.expand(B, -1, -1, -1)


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
        x = self.channel_attn(x, angle_enc)
        x = self.spatial_attn(x)
        return x


class UpdateBlock(nn.Module):
    def __init__(self, opts, hidden_dim: int, input_dim: int):
        super(UpdateBlock, self).__init__()
        self.opts = opts
        encoder_output_dim = 128
        self.prev_depth_dim = 32

        self.encoder        = MotionEncoder(opts.corr_levels * (2 * opts.corr_radius + 1))
        self.prev_depth_enc = PrevDepthEncoder(out_dim=self.prev_depth_dim)

        # 动态角度编码，无需传尺寸
        self.angle_encoding = PositionToAngleEncoding()

        # 三路融合：input_dim + encoder_output_dim + prev_depth_dim
        self.attn_fusion = AngleAwareAttentionFusion(
            inp_dim        = input_dim,
            motion_dim     = encoder_output_dim,
            prev_dim       = self.prev_depth_dim,
            angle_dim      = 4,
            reduction      = 16,
            spatial_kernel = 7,
        )

        # GRU 输入维度 = input_dim + encoder_output_dim + prev_depth_dim
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

    def forward(self, net, inp, corr=None, inv_depth=None,
                prev_depth=None, no_upsample=False):
        motion_feat = self.encoder(corr, inv_depth)

        # 前一帧深度（无则补零）
        if prev_depth is not None:
            prev_feat = self.prev_depth_enc(prev_depth)
        else:
            B, _, H, W = inv_depth.shape
            prev_feat = torch.zeros(
                B, self.prev_depth_dim, H, W,
                device=inv_depth.device, dtype=inv_depth.dtype
            )

        # 球面角度编码（动态适配当前尺寸）
        angle = self.angle_encoding(inp)

        # 角度感知注意力融合
        inp_fused = self.attn_fusion(inp, motion_feat, prev_feat, angle)

        net = self.gru(net, inp_fused)
        delta_inv_depth = self.depth_head(net)

        if no_upsample:
            return net, delta_inv_depth, None

        mask = .25 * self.mask(net)
        return net, delta_inv_depth, mask