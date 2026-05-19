# filora/cm_hetecd_filora.py

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class ConvBNAct(nn.Layer):
    def __init__(self, in_c, out_c, k=3, s=1, p=1, act=True):
        super().__init__()
        layers = [
            nn.Conv2D(in_c, out_c, kernel_size=k, stride=s, padding=p, bias_attr=False),
            nn.BatchNorm2D(out_c),
        ]
        if act:
            layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Layer):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = ConvBNAct(dim, dim, 3, 1, 1)
        self.conv2 = ConvBNAct(dim, dim, 3, 1, 1, act=False)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.conv2(self.conv1(x)))


class ModalityStem(nn.Layer):
    """
    Optical 和 SAR 的浅层分布差异很大，因此先用不同 stem 提取模态私有特征。
    XiongAn 数据中建议：
        Optical: RGB 3 channels
        SAR: HH/VV/HV or pseudo-color SAR 3 channels
    """

    def __init__(self, in_chans=3, out_chans=32):
        super().__init__()
        self.net = nn.Sequential(
            ConvBNAct(in_chans, out_chans, 3, 1, 1),
            ConvBNAct(out_chans, out_chans, 3, 1, 1),
        )

    def forward(self, x):
        return self.net(x)


class HeteEncoder(nn.Layer):
    """
    非共享权重双分支编码器。
    这是借鉴 HeteCD 的核心设计：
        optical branch 和 SAR branch 不共享参数，
        避免异构模态被强制映射到同一个低层特征空间。
    """

    def __init__(self, in_chans=3, base_dim=32):
        super().__init__()

        self.stem = ModalityStem(in_chans, base_dim)

        self.stage1 = nn.Sequential(
            ConvBNAct(base_dim, 64, 3, 2, 1),       # H/2
            ResidualBlock(64),
        )

        self.stage2 = nn.Sequential(
            ConvBNAct(64, 128, 3, 2, 1),           # H/4
            ResidualBlock(128),
        )

        self.stage3 = nn.Sequential(
            ConvBNAct(128, 256, 3, 2, 1),          # H/8
            ResidualBlock(256),
        )

        self.stage4 = nn.Sequential(
            ConvBNAct(256, 512, 3, 2, 1),          # H/16
            ResidualBlock(512),
        )

    def forward(self, x):
        x = self.stem(x)

        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)

        return [f1, f2, f3, f4]


class FourierBlock(nn.Layer):
    """
    FILoRA 频域思想的 Paddle 实现。
    对特征做 FFT，实部/虚部分离卷积，再 IFFT 回空间域。
    """

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2D(channels * 2, channels * 2, kernel_size=1, bias_attr=False)
        self.act = nn.GELU()

    def forward(self, x):
        b, c, h, w = x.shape

        ffted = paddle.fft.rfft2(x, norm="ortho")

        real = paddle.unsqueeze(ffted.real(), axis=-1)
        imag = paddle.unsqueeze(ffted.imag(), axis=-1)

        freq = paddle.concat([real, imag], axis=-1)        # [B, C, H, W/2+1, 2]
        freq = freq.transpose([0, 1, 4, 2, 3])             # [B, C, 2, H, W/2+1]
        freq = freq.reshape([b, c * 2, h, w // 2 + 1])     # [B, 2C, H, W/2+1]

        freq = self.act(self.conv(freq))

        freq = freq.reshape([b, c, 2, h, w // 2 + 1])
        freq = freq.transpose([0, 1, 3, 4, 2])             # [B, C, H, W/2+1, 2]

        real, imag = paddle.split(freq, 2, axis=-1)
        real = paddle.squeeze(real, axis=-1)
        imag = paddle.squeeze(imag, axis=-1)

        ffted = paddle.complex(real, imag)
        out = paddle.fft.irfft2(ffted, s=(h, w), norm="ortho")

        return out


class FrequencyEnhance(nn.Layer):
    """
    频域增强模块。
    保留 FILoRA 的 FDFE 思想，用来抑制 SAR speckle 和异构纹理伪变化。
    """

    def __init__(self, dim):
        super().__init__()
        self.pre = ConvBNAct(dim, dim, 1, 1, 0)
        self.fourier = FourierBlock(dim)
        self.post = nn.Sequential(
            ConvBNAct(dim, dim, 3, 1, 1),
            ResidualBlock(dim),
        )

    def forward(self, x):
        y = self.pre(x)
        y = y + self.fourier(y)
        y = self.post(y)
        return y


class STADifference(nn.Layer):
    """
    3D Spatio-temporal Attention Difference module.

    借鉴 HeteCD:
        1. 把 optical feature、SAR feature、abs difference 作为 temporal-like sequence
        2. 用 3D conv 提取时空差异
        3. 用 temporal attention 和 spatial attention 强化真实变化区域

    输入:
        fa, fb: [B, C, H, W]
    输出:
        diff: [B, C, H, W]
    """

    def __init__(self, dim):
        super().__init__()

        self.conv3 = nn.Sequential(
            nn.Conv3D(dim, dim, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias_attr=False),
            nn.BatchNorm3D(dim),
            nn.GELU(),
        )

        self.conv5 = nn.Sequential(
            nn.Conv3D(dim, dim, kernel_size=(3, 5, 5), padding=(1, 2, 2), bias_attr=False),
            nn.BatchNorm3D(dim),
            nn.GELU(),
        )

        self.conv7 = nn.Sequential(
            nn.Conv3D(dim, dim, kernel_size=(3, 7, 7), padding=(1, 3, 3), bias_attr=False),
            nn.BatchNorm3D(dim),
            nn.GELU(),
        )

        self.fuse3d = nn.Sequential(
            nn.Conv3D(dim * 3, dim, kernel_size=1, bias_attr=False),
            nn.BatchNorm3D(dim),
            nn.GELU(),
        )

        self.spatial_attn = nn.Sequential(
            nn.Conv2D(dim, 1, kernel_size=7, padding=3),
            nn.Sigmoid(),
        )

        self.out = nn.Sequential(
            ConvBNAct(dim, dim, 3, 1, 1),
            ResidualBlock(dim),
        )

    def forward(self, fa, fb):
        diff = paddle.abs(fa - fb)

        # [B, C, T=3, H, W]
        x = paddle.stack([fa, fb, diff], axis=2)

        y3 = self.conv3(x)
        y5 = self.conv5(x)
        y7 = self.conv7(x)

        y = paddle.concat([y3, y5, y7], axis=1)
        y = self.fuse3d(y)                                # [B, C, T, H, W]

        # temporal attention, [B, 1, T, 1, 1]
        t_score = paddle.mean(y, axis=[1, 3, 4], keepdim=True)
        t_attn = F.softmax(t_score, axis=2)
        y = y * t_attn

        # merge temporal dimension
        y = paddle.mean(y, axis=2)                         # [B, C, H, W]

        # spatial attention
        s_attn = self.spatial_attn(y)                      # [B, 1, H, W]
        y = y * s_attn

        y = self.out(y)
        return y


class PyramidHeteDecoder(nn.Layer):
    """
    多层级异构差异解码器。
    对 4 个尺度分别做 3D-STA difference，然后统一上采样融合。
    """

    def __init__(self, in_dims=(64, 128, 256, 512), decoder_dim=128):
        super().__init__()

        self.proj_a = nn.LayerList([
            ConvBNAct(in_c, decoder_dim, 1, 1, 0) for in_c in in_dims
        ])

        self.proj_b = nn.LayerList([
            ConvBNAct(in_c, decoder_dim, 1, 1, 0) for in_c in in_dims
        ])

        self.diff_blocks = nn.LayerList([
            STADifference(decoder_dim) for _ in in_dims
        ])

        self.fuse = nn.Sequential(
            ConvBNAct(decoder_dim * 4, decoder_dim, 1, 1, 0),
            FrequencyEnhance(decoder_dim),
            ConvBNAct(decoder_dim, decoder_dim, 3, 1, 1),
            ResidualBlock(decoder_dim),
        )

        self.pred = nn.Sequential(
            ConvBNAct(decoder_dim, 64, 3, 1, 1),
            nn.Conv2D(64, 2, kernel_size=1),
        )

    def forward(self, feats_a, feats_b):
        target_size = feats_a[1].shape[2:]  # use H/4 scale

        diff_feats = []
        align_feats_a = []
        align_feats_b = []

        for i in range(4):
            fa = self.proj_a[i](feats_a[i])
            fb = self.proj_b[i](feats_b[i])

            align_feats_a.append(fa)
            align_feats_b.append(fb)

            d = self.diff_blocks[i](fa, fb)

            if d.shape[2:] != target_size:
                d = F.interpolate(d, size=target_size, mode="bilinear", align_corners=True)

            diff_feats.append(d)

        x = paddle.concat(diff_feats, axis=1)
        x = self.fuse(x)
        logits = self.pred(x)

        return logits, align_feats_a, align_feats_b


class CMHeteCDFILoRA_BCD(nn.Layer):
    """
    最终模型：
        CM-HeteCD-FILoRA

    输入:
        x: [B, 6, H, W]
           first 3 channels  = Optical RGB
           last 3 channels   = SAR HH/VV/HV or pseudo-color SAR

    输出:
        logits: [B, 2, H, W]

    训练时 return_aux=True:
        返回 FCA loss 所需的中间特征。
    """

    def __init__(
        self,
        img_size=512,
        num_cls=2,
        in_chans_t1=3,
        in_chans_t2=3,
        base_dim=32,
        decoder_dim=128,
    ):
        super().__init__()

        self.img_size = [img_size, img_size]
        self.num_cls = num_cls
        self.in_chans_t1 = in_chans_t1
        self.in_chans_t2 = in_chans_t2

        self.optical_encoder = HeteEncoder(in_chans=in_chans_t1, base_dim=base_dim)
        self.sar_encoder = HeteEncoder(in_chans=in_chans_t2, base_dim=base_dim)

        self.decoder = PyramidHeteDecoder(
            in_dims=(64, 128, 256, 512),
            decoder_dim=decoder_dim,
        )

    def forward(self, x1, x2=None, return_aux=False):
        if x2 is None:
            expected = self.in_chans_t1 + self.in_chans_t2
            if x1.shape[1] != expected:
                raise ValueError(
                    f"Input channel mismatch: got {x1.shape[1]}, "
                    f"expected {expected} = {self.in_chans_t1}+{self.in_chans_t2}."
                )

            optical, sar = paddle.split(
                x1,
                num_or_sections=[self.in_chans_t1, self.in_chans_t2],
                axis=1,
            )
        else:
            optical, sar = x1, x2

        feats_opt = self.optical_encoder(optical)
        feats_sar = self.sar_encoder(sar)

        logits, align_opt, align_sar = self.decoder(feats_opt, feats_sar)

        logits = F.interpolate(
            logits,
            size=self.img_size,
            mode="bilinear",
            align_corners=True,
        )

        if return_aux:
            return {
                "logits": logits,
                "fca_opt": align_opt,
                "fca_sar": align_sar,
            }

        return logits
