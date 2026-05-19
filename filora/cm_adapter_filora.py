# filora/cm_adapter_filora.py

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from .modules import (
    CrossModalCoarseDifferenceFeaturesExtraction,
    FrequencyDomainFeatureEnhance,
    MultiScaleChangeFusion,
)


class ConvBNAct(nn.Layer):
    def __init__(self, in_c, out_c, k=3, s=1, p=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2D(in_c, out_c, kernel_size=k, stride=s, padding=p, bias_attr=False),
            nn.BatchNorm2D(out_c),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Layer):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = ConvBNAct(dim, dim, 3, 1, 1)
        self.conv2 = nn.Sequential(
            nn.Conv2D(dim, dim, kernel_size=3, stride=1, padding=1, bias_attr=False),
            nn.BatchNorm2D(dim),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.conv2(self.conv1(x)))


class ModalityAdapter(nn.Layer):
    """
    Modality-specific adapter.

    RGB and SAR have different statistics:
    RGB: color / texture / illumination
    SAR: backscatter / speckle / structure

    This adapter maps each modality into a common feature dimension.
    """

    def __init__(self, in_chans, out_chans=32):
        super().__init__()
        self.net = nn.Sequential(
            ConvBNAct(in_chans, out_chans, 3, 1, 1),
            ConvBNAct(out_chans, out_chans, 3, 1, 1),
        )

    def forward(self, x):
        return self.net(x)


class PrivateEncoder(nn.Layer):
    """
    Private encoder for each modality.

    The early layers are not shared because RGB and SAR low-level patterns
    are very different.
    """

    def __init__(self, in_chans=32, dim=256):
        super().__init__()
        self.net = nn.Sequential(
            ConvBNAct(in_chans, 64, 3, 2, 1),    # H/2
            ResidualBlock(64),
            ConvBNAct(64, 128, 3, 2, 1),        # H/4
            ResidualBlock(128),
            ConvBNAct(128, dim, 3, 2, 1),       # H/8
            ResidualBlock(dim),
        )

    def forward(self, x):
        return self.net(x)


class SharedEncoder(nn.Layer):
    """
    Shared high-level encoder.

    Since we currently do not load pretrained weights, this module should
    NOT be frozen during training.
    """

    def __init__(self, dim=256, depth=2):
        super().__init__()
        self.blocks = nn.Sequential(*[ResidualBlock(dim) for _ in range(depth)])

    def forward(self, x):
        return self.blocks(x)


class CrossModalAlignment(nn.Layer):
    """
    Lightweight cross-modal alignment.

    It uses f_rgb, f_sar, |f_rgb-f_sar| and f_rgb*f_sar to generate
    bidirectional gates, reducing modality-induced pseudo changes.
    """

    def __init__(self, dim=256):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2D(dim * 4, dim, kernel_size=1),
            nn.BatchNorm2D(dim),
            nn.GELU(),
            nn.Conv2D(dim, dim * 2, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, f_rgb, f_sar):
        diff = paddle.abs(f_rgb - f_sar)
        prod = f_rgb * f_sar
        x = paddle.concat([f_rgb, f_sar, diff, prod], axis=1)

        gates = self.gate(x)
        g_rgb, g_sar = paddle.split(gates, num_or_sections=2, axis=1)

        f_rgb0 = f_rgb
        f_sar0 = f_sar

        f_rgb = f_rgb0 + g_rgb * f_sar0
        f_sar = f_sar0 + g_sar * f_rgb0

        return f_rgb, f_sar


class CMAdapterFILoRA_BCD(nn.Layer):
    """
    CM-Adapter-FILoRA for RGB-SAR binary change detection.

    Input:
        x: [B, 4, H, W], channel order = RGB t1 + SAR t2
    or:
        x_rgb: [B, 3, H, W]
        x_sar: [B, 1, H, W]

    Output:
        logits: [B, num_cls, H, W]
    """

    def __init__(
        self,
        img_size=256,
        num_cls=2,
        in_chans_t1=3,
        in_chans_t2=1,
        dim=256,
        shared_depth=2,
    ):
        super().__init__()

        self.img_size = [img_size, img_size]
        self.in_chans_t1 = in_chans_t1
        self.in_chans_t2 = in_chans_t2

        self.rgb_adapter = ModalityAdapter(in_chans_t1, 32)
        self.sar_adapter = ModalityAdapter(in_chans_t2, 32)

        self.rgb_private = PrivateEncoder(32, dim)
        self.sar_private = PrivateEncoder(32, dim)

        self.shared_encoder = SharedEncoder(dim, depth=shared_depth)
        self.cma = CrossModalAlignment(dim)

        self.cife = CrossModalCoarseDifferenceFeaturesExtraction(dim)
        self.fdfe = FrequencyDomainFeatureEnhance(dim, 128, 64)
        self.mscf = MultiScaleChangeFusion(64)

        self.cls = nn.Conv2D(64, num_cls, kernel_size=3, stride=1, padding=1)

    def forward(self, x1, x2=None):
        """
        x1 mode:
            x1 = [B, 4, H, W], RGB3 + SAR1

        x1/x2 mode:
            x1 = RGB [B, 3, H, W]
            x2 = SAR [B, 1, H, W]
        """

        if x2 is None:
            expected = self.in_chans_t1 + self.in_chans_t2
            if x1.shape[1] != expected:
                raise ValueError(
                    f"Input channel mismatch: got {x1.shape[1]}, "
                    f"expected {expected} = RGB{self.in_chans_t1} + SAR{self.in_chans_t2}."
                )

            x_rgb, x_sar = paddle.split(
                x1,
                num_or_sections=[self.in_chans_t1, self.in_chans_t2],
                axis=1,
            )
        else:
            x_rgb, x_sar = x1, x2

        f_rgb = self.rgb_adapter(x_rgb)
        f_sar = self.sar_adapter(x_sar)

        f_rgb = self.rgb_private(f_rgb)
        f_sar = self.sar_private(f_sar)

        f_rgb = self.shared_encoder(f_rgb)
        f_sar = self.shared_encoder(f_sar)

        f_rgb, f_sar = self.cma(f_rgb, f_sar)

        y = self.cife(f_rgb, f_sar)
        y = self.fdfe(y)
        y = self.mscf(y)

        y = F.interpolate(
            y,
            size=self.img_size,
            mode="bilinear",
            align_corners=True,
        )

        y = self.cls(y)
        return y
