# filora/cm_hetecd_sam2_sf_v1_torch.py
# PyTorch implementation for CM-HeteCD-FILoRA with official SAM2 image encoder + LoRA.
# Put this file inside your local CM-HeteCD-FILoRA repo:
#   filora/cm_hetecd_sam2_sf_v1_torch.py
#
# Important:
# - Official SAM2 is PyTorch. This file cannot be trained by the original Paddle trainer directly.
# - You need to use a PyTorch training loop, or port your dataset/loader to PyTorch.
# - This model uses only SAM2 image_encoder; prompt encoder and mask decoder are not used.
# - Optical and SAR branches use two independent SAM2 image encoders with independent LoRA modules.

import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from sam2.build_sam import build_sam2
except Exception as e:
    build_sam2 = None
    _SAM2_IMPORT_ERROR = e


# -----------------------------
# LoRA
# -----------------------------
class LoRALinear(nn.Module):
    """Wrap nn.Linear with a trainable LoRA update."""

    def __init__(self, linear: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        if not isinstance(linear, nn.Linear):
            raise TypeError("LoRALinear only wraps nn.Linear")

        self.linear = linear
        self.r = int(r)
        self.alpha = int(alpha)
        self.scaling = float(alpha) / float(r) if r > 0 else 1.0
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        for p in self.linear.parameters():
            p.requires_grad = False

        if self.r > 0:
            self.lora_A = nn.Parameter(torch.zeros(self.r, linear.in_features))
            self.lora_B = nn.Parameter(torch.zeros(linear.out_features, self.r))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.linear(x)
        if self.r <= 0:
            return base
        # Supports input [..., in_features].
        update = F.linear(self.dropout(x), self.lora_A)
        update = F.linear(update, self.lora_B) * self.scaling
        return base + update


def inject_lora_to_sam2_image_encoder(
    module: nn.Module,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
    target_keywords: Sequence[str] = ("attn.qkv", "attn.proj"),
) -> int:
    """
    Replace selected nn.Linear modules by LoRALinear.
    Default targets SAM2 Hiera attention qkv/proj layers.
    """
    replaced = 0

    def _replace(parent: nn.Module, prefix: str = ""):
        nonlocal replaced
        for name, child in list(parent.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear) and any(k in full_name for k in target_keywords):
                setattr(parent, name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
                replaced += 1
            else:
                _replace(child, full_name)

    _replace(module)
    return replaced


def freeze_non_lora_params(module: nn.Module) -> None:
    for name, p in module.named_parameters():
        p.requires_grad = "lora_" in name


# -----------------------------
# Basic blocks
# -----------------------------
class ConvBNAct(nn.Module):
    def __init__(self, in_c: int, out_c: int, k: int = 3, s: int = 1, p: int = 1, act: bool = True):
        super().__init__()
        layers: List[nn.Module] = [
            nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_c),
        ]
        if act:
            layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv1 = ConvBNAct(dim, dim, 3, 1, 1)
        self.conv2 = ConvBNAct(dim, dim, 3, 1, 1, act=False)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.conv2(self.conv1(x)))


# -----------------------------
# SAM2 image encoder wrapper
# -----------------------------
class SAM2ImageEncoderLoRA(nn.Module):
    """
    Official SAM2 image_encoder wrapper.
    Returns four feature maps [F1, F2, F3, F4], high -> low resolution.

    SAM2.1 config usually has image_encoder.scalp=1, so backbone_fpn often returns
    three maps. We append an extra H/32 map using a trainable stride-2 conv.
    """

    def __init__(
        self,
        sam2_cfg: str,
        sam2_ckpt: str,
        device: str = "cuda",
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        out_dim: int = 256,
        normalize_input: bool = True,
        train_lora: bool = True,
        sam2_repo_dir: Optional[str] = None,
    ):
        super().__init__()
        if build_sam2 is None:
            raise ImportError(
                "Cannot import SAM2. Install official SAM2 first. Original error: "
                f"{repr(_SAM2_IMPORT_ERROR)}"
            )

        sam2_ckpt = os.path.abspath(sam2_ckpt)

        # SAM2's Hydra config is most reliable when built from the SAM2 repo root.
        # If your training script runs from CM-HeteCD-FILoRA root, pass:
        # sam2_repo_dir="/path/to/CM-HeteCD-FILoRA/third_party/sam2".
        old_cwd = os.getcwd()
        try:
            if sam2_repo_dir is not None:
                os.chdir(sam2_repo_dir)
            sam2_model = build_sam2(
                sam2_cfg,
                sam2_ckpt,
                device=device,
                mode="eval",
                apply_postprocessing=False)
            print(f"[SAM2] Loaded checkpoint:{sam2_ckpt}")
            print(f"[SAM2] Config: {sam2_cfg}")
        finally:
            os.chdir(old_cwd)

        self.image_encoder = sam2_model.image_encoder

        # Remove reference to the full model to save memory.
        del sam2_model

        # Freeze base encoder and inject LoRA.
        for p in self.image_encoder.parameters():
            p.requires_grad = False

        self.num_lora_layers = 0
        if train_lora and lora_r > 0:
            self.num_lora_layers = inject_lora_to_sam2_image_encoder(
                self.image_encoder,
                r=lora_r,
                alpha=lora_alpha,
                dropout=lora_dropout,
                target_keywords=("attn.qkv", "attn.proj"),
            )
            print(f"[LoRA] Injected LoRA Linear layers:{self.num_lora_layers}")
            freeze_non_lora_params(self.image_encoder)

        self.extra_down = ConvBNAct(out_dim, out_dim, 3, 2, 1)
        self.normalize_input = normalize_input
        self.register_buffer("pixel_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("pixel_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        # Expected input x in [0, 1]. If your dataset already normalizes images, set normalize_input=False.
        if self.normalize_input:
            x = (x - self.pixel_mean.to(x.dtype)) / self.pixel_std.to(x.dtype)
        return x

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self._normalize(x)
        out = self.image_encoder(x)
        feats = out["backbone_fpn"]

        # Sort high -> low resolution for safety.
        feats = sorted(list(feats), key=lambda t: t.shape[-2] * t.shape[-1], reverse=True)

        if len(feats) >= 4:
            feats = feats[:4]
        elif len(feats) == 3:
            feats = feats + [self.extra_down(feats[-1])]
        else:
            raise RuntimeError(f"SAM2 image_encoder returned {len(feats)} features; expected 3 or 4.")

        return feats


# -----------------------------
# Common / Private decomposition
# -----------------------------
class CommonPrivateDecomposition(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.common = nn.Sequential(
            ConvBNAct(channels, channels, 1, 1, 0),
            ConvBNAct(channels, channels, 3, 1, 1),
        )
        self.private = nn.Sequential(
            ConvBNAct(channels, channels, 1, 1, 0),
            ConvBNAct(channels, channels, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.common(x), self.private(x)


# -----------------------------
# Spatial Difference Branch
# -----------------------------
class SpatialDifferenceBranch(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            ConvBNAct(channels * 6, channels, 3, 1, 1),
            ResidualBlock(channels),
        )

    def forward(self, co: torch.Tensor, cs: torch.Tensor, po: torch.Tensor, ps: torch.Tensor) -> torch.Tensor:
        d_common = torch.abs(co - cs)
        d_private = torch.abs(po - ps)
        x = torch.cat([co, cs, d_common, po, ps, d_private], dim=1)
        return self.net(x)


# -----------------------------
# Haar DWT / IDWT
# -----------------------------
def _haar_filters(device, dtype):
    return torch.tensor(
        [
            [[[0.5, 0.5], [0.5, 0.5]]],
            [[[-0.5, -0.5], [0.5, 0.5]]],
            [[[-0.5, 0.5], [-0.5, 0.5]]],
            [[[0.5, -0.5], [-0.5, 0.5]]],
        ],
        device=device,
        dtype=dtype,
    )


class HaarDWT2D(nn.Module):
    def forward(self, x: torch.Tensor):
        b, c, h, w = x.shape
        pad_h = h % 2
        pad_w = w % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
        weight = _haar_filters(x.device, x.dtype).repeat(c, 1, 1, 1)
        y = F.conv2d(x, weight, stride=2, groups=c)
        _, _, h2, w2 = y.shape
        y = y.contiguous().reshape(b, c, 4, h2, w2)
        return y[:, :, 0], y[:, :, 1], y[:, :, 2], y[:, :, 3]


class HaarIDWT2D(nn.Module):
    def forward(self, ll, lh, hl, hh, out_size: Optional[Tuple[int, int]] = None):
        b, c, h, w = ll.shape
        x = torch.stack([ll, lh, hl, hh], dim=2)
        x = x.contiguous().reshape(b, c * 4, h, w)
        weight = _haar_filters(x.device, x.dtype).repeat(c, 1, 1, 1)
        y = F.conv_transpose2d(x, weight, stride=2, groups=c)
        if out_size is not None:
            y = y[:, :, : out_size[0], : out_size[1]]
        return y


class SubBandAlign(nn.Module):
    def __init__(self, channels: int, residual_scale: float = 1.0):
        super().__init__()
        self.residual_scale = residual_scale
        self.align = nn.Sequential(
            ConvBNAct(channels * 3, channels, 1, 1, 0),
            ConvBNAct(channels, channels, 3, 1, 1),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
        )
        self.gate = nn.Sequential(
            ConvBNAct(channels * 3, channels, 1, 1, 0),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, o_band: torch.Tensor, s_band: torch.Tensor):
        x = torch.cat([o_band, s_band, torch.abs(o_band - s_band)], dim=1)
        residual = self.align(x)
        gate = self.gate(x)
        s_aligned = s_band + self.residual_scale * gate * residual
        return s_aligned, residual


class DWTFrequencyAlignmentBranch(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.dwt = HaarDWT2D()
        self.idwt = HaarIDWT2D()
        self.align_ll = SubBandAlign(channels, residual_scale=1.0)
        self.align_lh = SubBandAlign(channels, residual_scale=0.7)
        self.align_hl = SubBandAlign(channels, residual_scale=0.7)
        self.align_hh = SubBandAlign(channels, residual_scale=0.3)

    def forward(self, common_o: torch.Tensor, common_s: torch.Tensor):
        h, w = common_s.shape[-2:]
        o_ll, o_lh, o_hl, o_hh = self.dwt(common_o)
        s_ll, s_lh, s_hl, s_hh = self.dwt(common_s)

        s_ll_a, r_ll = self.align_ll(o_ll, s_ll)
        s_lh_a, r_lh = self.align_lh(o_lh, s_lh)
        s_hl_a, r_hl = self.align_hl(o_hl, s_hl)
        s_hh_a, r_hh = self.align_hh(o_hh, s_hh)

        s_dwt = self.idwt(s_ll_a, s_lh_a, s_hl_a, s_hh_a, out_size=(h, w))
        aux = {
            "o_bands": [o_ll, o_lh, o_hl, o_hh],
            "s_bands": [s_ll, s_lh, s_hl, s_hh],
            "s_bands_aligned": [s_ll_a, s_lh_a, s_hl_a, s_hh_a],
            "residuals": [r_ll, r_lh, r_hl, r_hh],
        }
        return s_dwt, aux


class FrequencyDifferenceBranch(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            ConvBNAct(channels * 4, channels, 3, 1, 1),
            ResidualBlock(channels),
        )

    def forward(self, common_o: torch.Tensor, s_dwt: torch.Tensor) -> torch.Tensor:
        x = torch.cat([common_o, s_dwt, torch.abs(common_o - s_dwt), common_o * s_dwt], dim=1)
        return self.net(x)


class SpatialFrequencyGateFusion(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.gate = nn.Sequential(
            ConvBNAct(channels * 2, channels, 1, 1, 0),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.refine = nn.Sequential(
            ConvBNAct(channels, channels, 3, 1, 1),
            ResidualBlock(channels),
        )

    def forward(self, ds: torch.Tensor, df: torch.Tensor):
        alpha = self.gate(torch.cat([ds, df], dim=1))
        d = alpha * ds + (1.0 - alpha) * df
        return self.refine(d), alpha


class SFDifferenceBlock(nn.Module):
    def __init__(self, in_c_opt: int, in_c_sar: int, decoder_dim: int = 128):
        super().__init__()
        self.proj_o = ConvBNAct(in_c_opt, decoder_dim, 1, 1, 0)
        self.proj_s = ConvBNAct(in_c_sar, decoder_dim, 1, 1, 0)
        self.decomp_o = CommonPrivateDecomposition(decoder_dim)
        self.decomp_s = CommonPrivateDecomposition(decoder_dim)
        self.spatial_diff = SpatialDifferenceBranch(decoder_dim)
        self.dwt_align = DWTFrequencyAlignmentBranch(decoder_dim)
        self.freq_diff = FrequencyDifferenceBranch(decoder_dim)
        self.gate_fusion = SpatialFrequencyGateFusion(decoder_dim)

    def forward(self, fo: torch.Tensor, fs: torch.Tensor):
        fo = self.proj_o(fo)
        fs = self.proj_s(fs)
        co, po = self.decomp_o(fo)
        cs, ps = self.decomp_s(fs)
        ds = self.spatial_diff(co, cs, po, ps)
        s_dwt, freq_aux = self.dwt_align(co, cs)
        df = self.freq_diff(co, s_dwt)
        d, alpha = self.gate_fusion(ds, df)
        aux = {
            "common_o": co,
            "common_s": cs,
            "private_o": po,
            "private_s": ps,
            "ds": ds,
            "df": df,
            "alpha": alpha,
            "freq": freq_aux,
        }
        return d, aux


class PyramidSFDecoderV1(nn.Module):
    def __init__(self, in_dims=(256, 256, 256, 256), decoder_dim: int = 128, num_classes: int = 2):
        super().__init__()
        self.blocks = nn.ModuleList(
            [SFDifferenceBlock(c, c, decoder_dim=decoder_dim) for c in in_dims]
        )
        self.fuse = nn.Sequential(
            ConvBNAct(decoder_dim * 4, decoder_dim, 1, 1, 0),
            ConvBNAct(decoder_dim, decoder_dim, 3, 1, 1),
            ResidualBlock(decoder_dim),
        )
        self.pred = nn.Sequential(
            ConvBNAct(decoder_dim, 64, 3, 1, 1),
            nn.Conv2d(64, num_classes, kernel_size=1),
        )

    def forward(self, feats_opt: List[torch.Tensor], feats_sar: List[torch.Tensor]):
        target_size = feats_opt[0].shape[-2:]
        diff_feats = []
        aux_list = []
        for i, block in enumerate(self.blocks):
            d, aux = block(feats_opt[i], feats_sar[i])
            if d.shape[-2:] != target_size:
                d = F.interpolate(d, size=target_size, mode="bilinear", align_corners=False)
            diff_feats.append(d)
            aux_list.append(aux)
        x = torch.cat(diff_feats, dim=1)
        x = self.fuse(x)
        logits = self.pred(x)
        return logits, aux_list


# -----------------------------
# Full model
# -----------------------------
class SAM2LoRASFChangeDetectorV1(nn.Module):
    def __init__(
        self,
        sam2_cfg: str,
        sam2_ckpt: str,
        img_size: Tuple[int, int] = (512, 512),
        num_classes: int = 2,
        decoder_dim: int = 128,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        normalize_input: bool = True,
        device: str = "cuda",
        sam2_repo_dir: Optional[str] = None,
    ):
        super().__init__()
        self.img_size = img_size
        self.optical_encoder = SAM2ImageEncoderLoRA(
            sam2_cfg=sam2_cfg,
            sam2_ckpt=sam2_ckpt,
            device=device,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            normalize_input=normalize_input,
            sam2_repo_dir=sam2_repo_dir,
        )
        self.sar_encoder = SAM2ImageEncoderLoRA(
            sam2_cfg=sam2_cfg,
            sam2_ckpt=sam2_ckpt,
            device=device,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            normalize_input=normalize_input,
            sam2_repo_dir=sam2_repo_dir,
        )
        self.decoder = PyramidSFDecoderV1(
            in_dims=(256, 256, 256, 256),
            decoder_dim=decoder_dim,
            num_classes=num_classes,
        )

    def forward(self, optical: torch.Tensor, sar: torch.Tensor, return_aux: bool = False):
        feats_opt = self.optical_encoder(optical)
        feats_sar = self.sar_encoder(sar)
        logits, aux_list = self.decoder(feats_opt, feats_sar)
        logits = F.interpolate(logits, size=self.img_size, mode="bilinear", align_corners=False)
        if return_aux:
            return {"logits": logits, "aux": aux_list}
        return logits


# -----------------------------
# Auxiliary losses, no intermediate labels
# -----------------------------
def _flatten_bhwc(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 2, 3, 1).reshape(-1, x.shape[1])


def coral_loss(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = _flatten_bhwc(x)
    y = _flatten_bhwc(y)
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    cov_x = x.t().matmul(x) / (x.shape[0] - 1 + eps)
    cov_y = y.t().matmul(y) / (y.shape[0] - 1 + eps)
    return (cov_x - cov_y).pow(2).mean()


def orthogonality_loss(common: torch.Tensor, private: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    b, c, h, w = common.shape
    common = common.reshape(b, c, h * w)
    private = private.reshape(b, c, h * w)
    common = F.normalize(common, dim=-1, eps=eps)
    private = F.normalize(private, dim=-1, eps=eps)
    corr = torch.bmm(common, private.transpose(1, 2))
    return corr.pow(2).mean()


def band_energy_ratio_loss(o_bands: List[torch.Tensor], s_bands_aligned: List[torch.Tensor], eps: float = 1e-6):
    def ratios(bands):
        e = torch.stack([b.abs().mean(dim=(1, 2, 3)) for b in bands], dim=1)
        return e / (e.sum(dim=1, keepdim=True) + eps)

    return (ratios(o_bands) - ratios(s_bands_aligned)).abs().mean()


def residual_regularization(residuals: List[torch.Tensor]) -> torch.Tensor:
    return sum(r.abs().mean() for r in residuals) / max(1, len(residuals))


def first_version_auxiliary_losses(
    aux_list: List[Dict],
    lambda_spa: float = 0.03,
    lambda_freq: float = 0.02,
    lambda_orth: float = 0.01,
    beta_energy: float = 0.1,
    beta_res: float = 0.01,
):
    if aux_list is None or len(aux_list) == 0:
        zero = torch.tensor(0.0)
        return {"loss_aux": zero, "loss_spa": zero, "loss_freq": zero, "loss_orth": zero}

    device = aux_list[0]["common_o"].device
    loss_spa = torch.zeros([], device=device)
    loss_freq = torch.zeros([], device=device)
    loss_orth = torch.zeros([], device=device)
    loss_energy = torch.zeros([], device=device)
    loss_res = torch.zeros([], device=device)
    band_weights = [1.0, 0.5, 0.5, 0.1]

    for aux in aux_list:
        co, cs = aux["common_o"], aux["common_s"]
        po, ps = aux["private_o"], aux["private_s"]
        loss_spa = loss_spa + coral_loss(co, cs)
        loss_orth = loss_orth + orthogonality_loss(co, po) + orthogonality_loss(cs, ps)

        freq = aux["freq"]
        o_bands = freq["o_bands"]
        s_bands_aligned = freq["s_bands_aligned"]
        residuals = freq["residuals"]
        for w, ob, sb in zip(band_weights, o_bands, s_bands_aligned):
            loss_freq = loss_freq + w * coral_loss(ob, sb)
        loss_energy = loss_energy + band_energy_ratio_loss(o_bands, s_bands_aligned)
        loss_res = loss_res + residual_regularization(residuals)

    n = float(len(aux_list))
    loss_spa = loss_spa / n
    loss_freq = loss_freq / n
    loss_orth = loss_orth / n
    loss_energy = loss_energy / n
    loss_res = loss_res / n
    loss_freq_total = loss_freq + beta_energy * loss_energy + beta_res * loss_res
    loss_aux = lambda_spa * loss_spa + lambda_freq * loss_freq_total + lambda_orth * loss_orth

    return {
        "loss_aux": loss_aux,
        "loss_spa": loss_spa,
        "loss_freq": loss_freq,
        "loss_orth": loss_orth,
        "loss_energy": loss_energy,
        "loss_res": loss_res,
    }


def dice_loss_from_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # target: [B,H,W] with 0/1 or [B,1,H,W]
    if target.dim() == 4:
        target = target.squeeze(1)
    prob = torch.softmax(logits, dim=1)[:, 1]
    target = target.float()
    inter = (prob * target).sum(dim=(1, 2))
    union = prob.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
    return (1.0 - (2.0 * inter + eps) / (union + eps)).mean()


def ce_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if target.dim() == 4:
        target = target.squeeze(1)
    target = target.long()
    ce = F.cross_entropy(logits, target)
    dice = dice_loss_from_logits(logits, target)
    return ce + dice


if __name__ == "__main__":
    # Smoke test after SAM2 is installed and ckpt exists.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SAM2LoRASFChangeDetectorV1(
        sam2_cfg="configs/sam2.1/sam2.1_hiera_t.yaml",
        sam2_ckpt="third_party/sam2/checkpoints/sam2.1_hiera_tiny.pt",
        img_size=(512, 512),
        num_classes=2,
        decoder_dim=96,
        lora_r=4,
        device=device,
        sam2_repo_dir="third_party/sam2",
    ).to(device)
    optical = torch.rand(1, 3, 512, 512, device=device)
    sar = torch.rand(1, 3, 512, 512, device=device)
    out = model(optical, sar, return_aux=True)
    print(out["logits"].shape)
    losses = first_version_auxiliary_losses(out["aux"])
    print({k: float(v.detach().cpu()) for k, v in losses.items()})
