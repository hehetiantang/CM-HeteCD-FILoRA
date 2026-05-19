# filora/cm_croma.py

import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class LoRALinear(nn.Layer):
    """
    Paddle implementation of LoRA Linear:
        y = xW + scale * xAB

    The base weight W can be frozen. Only A and B are trained.
    """

    def __init__(self, in_features, out_features, rank=8, alpha=16, dropout=0.0, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank

        self.weight = self.create_parameter(
            shape=[in_features, out_features],
            default_initializer=nn.initializer.XavierUniform()
        )
        self.weight.stop_gradient = True

        if bias:
            self.bias = self.create_parameter(
                shape=[out_features],
                is_bias=True,
                default_initializer=nn.initializer.Constant(0.0)
            )
            self.bias.stop_gradient = True
        else:
            self.bias = None

        self.lora_a = self.create_parameter(
            shape=[in_features, rank],
            default_initializer=nn.initializer.Normal(std=0.02)
        )
        self.lora_b = self.create_parameter(
            shape=[rank, out_features],
            default_initializer=nn.initializer.Constant(0.0)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        base = paddle.matmul(x, self.weight)
        if self.bias is not None:
            base = base + self.bias

        lora = paddle.matmul(self.dropout(x), self.lora_a)
        lora = paddle.matmul(lora, self.lora_b) * self.scale
        return base + lora


class ModalityAdapter(nn.Layer):
    """
    Lightweight modality adapter.

    For SAR and Optical, the low-level statistics are very different.
    This adapter first normalizes modality-specific local patterns before patch embedding.
    """

    def __init__(self, in_chans):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2D(in_chans, in_chans, kernel_size=3, padding=1, groups=in_chans),
            nn.BatchNorm2D(in_chans),
            nn.GELU(),
            nn.Conv2D(in_chans, in_chans, kernel_size=1),
            nn.BatchNorm2D(in_chans),
            nn.GELU(),
        )

    def forward(self, x):
        return x + self.net(x)


class PatchEmbed(nn.Layer):
    def __init__(self, in_chans, embed_dim=256, patch_size=8):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2D(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)  # [B, C, H/P, W/P]
        h, w = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose([0, 2, 1])  # [B, N, C]
        return x, h, w


class LoRAAttention(nn.Layer):
    def __init__(self, dim, num_heads=8, rank=8, dropout=0.0):
        super().__init__()
        assert dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.qkv = LoRALinear(dim, dim * 3, rank=rank, alpha=rank * 2, dropout=dropout, bias=False)
        self.proj = LoRALinear(dim, dim, rank=rank, alpha=rank * 2, dropout=dropout, bias=True)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x):
        b, n, c = x.shape
        x_norm = self.norm(x)

        qkv = self.qkv(x_norm)
        qkv = qkv.reshape([b, n, 3, self.num_heads, self.head_dim])
        qkv = qkv.transpose([2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = paddle.matmul(q, k, transpose_y=True) * self.scale
        attn = F.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        out = paddle.matmul(attn, v)
        out = out.transpose([0, 2, 1, 3]).reshape([b, n, c])
        out = self.proj(out)
        return out


class FFN(nn.Layer):
    def __init__(self, dim, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.norm = nn.LayerNorm(dim)
        self.fc1 = LoRALinear(dim, hidden_dim, rank=8, alpha=16, dropout=dropout)
        self.fc2 = LoRALinear(hidden_dim, dim, rank=8, alpha=16, dropout=dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        y = self.norm(x)
        y = self.fc1(y)
        y = F.gelu(y)
        y = self.drop(y)
        y = self.fc2(y)
        return y


class LoRATransformerBlock(nn.Layer):
    def __init__(self, dim=256, num_heads=8, rank=8, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.attn = LoRAAttention(dim, num_heads=num_heads, rank=rank, dropout=dropout)
        self.ffn = FFN(dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.ffn(x)
        return x


class CrossModalAlignment(nn.Layer):
    """
    Cross-modal alignment module.

    It performs bidirectional feature interaction:
        SAR attends to Optical
        Optical attends to SAR
    """

    def __init__(self, dim=256, num_heads=8, rank=8, dropout=0.0):
        super().__init__()
        assert dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm_a = nn.LayerNorm(dim)
        self.norm_b = nn.LayerNorm(dim)

        self.q_a = LoRALinear(dim, dim, rank=rank, alpha=rank * 2, dropout=dropout, bias=False)
        self.k_b = LoRALinear(dim, dim, rank=rank, alpha=rank * 2, dropout=dropout, bias=False)
        self.v_b = LoRALinear(dim, dim, rank=rank, alpha=rank * 2, dropout=dropout, bias=False)

        self.q_b = LoRALinear(dim, dim, rank=rank, alpha=rank * 2, dropout=dropout, bias=False)
        self.k_a = LoRALinear(dim, dim, rank=rank, alpha=rank * 2, dropout=dropout, bias=False)
        self.v_a = LoRALinear(dim, dim, rank=rank, alpha=rank * 2, dropout=dropout, bias=False)

        self.proj_a = LoRALinear(dim, dim, rank=rank, alpha=rank * 2, dropout=dropout)
        self.proj_b = LoRALinear(dim, dim, rank=rank, alpha=rank * 2, dropout=dropout)

    def _reshape_heads(self, x):
        b, n, c = x.shape
        x = x.reshape([b, n, self.num_heads, self.head_dim])
        x = x.transpose([0, 2, 1, 3])
        return x

    def _cross_attn(self, q, k, v):
        attn = paddle.matmul(q, k, transpose_y=True) * self.scale
        attn = F.softmax(attn, axis=-1)
        out = paddle.matmul(attn, v)
        out = out.transpose([0, 2, 1, 3])
        b, n, _, _ = out.shape
        out = out.reshape([b, n, self.num_heads * self.head_dim])
        return out

    def forward(self, fa, fb):
        fa_norm = self.norm_a(fa)
        fb_norm = self.norm_b(fb)

        qa = self._reshape_heads(self.q_a(fa_norm))
        kb = self._reshape_heads(self.k_b(fb_norm))
        vb = self._reshape_heads(self.v_b(fb_norm))

        qb = self._reshape_heads(self.q_b(fb_norm))
        ka = self._reshape_heads(self.k_a(fa_norm))
        va = self._reshape_heads(self.v_a(fa_norm))

        fa2 = self._cross_attn(qa, kb, vb)
        fb2 = self._cross_attn(qb, ka, va)

        fa = fa + self.proj_a(fa2)
        fb = fb + self.proj_b(fb2)

        return fa, fb


class CrossModalFoundationEncoder(nn.Layer):
    """
    Paddle CROMA-style encoder for FILoRA replacement.

    Input:
        xa: SAR image, usually [B, 2, H, W] or [B, 1, H, W]
        xb: Optical image, usually [B, 3, H, W] or [B, 12, H, W]

    Output:
        fa, fb: feature maps [B, embed_dim, H/patch, W/patch]
    """

    def __init__(
        self,
        img_size=256,
        in_chans_a=2,
        in_chans_b=3,
        embed_dim=256,
        depth=4,
        num_heads=8,
        patch_size=8,
        rank=8,
        dropout=0.0,
        use_shared_prior=True,
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.use_shared_prior = use_shared_prior

        self.a_adapter = ModalityAdapter(in_chans_a)
        self.b_adapter = ModalityAdapter(in_chans_b)

        self.a_patch = PatchEmbed(in_chans_a, embed_dim=embed_dim, patch_size=patch_size)
        self.b_patch = PatchEmbed(in_chans_b, embed_dim=embed_dim, patch_size=patch_size)

        self.a_blocks = nn.LayerList([
            LoRATransformerBlock(embed_dim, num_heads=num_heads, rank=rank, dropout=dropout)
            for _ in range(depth)
        ])
        self.b_blocks = nn.LayerList([
            LoRATransformerBlock(embed_dim, num_heads=num_heads, rank=rank, dropout=dropout)
            for _ in range(depth)
        ])

        self.shared_prior = self.create_parameter(
            shape=[embed_dim, embed_dim],
            default_initializer=nn.initializer.XavierUniform()
        )

        self.cma = CrossModalAlignment(
            dim=embed_dim,
            num_heads=num_heads,
            rank=rank,
            dropout=dropout
        )

        self.norm_a = nn.LayerNorm(embed_dim)
        self.norm_b = nn.LayerNorm(embed_dim)

    def _apply_shared_prior(self, x):
        if not self.use_shared_prior:
            return x
        return x + 0.1 * paddle.matmul(x, self.shared_prior)

    def _tokens_to_map(self, x, h, w):
        b, n, c = x.shape
        x = x.transpose([0, 2, 1]).reshape([b, c, h, w])
        return x

    def forward(self, xa, xb):
        xa = self.a_adapter(xa)
        xb = self.b_adapter(xb)

        fa, ha, wa = self.a_patch(xa)
        fb, hb, wb = self.b_patch(xb)

        for blk in self.a_blocks:
            fa = blk(fa)
            fa = self._apply_shared_prior(fa)

        for blk in self.b_blocks:
            fb = blk(fb)
            fb = self._apply_shared_prior(fb)

        fa, fb = self.cma(fa, fb)

        fa = self.norm_a(fa)
        fb = self.norm_b(fb)

        fa = self._tokens_to_map(fa, ha, wa)
        fb = self._tokens_to_map(fb, hb, wb)

        return fa, fb
