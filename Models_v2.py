# Copyright (c) MONAI Consortium
# Modified minimally to add optional big-kernel & symmetric mirroring support.
from __future__ import annotations

import importlib.util
import math
from collections.abc import Sequence
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks import Convolution
from monai.utils import ensure_tuple_rep

# Optional xformers support (same as original)
if importlib.util.find_spec("xformers") is not None:
    import xformers
    import xformers.ops
    has_xformers = True
else:
    xformers = None
    has_xformers = False

__all__ = ["AutoencoderKL"]


# -------------------------
#         Blocks
# -------------------------
class Upsample(nn.Module):
    """
    Convolution-based upsampling layer (kept same behavior).
    """
    def __init__(self, spatial_dims: int, in_channels: int, use_convtranspose: bool) -> None:
        super().__init__()
        if use_convtranspose:
            self.conv = Convolution(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=in_channels,
                strides=2,
                kernel_size=3,
                padding=1,
                conv_only=True,
                is_transposed=True,
            )
        else:
            self.conv = Convolution(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=in_channels,
                strides=1,
                kernel_size=3,
                padding=1,
                conv_only=True,
            )
        self.use_convtranspose = use_convtranspose

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_convtranspose:
            return self.conv(x)
        # bfloat16 safe upsample
        dtype = x.dtype
        if dtype == torch.bfloat16:
            x = x.to(torch.float32)
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if dtype == torch.bfloat16:
            x = x.to(dtype)
        x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    Convolution-based downsampling layer (kept same behavior).
    """
    def __init__(self, spatial_dims: int, in_channels: int) -> None:
        super().__init__()
        self.pad = (0, 1) * spatial_dims
        self.conv = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=in_channels,
            strides=2,
            kernel_size=3,
            padding=0,
            conv_only=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.pad(x, self.pad, mode="constant", value=0.0)
        x = self.conv(x)
        return x


class ResBlock(nn.Module):
    """
    Residual block with two convs; now supports configurable kernel_size (default 3).
    """
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        norm_num_groups: int,
        norm_eps: float,
        out_channels: int,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        pad = kernel_size // 2

        self.norm1 = nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=norm_eps, affine=True)
        self.conv1 = Convolution(
            spatial_dims=spatial_dims,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            strides=1,
            kernel_size=kernel_size,
            padding=pad,
            conv_only=True,
        )
        self.norm2 = nn.GroupNorm(num_groups=norm_num_groups, num_channels=self.out_channels, eps=norm_eps, affine=True)
        self.conv2 = Convolution(
            spatial_dims=spatial_dims,
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            strides=1,
            kernel_size=kernel_size,
            padding=pad,
            conv_only=True,
        )

        if self.in_channels != self.out_channels:
            self.nin_shortcut = Convolution(
                spatial_dims=spatial_dims,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                strides=1,
                kernel_size=1,
                padding=0,
                conv_only=True,
            )
        else:
            self.nin_shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return x + h


class AttentionBlock(nn.Module):
    """
    Attention block with optional xformers (kept compatible).
    """
    def __init__(
        self,
        spatial_dims: int,
        num_channels: int,
        num_head_channels: int | None = None,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        use_flash_attention: bool = False,
    ) -> None:
        super().__init__()
        self.use_flash_attention = use_flash_attention and has_xformers
        self.spatial_dims = spatial_dims
        self.num_channels = num_channels

        self.num_heads = num_channels // num_head_channels if num_head_channels is not None else 1
        self.scale = 1 / math.sqrt(num_channels / self.num_heads)

        self.norm = nn.GroupNorm(num_groups=norm_num_groups, num_channels=num_channels, eps=norm_eps, affine=True)
        self.to_q = nn.Linear(num_channels, num_channels)
        self.to_k = nn.Linear(num_channels, num_channels)
        self.to_v = nn.Linear(num_channels, num_channels)
        self.proj_attn = nn.Linear(num_channels, num_channels)

    def reshape_heads_to_batch_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, self.num_heads, c // self.num_heads)
        x = x.permute(0, 2, 1, 3).reshape(b * self.num_heads, n, c // self.num_heads)
        return x

    def reshape_batch_dim_to_heads(self, x: torch.Tensor) -> torch.Tensor:
        bnh, n, ch = x.shape
        b = bnh // self.num_heads
        x = x.reshape(b, self.num_heads, n, ch)
        x = x.permute(0, 2, 1, 3).reshape(b, n, ch * self.num_heads)
        return x

    def _memory_efficient_attention_xformers(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # xFormers scaled dot-product attention
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        x = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None)
        return x

    def _attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        attn = torch.baddbmm(
            torch.empty(q.shape[0], q.shape[1], k.shape[1], dtype=q.dtype, device=q.device),
            q, k.transpose(-1, -2), beta=0, alpha=self.scale
        )
        attn = attn.softmax(dim=-1)
        x = torch.bmm(attn, v)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        if self.spatial_dims == 2:
            b, c, h, w = x.shape
            x = self.norm(x)
            x = x.view(b, c, h * w).transpose(1, 2)  # [B, N, C]
        else:
            b, c, h, w, d = x.shape
            x = self.norm(x)
            x = x.view(b, c, h * w * d).transpose(1, 2)

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q = self.reshape_heads_to_batch_dim(q)
        k = self.reshape_heads_to_batch_dim(k)
        v = self.reshape_heads_to_batch_dim(v)

        if self.use_flash_attention:
            x = self._memory_efficient_attention_xformers(q, k, v)
        else:
            x = self._attention(q, k, v)

        x = self.reshape_batch_dim_to_heads(x)
        x = self.proj_attn(x)
        if self.spatial_dims == 2:
            x = x.transpose(-1, -2).reshape(b, c, h, w)
        else:
            x = x.transpose(-1, -2).reshape(b, c, h, w, d)

        return x + residual


# -------------------------
#    Helper (big kernels)
# -------------------------
def _as_list(x, n: int) -> List[int]:
    if isinstance(x, Sequence):
        assert len(x) == n, f"Expected length {n}, got {len(x)}."
        return list(x)
    return [x] * n


# -------------------------
#    Encoder / Decoder
# -------------------------
class Encoder(nn.Module):
    """
    Encoder with optional big kernels per level (mirrors original, default identical behavior).
    """
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        num_channels: Sequence[int],
        out_channels: int,
        num_res_blocks: Sequence[int],
        norm_num_groups: int,
        norm_eps: float,
        attention_levels: Sequence[bool],
        with_nonlocal_attn: bool = True,
        use_flash_attention: bool = False,
        # new options
        stem_kernel: int = 3,
        use_big_kernels: bool = False,
        big_kernel_levels: Sequence[bool] | None = None,
        big_kernel_size: int | Sequence[int] = 7,
        big_kernel_blocks: int | Sequence[int] = 1,
    ) -> None:
        super().__init__()
        self.spatial_dims = spatial_dims
        self.num_channels = list(num_channels)
        self.num_res_blocks = list(num_res_blocks)
        self.attention_levels = list(attention_levels)
        L = len(self.num_channels)

        if big_kernel_levels is None:
            big_kernel_levels = [False] * L
        self.use_big_kernels = use_big_kernels
        self.big_kernel_levels = list(big_kernel_levels)
        self.big_kernel_size = _as_list(big_kernel_size, L)
        self.big_kernel_blocks = _as_list(big_kernel_blocks, L)

        # stem
        stem_pad = stem_kernel // 2
        blocks: List[nn.Module] = []
        blocks.append(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=self.num_channels[0],
                strides=1,
                kernel_size=stem_kernel,
                padding=stem_pad,
                conv_only=True,
            )
        )

        # stages
        ch_out = self.num_channels[0]
        for i in range(L):
            ch_in = ch_out
            ch_out = self.num_channels[i]
            is_final = (i == L - 1)

            for rb in range(self.num_res_blocks[i]):
                ksz = 3
                if self.use_big_kernels and self.big_kernel_levels[i] and rb < self.big_kernel_blocks[i]:
                    ksz = self.big_kernel_size[i]
                blocks.append(
                    ResBlock(
                        spatial_dims=spatial_dims,
                        in_channels=ch_in,
                        norm_num_groups=norm_num_groups,
                        norm_eps=norm_eps,
                        out_channels=ch_out,
                        kernel_size=ksz,
                    )
                )
                ch_in = ch_out

                if self.attention_levels[i]:
                    blocks.append(
                        AttentionBlock(
                            spatial_dims=spatial_dims,
                            num_channels=ch_in,
                            norm_num_groups=norm_num_groups,
                            norm_eps=norm_eps,
                            use_flash_attention=use_flash_attention,
                        )
                    )

            if not is_final:
                blocks.append(Downsample(spatial_dims=spatial_dims, in_channels=ch_in))

        # bottleneck (non-local)
        if with_nonlocal_attn:
            blocks.append(
                ResBlock(
                    spatial_dims=spatial_dims,
                    in_channels=self.num_channels[-1],
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    out_channels=self.num_channels[-1],
                    kernel_size=3,
                )
            )
            blocks.append(
                AttentionBlock(
                    spatial_dims=spatial_dims,
                    num_channels=self.num_channels[-1],
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    use_flash_attention=use_flash_attention,
                )
            )
            blocks.append(
                ResBlock(
                    spatial_dims=spatial_dims,
                    in_channels=self.num_channels[-1],
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    out_channels=self.num_channels[-1],
                    kernel_size=3,
                )
            )

        # norm + projection
        blocks.append(nn.GroupNorm(num_groups=norm_num_groups, num_channels=self.num_channels[-1], eps=norm_eps, affine=True))
        blocks.append(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=self.num_channels[-1],
                out_channels=out_channels,
                strides=1,
                kernel_size=3,
                padding=1,
                conv_only=True,
            )
        )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for m in self.blocks:
            x = m(x)
        return x


class Decoder(nn.Module):
    """
    Decoder that mirrors the encoder, with symmetric big-kernel placement.
    """
    def __init__(
        self,
        spatial_dims: int,
        num_channels: Sequence[int],
        in_channels: int,
        out_channels: int,
        num_res_blocks: Sequence[int],
        norm_num_groups: int,
        norm_eps: float,
        attention_levels: Sequence[bool],
        with_nonlocal_attn: bool = True,
        use_flash_attention: bool = False,
        use_convtranspose: bool = False,
        # mirror big-kernel config
        stem_kernel: int = 3,
        use_big_kernels: bool = False,
        big_kernel_levels: Sequence[bool] | None = None,
        big_kernel_size: int | Sequence[int] = 7,
        big_kernel_blocks: int | Sequence[int] = 1,
    ) -> None:
        super().__init__()
        self.spatial_dims = spatial_dims
        enc_channels = list(num_channels)
        enc_res_blocks = list(num_res_blocks)
        enc_attn = list(attention_levels)
        L = len(enc_channels)

        if big_kernel_levels is None:
            big_kernel_levels = [False] * L
        self.use_big_kernels = use_big_kernels
        self.big_kernel_levels = list(big_kernel_levels)
        self.big_kernel_size = _as_list(big_kernel_size, L)
        self.big_kernel_blocks = _as_list(big_kernel_blocks, L)

        # mirror lists
        dec_channels = list(reversed(enc_channels))
        dec_res_blocks = list(reversed(enc_res_blocks))
        dec_attn = list(reversed(enc_attn))
        dec_big_levels = list(reversed(self.big_kernel_levels))
        dec_big_sizes = list(reversed(self.big_kernel_size))
        dec_big_blocks = list(reversed(self.big_kernel_blocks))

        blocks: List[nn.Module] = []
        # head
        blocks.append(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=dec_channels[0],
                strides=1,
                kernel_size=stem_kernel,
                padding=stem_kernel // 2,
                conv_only=True,
            )
        )

        # bottleneck (non-local)
        if with_nonlocal_attn:
            blocks.append(
                ResBlock(
                    spatial_dims=spatial_dims,
                    in_channels=dec_channels[0],
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    out_channels=dec_channels[0],
                    kernel_size=3,
                )
            )
            blocks.append(
                AttentionBlock(
                    spatial_dims=spatial_dims,
                    num_channels=dec_channels[0],
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    use_flash_attention=use_flash_attention,
                )
            )
            blocks.append(
                ResBlock(
                    spatial_dims=spatial_dims,
                    in_channels=dec_channels[0],
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    out_channels=dec_channels[0],
                    kernel_size=3,
                )
            )

        ch_out = dec_channels[0]
        for i in range(L):
            ch_in = ch_out
            ch_out = dec_channels[i]
            is_final = (i == L - 1)

            for rb in range(dec_res_blocks[i]):
                ksz = 3
                if self.use_big_kernels and dec_big_levels[i] and rb < dec_big_blocks[i]:
                    ksz = dec_big_sizes[i]
                blocks.append(
                    ResBlock(
                        spatial_dims=spatial_dims,
                        in_channels=ch_in,
                        norm_num_groups=norm_num_groups,
                        norm_eps=norm_eps,
                        out_channels=ch_out,
                        kernel_size=ksz,
                    )
                )
                ch_in = ch_out

                if dec_attn[i]:
                    blocks.append(
                        AttentionBlock(
                            spatial_dims=spatial_dims,
                            num_channels=ch_in,
                            norm_num_groups=norm_num_groups,
                            norm_eps=norm_eps,
                            use_flash_attention=use_flash_attention,
                        )
                    )

            if not is_final:
                blocks.append(Upsample(spatial_dims=spatial_dims, in_channels=ch_in, use_convtranspose=use_convtranspose))

        blocks.append(nn.GroupNorm(num_groups=norm_num_groups, num_channels=ch_in, eps=norm_eps, affine=True))
        blocks.append(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=ch_in,
                out_channels=out_channels,
                strides=1,
                kernel_size=3,
                padding=1,
                conv_only=True,
            )
        )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for m in self.blocks:
            x = m(x)
        return torch.tanh(x)  # keep same output nonlinearity

class Decoder_v2(nn.Module):
    """
    Decoder that mirrors the encoder, with optional multi-layer style injection.

    - 保持原有初始化参数不变；
    - 额外增加:
        inject_style: 是否启用风格注入逻辑（默认 False）
        style_in_channels: 风格特征输入通道数（style_sampler 输出通道数）
        style_latent_channels: 内部风格通道数，不填则默认 = dec_channels[0]
    """
    def __init__(
        self,
        spatial_dims: int,
        num_channels: Sequence[int],
        in_channels: int,
        out_channels: int,
        num_res_blocks: Sequence[int],
        norm_num_groups: int,
        norm_eps: float,
        attention_levels: Sequence[bool],
        with_nonlocal_attn: bool = True,
        use_flash_attention: bool = False,
        use_convtranspose: bool = False,
        # mirror big-kernel config
        stem_kernel: int = 3,
        use_big_kernels: bool = False,
        big_kernel_levels: Sequence[bool] | None = None,
        big_kernel_size: int | Sequence[int] = 7,
        big_kernel_blocks: int | Sequence[int] = 1,
        # ---------- 新增：风格注入相关 ----------
        inject_style: bool = False,
        style_in_channels: int | None = None,
        style_latent_channels: int | None = None,
    ) -> None:
        super().__init__()
        self.spatial_dims = spatial_dims
        enc_channels = list(num_channels)
        enc_res_blocks = list(num_res_blocks)
        enc_attn = list(attention_levels)
        L = len(enc_channels)

        if big_kernel_levels is None:
            big_kernel_levels = [False] * L
        self.use_big_kernels = use_big_kernels
        self.big_kernel_levels = list(big_kernel_levels)
        self.big_kernel_size = _as_list(big_kernel_size, L)
        self.big_kernel_blocks = _as_list(big_kernel_blocks, L)

        # mirror lists
        dec_channels = list(reversed(enc_channels))
        dec_res_blocks = list(reversed(enc_res_blocks))
        dec_attn = list(reversed(enc_attn))
        dec_big_levels = list(reversed(self.big_kernel_levels))
        dec_big_sizes = list(reversed(self.big_kernel_size))
        dec_big_blocks = list(reversed(self.big_kernel_blocks))

        # ---------- 风格注入模块（可选） ----------
        self.inject_style = inject_style and (style_in_channels is not None)
        self.style_adapters = nn.ModuleList()
        self.style_proj = None
        self.style_latent_channels = None

        if self.inject_style:
            first_dec_ch = dec_channels[0]
            if style_latent_channels is None:
                style_latent_channels = min(style_in_channels, first_dec_ch)
            self.style_latent_channels = style_latent_channels

            # 把输入的 style 特征映射到内部统一通道数
            self.style_proj = Convolution(
                spatial_dims=spatial_dims,
                in_channels=style_in_channels,
                out_channels=style_latent_channels,
                strides=1,
                kernel_size=1,
                padding=0,
                conv_only=True,
            )

        blocks: List[nn.Module] = []

        # ---------- head ----------
        blocks.append(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=dec_channels[0],
                strides=1,
                kernel_size=stem_kernel,
                padding=stem_kernel // 2,
                conv_only=True,
            )
        )

        # ---------- bottleneck (non-local) ----------
        if with_nonlocal_attn:
            # ResBlock 1
            blocks.append(
                ResBlock(
                    spatial_dims=spatial_dims,
                    in_channels=dec_channels[0],
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    out_channels=dec_channels[0],
                    kernel_size=3,
                )
            )
            if self.inject_style:
                self.style_adapters.append(
                    Convolution(
                        spatial_dims=spatial_dims,
                        in_channels=self.style_latent_channels,
                        out_channels=dec_channels[0],
                        strides=1,
                        kernel_size=1,
                        padding=0,
                        conv_only=True,
                    )
                )

            # Attention
            blocks.append(
                AttentionBlock(
                    spatial_dims=spatial_dims,
                    num_channels=dec_channels[0],
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    use_flash_attention=use_flash_attention,
                )
            )

            # ResBlock 2
            blocks.append(
                ResBlock(
                    spatial_dims=spatial_dims,
                    in_channels=dec_channels[0],
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    out_channels=dec_channels[0],
                    kernel_size=3,
                )
            )
            if self.inject_style:
                self.style_adapters.append(
                    Convolution(
                        spatial_dims=spatial_dims,
                        in_channels=self.style_latent_channels,
                        out_channels=dec_channels[0],
                        strides=1,
                        kernel_size=1,
                        padding=0,
                        conv_only=True,
                    )
                )

        # ---------- 主体层级 ----------
        ch_out = dec_channels[0]
        for i in range(L):
            ch_in = ch_out
            ch_out = dec_channels[i]
            is_final = (i == L - 1)

            for rb in range(dec_res_blocks[i]):
                ksz = 3
                if self.use_big_kernels and dec_big_levels[i] and rb < dec_big_blocks[i]:
                    ksz = dec_big_sizes[i]
                # ResBlock
                blocks.append(
                    ResBlock(
                        spatial_dims=spatial_dims,
                        in_channels=ch_in,
                        norm_num_groups=norm_num_groups,
                        norm_eps=norm_eps,
                        out_channels=ch_out,
                        kernel_size=ksz,
                    )
                )
                if self.inject_style:
                    self.style_adapters.append(
                        Convolution(
                            spatial_dims=spatial_dims,
                            in_channels=self.style_latent_channels,
                            out_channels=ch_in,
                            strides=1,
                            kernel_size=1,
                            padding=0,
                            conv_only=True,
                        )
                    )
                ch_in = ch_out

                if dec_attn[i]:
                    blocks.append(
                        AttentionBlock(
                            spatial_dims=spatial_dims,
                            num_channels=ch_in,
                            norm_num_groups=norm_num_groups,
                            norm_eps=norm_eps,
                            use_flash_attention=use_flash_attention,
                        )
                    )

            if not is_final:
                blocks.append(Upsample(
                    spatial_dims=spatial_dims,
                    in_channels=ch_in,
                    use_convtranspose=use_convtranspose,
                ))

        # ---------- 尾部 norm + conv ----------
        blocks.append(nn.GroupNorm(num_groups=norm_num_groups, num_channels=ch_in, eps=norm_eps, affine=True))
        blocks.append(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=ch_in,
                out_channels=out_channels,
                strides=1,
                kernel_size=3,
                padding=1,
                conv_only=True,
            )
        )
        self.blocks = nn.ModuleList(blocks)

        # 初始化所有 style_adapters 为 0：这样即使传入 style，
        # 初始行为也尽量接近「没有风格注入」的 decoder，帮助稳定训练。
        if self.inject_style:
            for m in self.style_adapters:
                nn.init.zeros_(m.conv.weight)
                if m.conv.bias is not None:
                    nn.init.zeros_(m.conv.bias)

    def forward(self, x: torch.Tensor, style: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: 主干输入（可以是原分布采样点 z，也可以是 content 特征）
        style: 风格特征 [B, C_style, H, W]，可为 None
        """
        # 准备风格分支
        s = None
        if self.inject_style and (style is not None):
            s = self.style_proj(style)
        style_idx = 0

        for m in self.blocks:
            if isinstance(m, ResBlock):
                # 在每个 ResBlock 前注入风格
                if s is not None:
                    # 确保空间尺寸一致（如果 Upsample 后略有差异，就补一个 resize）
                    if s.shape[2:] != x.shape[2:]:
                        s = F.interpolate(s, size=x.shape[2:], mode="nearest")
                    x = x + self.style_adapters[style_idx](s)
                    style_idx += 1
                x = m(x)

            elif isinstance(m, Upsample):
                x = m(x)
                # 风格分支同步上采样
                if s is not None:
                    s = F.interpolate(s, scale_factor=2.0, mode="nearest")

            else:
                # Attention / Norm / Final Conv 直接应用
                x = m(x)

        return torch.tanh(x)


# -------------------------
#      AutoencoderKL
# -------------------------
class AutoencoderKL(nn.Module):
    """
    Autoencoder with KL-regularized latent space. Extended with optional big-kernel support (default off).
    """
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int = 1,
        out_channels: int = 1,
        num_res_blocks: Sequence[int] | int = (2, 2, 2, 2),
        num_channels: Sequence[int] = (32, 64, 64, 64),
        attention_levels: Sequence[bool] = (False, False, True, True),
        latent_channels: int = 3,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        with_encoder_nonlocal_attn: bool = True,
        with_decoder_nonlocal_attn: bool = True,
        use_flash_attention: bool = False,
        use_checkpointing: bool = False,
        use_convtranspose: bool = False,
        # ---- new big-kernel options ----
        stem_kernel: int = 3,
        use_big_kernels: bool = False,
        big_kernel_levels: Sequence[bool] | None = None,
        big_kernel_size: int | Sequence[int] = 7,
        big_kernel_blocks: int | Sequence[int] = 1,
        sty_ch: int | None = None,

    ) -> None:
        super().__init__()

        if any((out_channel % norm_num_groups) != 0 for out_channel in num_channels):
            raise ValueError("AutoencoderKL expects all num_channels being multiple of norm_num_groups")
        if len(num_channels) != len(attention_levels):
            raise ValueError("AutoencoderKL expects num_channels being same size of attention_levels")
        if isinstance(num_res_blocks, int):
            num_res_blocks = ensure_tuple_rep(num_res_blocks, len(num_channels))
        if len(num_res_blocks) != len(num_channels):
            raise ValueError("num_res_blocks length must match num_channels length")
        if use_flash_attention and not torch.cuda.is_available():
            # 保持原版安全提示
            raise ValueError("Flash attention requires CUDA. Set use_flash_attention=False on CPU.")

        # Encoder
        self.encoder = Encoder(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            num_channels=num_channels,
            out_channels=latent_channels,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            attention_levels=attention_levels,
            with_nonlocal_attn=with_encoder_nonlocal_attn,
            use_flash_attention=use_flash_attention,
            stem_kernel=stem_kernel,
            use_big_kernels=use_big_kernels,
            big_kernel_levels=big_kernel_levels,
            big_kernel_size=big_kernel_size,
            big_kernel_blocks=big_kernel_blocks,
        )

        # Two symmetric decoders (CT / MR)
        self.decoder = Decoder_v2(
            spatial_dims=spatial_dims,
            num_channels=num_channels,
            in_channels=latent_channels,
            out_channels=out_channels,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            attention_levels=attention_levels,
            with_nonlocal_attn=with_decoder_nonlocal_attn,
            use_flash_attention=use_flash_attention,
            use_convtranspose=use_convtranspose,
            stem_kernel=stem_kernel,
            use_big_kernels=use_big_kernels,
            big_kernel_levels=big_kernel_levels,
            big_kernel_size=big_kernel_size,
            big_kernel_blocks=big_kernel_blocks,
            inject_style=True,
            style_in_channels=sty_ch,
            style_latent_channels=None,
        )
        self.decoder_2 = Decoder_v2(
            spatial_dims=spatial_dims,
            num_channels=num_channels,
            in_channels=latent_channels,
            out_channels=out_channels,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            attention_levels=attention_levels,
            with_nonlocal_attn=with_decoder_nonlocal_attn,
            use_flash_attention=use_flash_attention,
            use_convtranspose=use_convtranspose,
            stem_kernel=stem_kernel,
            use_big_kernels=use_big_kernels,
            big_kernel_levels=big_kernel_levels,
            big_kernel_size=big_kernel_size,
            big_kernel_blocks=big_kernel_blocks,
            inject_style=True,
            style_in_channels=sty_ch,
            style_latent_channels=None,
        )

        # latent heads (same logic as original)
        self.quant_conv_mu = Convolution(
            spatial_dims=spatial_dims, in_channels=latent_channels, out_channels=latent_channels,
            strides=1, kernel_size=1, padding=0, conv_only=True,
        )
        self.quant_conv_logvar = Convolution(
            spatial_dims=spatial_dims, in_channels=latent_channels, out_channels=latent_channels,
            strides=1, kernel_size=1, padding=0, conv_only=True,
        )
        self.post_quant_conv = Convolution(
            spatial_dims=spatial_dims, in_channels=latent_channels, out_channels=latent_channels,
            strides=1, kernel_size=1, padding=0, conv_only=True,
        )
        self.post_quant_conv_2 = Convolution(
            spatial_dims=spatial_dims, in_channels=latent_channels, out_channels=latent_channels,
            strides=1, kernel_size=1, padding=0, conv_only=True,
        )

        self.latent_channels = latent_channels
        self.use_checkpointing = use_checkpointing

    # --------- core ops ---------
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.use_checkpointing:
            h = torch.utils.checkpoint.checkpoint(self.encoder, x, use_reentrant=False)
        else:
            h = self.encoder(x)
        mu = self.quant_conv_mu(h)
        logvar = self.quant_conv_logvar(h)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        sigma = torch.exp(logvar / 2)
        return mu, sigma

    @staticmethod
    def flip_distribution(mu: torch.Tensor, sigma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return -mu, sigma

    @staticmethod
    def sampling(mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        eps = torch.randn_like(sigma)
        return mu + eps * sigma

    # def decode_A(self, z: torch.Tensor) -> torch.Tensor:
    #     z = self.post_quant_conv(z)
    #     if self.use_checkpointing:
    #         return torch.utils.checkpoint.checkpoint(self.decoder, z, use_reentrant=False)
    #     return self.decoder(z)

    # def decode_B(self, z: torch.Tensor) -> torch.Tensor:
    #     z = self.post_quant_conv_2(z)
    #     if self.use_checkpointing:
    #         return torch.utils.checkpoint.checkpoint(self.decoder_2, z, use_reentrant=False)
    #     return self.decoder_2(z)

    # ---------- 解码入口 1：原分布采样点 ----------
    def decode_A_raw(self, z: torch.Tensor) -> torch.Tensor:
        z = self.post_quant_conv(z)
        # z = self.post_quant_norm(z)
        # z = F.silu(z)
        return self.decoder(z, style=None)

    def decode_B_raw(self, z: torch.Tensor) -> torch.Tensor:
        z = self.post_quant_conv_2(z)
        # z = self.post_quant_norm_2(z)
        # z = F.silu(z)
        return self.decoder_2(z, style=None)

    # ---------- 解码入口 2：内容 + 风格 ----------
    def decode_A_cs(self, cont: torch.Tensor, sty: torch.Tensor) -> torch.Tensor:
        z = self.post_quant_conv(cont)
        # z = self.post_quant_norm(z)
        # z = F.silu(z)
        return self.decoder(z, style=sty)

    def decode_B_cs(self, cont: torch.Tensor, sty: torch.Tensor) -> torch.Tensor:
        z = self.post_quant_conv_2(cont)
        # z = self.post_quant_norm_2(z)
        # z = F.silu(z)
        return self.decoder_2(z, style=sty)



    # --------- convenience I/O ---------
    # def val_reconstruct(self, x: torch.Tensor, label: str) -> tuple[torch.Tensor, torch.Tensor]:
    #     mu, sigma = self.encode(x)
    #     if label == 'ct':
    #         rec_ct = self.decode_ct(mu)
    #         mu_flip, _ = self.flip_distribution(mu, sigma)
    #         rec_mr = self.decode_mr(mu_flip)
    #     elif label == 'mr':
    #         rec_mr = self.decode_mr(mu)
    #         mu_flip, _ = self.flip_distribution(mu, sigma)
    #         rec_ct = self.decode_ct(mu_flip)
    #     else:
    #         raise ValueError("label must be 'ct' or 'mr'")
    #     return rec_ct, rec_mr

    def forward(self, x: torch.Tensor, label: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, sigma = self.encode(x)
        if label == 'A':
            z_A = self.sampling(mu, sigma)
            rec_A = self.decode_A(z_A)
            mu_flip, sig_flip = self.flip_distribution(mu, sigma)
            z_A2B = self.sampling(mu_flip, sig_flip)
            rec_B = self.decode_mr(z_A2B)
        elif label == 'B':
            z_B = self.sampling(mu, sigma)
            rec_B = self.decode_B(z_B)
            mu_flip, sig_flip = self.flip_distribution(mu, sigma)
            z_B2A = self.sampling(mu_flip, sig_flip)
            rec_A = self.decode_ct(z_B2A)
        else:
            raise ValueError("label must be 'ct' or 'mr'")
        return rec_A, rec_B, mu  # mu 作为可用的 latent 代表（你也可返回 sigma）


# -------------------------
#        Quick test
# -------------------------
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AutoencoderKL(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        latent_channels=32,
        num_channels=[64, 128, 256],
        num_res_blocks=2,
        norm_num_groups=32,
        norm_eps=1e-6,
        attention_levels=[False, False, False],
        with_encoder_nonlocal_attn=True,
        with_decoder_nonlocal_attn=True,
        use_convtranspose=True,
        use_checkpointing=False,
        use_flash_attention=True,  # True 需 CUDA + xformers
        # ------ big-kernel (optional) ------
        stem_kernel=7,
        use_big_kernels=True,
        big_kernel_levels=[True, True, False],   # 前两层用大核
        big_kernel_size=[7, 5, 3],               # 对应每层大核尺寸
        big_kernel_blocks=[1, 1, 0],             # 每层前1个 ResBlock 用大核
    ).to(device)

    with torch.no_grad():
        x = torch.randn(24, 1, 256, 256, device=device)
        y_ct, y_mr, mu = model(x, 'ct')
        print("mu:", tuple(mu.shape), "ct:", tuple(y_ct.shape), "mr:", tuple(y_mr.shape))