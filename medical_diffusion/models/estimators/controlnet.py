import torch
import torch.nn as nn
from medical_diffusion.models.embedders import TimeEmbbeding
from medical_diffusion.models.utils.conv_blocks import (
    BasicBlock,
    UnetResBlock,
    save_add,
    BasicDown,
    SequentialEmb,
)
from medical_diffusion.models.utils.attention_blocks import zero_module


class ControlNet(nn.Module):
    """
    ControlNet for UNet(hid_chs=[256,256,512,1024]).

    输出 residuals:
        - res32: [B, 256, 32, 32]
        - res16: [B, 512, 16, 16]
        - res8:  [B, 1024, 8, 8]

    设计要点：
        * 输入为 256x256 的结构图 (vessel / disc / cup)
        * 通过 3 个 stride=2 的 BasicDown 将 256x256 → 32x32
        * 每个尺度使用 2 个 ResBlock
        * 每个尺度用 zero-init 的 1×1 conv 生成 residual，初始时对冻结 UNet 几乎无扰动
    """

    def __init__(
        self,
        in_ch=3,
        spatial_dims=2,
        hid_chs=[256, 256, 512, 1024],
        kernel_sizes=[1, 3, 3, 3],
        strides=[1, 2, 2, 2],
        time_embedder=TimeEmbbeding,
        time_embedder_kwargs={},
        cond_embedder=None,
        cond_embedder_kwargs={},
    ):
        super().__init__()

        # 这里我们只用 hid_chs 中的 0/2/3 三个值作为三个尺度的通道数：
        #   32x32 → 256 通道
        #   16x16 → 512 通道
        #   8x8  → 1024 通道
        ch_32 = hid_chs[0]   # 256
        ch_16 = hid_chs[2]   # 512
        ch_8  = hid_chs[3]   # 1024

        self.ch_32 = ch_32
        self.ch_16 = ch_16
        self.ch_8 = ch_8

        # -------- Time & Condition --------
        self.time_embedder = time_embedder(**time_embedder_kwargs)
        time_emb_dim = self.time_embedder.emb_dim

        if cond_embedder is not None:
            self.cond_embedder = cond_embedder(**cond_embedder_kwargs)
            cond_emb_dim = self.cond_embedder.emb_dim
        else:
            self.cond_embedder = None
            cond_emb_dim = None

        emb_dim = time_emb_dim  # 与 UNet 一致，仅通过时间步（和可选标签）调制

        # 和原 Diffusion UNet 保持一致的激活与归一化
        act_name = ("SWISH", {})
        norm_name = ("GROUP", {"num_groups": 32, "affine": True})

        # ========== High-res path: 256×256 → 32×32 ==========
        # 先做一个基本卷积，将 control hint 编码到 ch_32 通道
        self.input_hint = BasicBlock(
            spatial_dims,
            in_ch,
            ch_32,
            kernel_size=3,
            stride=1,
        )

        # 用 3 个 stride=2 的 BasicDown 替代原来的一步 8× bilinear 下采样
        # 256x256 → 128x128
        self.down_256_128 = BasicDown(
            spatial_dims=spatial_dims,
            in_channels=ch_32,
            out_channels=ch_32,
            kernel_size=3,
            stride=2,
        )
        # 128x128 → 64x64
        self.down_128_64 = BasicDown(
            spatial_dims=spatial_dims,
            in_channels=ch_32,
            out_channels=ch_32,
            kernel_size=3,
            stride=2,
        )
        # 64x64 → 32x32
        self.down_64_32 = BasicDown(
            spatial_dims=spatial_dims,
            in_channels=ch_32,
            out_channels=ch_32,
            kernel_size=3,
            stride=2,
        )

        # ========== Stage 0: 32×32, 256 ch ==========
        # 两个 ResBlock 提高 capacity
        self.block_32 = SequentialEmb(
            UnetResBlock(
                spatial_dims=spatial_dims,
                in_channels=ch_32,
                out_channels=ch_32,
                kernel_size=3,
                stride=1,
                norm_name=norm_name,
                act_name=act_name,
                emb_channels=emb_dim,
            ),
            UnetResBlock(
                spatial_dims=spatial_dims,
                in_channels=ch_32,
                out_channels=ch_32,
                kernel_size=3,
                stride=1,
                norm_name=norm_name,
                act_name=act_name,
                emb_channels=emb_dim,
            ),
        )
        # zero-init 的 hint conv，用来生成 res32
        self.hint_32 = zero_module(
            nn.Conv2d(ch_32, ch_32, kernel_size=1)
        )

        # ========== Downsample to 16×16, 512 ch ==========
        # 32x32, 256 → 16x16, 512
        self.down_32_16 = BasicDown(
            spatial_dims=spatial_dims,
            in_channels=ch_32,
            out_channels=ch_16,
            kernel_size=3,
            stride=strides[1],  # 通常为 2
        )

        self.block_16 = SequentialEmb(
            UnetResBlock(
                spatial_dims=spatial_dims,
                in_channels=ch_16,
                out_channels=ch_16,
                kernel_size=3,
                stride=1,
                norm_name=norm_name,
                act_name=act_name,
                emb_channels=emb_dim,
            ),
            UnetResBlock(
                spatial_dims=spatial_dims,
                in_channels=ch_16,
                out_channels=ch_16,
                kernel_size=3,
                stride=1,
                norm_name=norm_name,
                act_name=act_name,
                emb_channels=emb_dim,
            ),
        )
        self.hint_16 = zero_module(
            nn.Conv2d(ch_16, ch_16, kernel_size=1)
        )

        # ========== Downsample to 8×8, 1024 ch ==========
        # 16x16, 512 → 8x8, 1024
        self.down_16_8 = BasicDown(
            spatial_dims=spatial_dims,
            in_channels=ch_16,
            out_channels=ch_8,
            kernel_size=3,
            stride=strides[2],  # 通常为 2
        )

        self.block_8 = SequentialEmb(
            UnetResBlock(
                spatial_dims=spatial_dims,
                in_channels=ch_8,
                out_channels=ch_8,
                kernel_size=3,
                stride=1,
                norm_name=norm_name,
                act_name=act_name,
                emb_channels=emb_dim,
            ),
            UnetResBlock(
                spatial_dims=spatial_dims,
                in_channels=ch_8,
                out_channels=ch_8,
                kernel_size=3,
                stride=1,
                norm_name=norm_name,
                act_name=act_name,
                emb_channels=emb_dim,
            ),
        )
        self.hint_8 = zero_module(
            nn.Conv2d(ch_8, ch_8, kernel_size=1)
        )

    def forward(self, control, t=None, condition=None):
        """
        control: [B,3,256,256]
        return 3 residuals:
            res32: [B,256,32,32]
            res16: [B,512,16,16]
            res8:  [B,1024,8,8]
        """

        # ---- embeddings ----
        if t is not None:
            t_emb = self.time_embedder(t)
        else:
            t_emb = None

        if self.cond_embedder is not None and condition is not None:
            c_emb = self.cond_embedder(condition)
        else:
            c_emb = None

        emb = save_add(t_emb, c_emb)

        # ===== 256×256 → 32×32 =====
        x = self.input_hint(control)   # [B,256,256,256]
        x = self.down_256_128(x)       # → [B,256,128,128]
        x = self.down_128_64(x)        # → [B,256, 64, 64]
        x = self.down_64_32(x)         # → [B,256, 32, 32]

        x = self.block_32(x, emb)      # → [B,256,32,32]
        # zero-init 的 hint conv，一开始 res32 ≈ 0，不破坏预训练模型
        res32 = self.hint_32(x)        # [B,256,32,32]

        # ===== 32×32 → 16×16, 512 ch =====
        x = self.down_32_16(x)         # → [B,512,16,16]
        x = self.block_16(x, emb)      # → [B,512,16,16]
        res16 = self.hint_16(x)        # [B,512,16,16]

        # ===== 16×16 → 8×8, 1024 ch =====
        x = self.down_16_8(x)          # → [B,1024,8,8]
        x = self.block_8(x, emb)       # → [B,1024,8,8]
        res8 = self.hint_8(x)          # [B,1024,8,8]

        return [res32, res16, res8]
