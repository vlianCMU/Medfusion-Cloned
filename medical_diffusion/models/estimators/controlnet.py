import torch
import torch.nn as nn

from medical_diffusion.models.embedders import TimeEmbbeding
from medical_diffusion.models.utils.conv_blocks import UnetResBlock, UnetBasicBlock, DownBlock, save_add


class ControlNet(nn.Module):
    """Lightweight ControlNet implementation aligned with the UNet backbone.

    The module mirrors the down-sampling path of :class:`medical_diffusion.models.estimators.unet.UNet`
    and outputs residual feature maps that can be injected into the main UNet decoder.
    """

    def __init__(
        self,
        in_ch: int = 3,
        spatial_dims: int = 2,
        hid_chs=None,
        kernel_sizes=None,
        strides=None,
        act_name=("SWISH", {}),
        norm_name=("GROUP", {"num_groups": 32, "affine": True}),
        time_embedder=TimeEmbbeding,
        time_embedder_kwargs=None,
        cond_embedder=None,
        cond_embedder_kwargs=None,
        use_res_block: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        hid_chs = [32, 64, 128, 256] if hid_chs is None else hid_chs
        kernel_sizes = [1, 3, 3, 3] if kernel_sizes is None else kernel_sizes
        strides = [1, 2, 2, 2] if strides is None else strides
        time_embedder_kwargs = time_embedder_kwargs or {}
        cond_embedder_kwargs = cond_embedder_kwargs or {}

        ConvBlock = UnetResBlock if use_res_block else UnetBasicBlock

        # Embedders
        if time_embedder is not None:
            self.time_embedder = time_embedder(**time_embedder_kwargs)
            time_emb_dim = self.time_embedder.emb_dim
        else:
            self.time_embedder = None
            time_emb_dim = None

        if cond_embedder is not None:
            self.cond_embedder = cond_embedder(**cond_embedder_kwargs)
        else:
            self.cond_embedder = None

        # Stem
        self.inc = ConvBlock(
            spatial_dims=spatial_dims,
            in_channels=in_ch,
            out_channels=hid_chs[0],
            kernel_size=kernel_sizes[0],
            stride=strides[0],
            act_name=act_name,
            norm_name=norm_name,
            emb_channels=time_emb_dim,
        )

        # Down path
        self.encoders = nn.ModuleList(
            [
                DownBlock(
                    spatial_dims=spatial_dims,
                    in_channels=hid_chs[i - 1],
                    out_channels=hid_chs[i],
                    kernel_size=kernel_sizes[i],
                    stride=strides[i],
                    downsample_kernel_size=kernel_sizes[i],
                    norm_name=norm_name,
                    act_name=act_name,
                    dropout=dropout,
                    use_res_block=use_res_block,
                    learnable_interpolation=True,
                    use_attention='none',
                    emb_channels=time_emb_dim,
                )
                for i in range(1, len(strides))
            ]
        )

        # Zero convolutions that project features into residual hints
        self.zero_convs = nn.ModuleList(
            [nn.Conv2d(ch, ch, kernel_size=1) for ch in hid_chs]
        )
        for layer in self.zero_convs:
            nn.init.zeros_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, x, t=None, condition=None):
        # Build embeddings
        if t is None:
            time_emb = None
        else:
            time_emb = self.time_embedder(t) if self.time_embedder is not None else None

        if (condition is None) or (self.cond_embedder is None):
            cond_emb = None
        else:
            cond_emb = self.cond_embedder(condition)

        emb = save_add(time_emb, cond_emb)

        residuals = []
        x_cur = self.inc(x, emb)
        residuals.append(self.zero_convs[0](x_cur))

        for i, encoder in enumerate(self.encoders, start=1):
            x_cur = encoder(x_cur, emb)
            residuals.append(self.zero_convs[i](x_cur))

        return residuals