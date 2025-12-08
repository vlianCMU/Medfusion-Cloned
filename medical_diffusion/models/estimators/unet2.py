import torch 
import torch.nn as nn 
from monai.networks.blocks import UnetOutBlock

from medical_diffusion.models.utils.conv_blocks import (
    BasicBlock, 
    UpBlock, 
    DownBlock, 
    UnetBasicBlock, 
    UnetResBlock, 
    save_add, 
    BasicDown, 
    BasicUp, 
    SequentialEmb
)
from medical_diffusion.models.embedders import TimeEmbbeding
from medical_diffusion.models.utils.attention_blocks import Attention, zero_module


class UNet(nn.Module):
    """UNet2 with ControlNet support.
    
    这个 UNet 设计用于在 latent 空间 (32x32) 进行扩散，
    并支持从 ControlNet 接收控制残差。
    
    ControlNet residuals 预期形状:
        - residuals[0]: [B, 256, 32, 32]  加到 in_conv 输出后
        - residuals[1]: [B, 512, 16, 16]  加到 16x16 第一个 ResBlock 之后
        - residuals[2]: [B, 1024, 8, 8]   加到 8x8  第一个 ResBlock 之后
    """

    def __init__(self, 
            in_ch=1, 
            out_ch=1, 
            spatial_dims = 3,
            hid_chs =    [256, 256, 512,  1024],
            kernel_sizes=[ 3,  3,   3,   3],
            strides =    [ 1,  2,   2,   2], # WARNING, last stride is ignored (follows OpenAI)
            act_name=("SWISH", {}),
            norm_name = ("GROUP", {'num_groups':32, "affine": True}),
            time_embedder=TimeEmbbeding,
            time_embedder_kwargs={},
            cond_embedder=None,
            cond_embedder_kwargs={},
            deep_supervision=True, # True = all but last layer, 0/False=disable, 1=only first layer, ... 
            use_res_block=True,
            estimate_variance=False ,
            use_self_conditioning = False, 
            dropout=0.0, 
            learnable_interpolation=True,
            use_attention='none',
            num_res_blocks=2,
        ):
        super().__init__()
        use_attention = use_attention if isinstance(use_attention, list) else [use_attention]*len(strides) 
        self.use_self_conditioning = use_self_conditioning
        self.use_res_block = use_res_block
        self.depth = len(strides)
        self.num_res_blocks = num_res_blocks

        # ------------- Time-Embedder-----------
        if time_embedder is not None:
            self.time_embedder=time_embedder(**time_embedder_kwargs)
            time_emb_dim = self.time_embedder.emb_dim
        else:
            self.time_embedder = None 
            time_emb_dim = None 

        # ------------- Condition-Embedder-----------
        if cond_embedder is not None:
            self.cond_embedder=cond_embedder(**cond_embedder_kwargs)
            cond_emb_dim = self.cond_embedder.emb_dim
        else:
            self.cond_embedder = None 
            cond_emb_dim = None 

        ConvBlock = UnetResBlock if use_res_block else UnetBasicBlock

        # ----------- In-Convolution ------------
        in_ch = in_ch*2 if self.use_self_conditioning else in_ch 
        self.in_conv = BasicBlock(spatial_dims, in_ch, hid_chs[0], kernel_size=kernel_sizes[0], stride=strides[0])
        
        
        # ----------- Encoder ------------
        in_blocks = [] 
        for i in range(1, self.depth):
            for k in range(num_res_blocks):
                seq_list = [] 
                seq_list.append(
                    ConvBlock(
                        spatial_dims=spatial_dims,
                        in_channels=hid_chs[i-1 if k==0 else i],
                        out_channels=hid_chs[i],
                        kernel_size=kernel_sizes[i],
                        stride=1,
                        norm_name=norm_name,
                        act_name=act_name,
                        dropout=dropout,
                        emb_channels=time_emb_dim
                    )
                )

                seq_list.append(
                    Attention(
                        spatial_dims=spatial_dims,
                        in_channels=hid_chs[i],
                        out_channels=hid_chs[i],
                        num_heads=8,
                        ch_per_head=hid_chs[i]//8,
                        depth=1,
                        norm_name=norm_name,
                        dropout=dropout,
                        emb_dim=time_emb_dim,
                        attention_type=use_attention[i]
                    )
                )
                in_blocks.append(SequentialEmb(*seq_list))

            if i < self.depth-1:
                in_blocks.append(
                    BasicDown(
                        spatial_dims=spatial_dims,
                        in_channels=hid_chs[i],
                        out_channels=hid_chs[i],
                        kernel_size=kernel_sizes[i],
                        stride=strides[i],
                        learnable_interpolation=learnable_interpolation 
                    )
                )
 
        self.in_blocks = nn.ModuleList(in_blocks)
        
        # ----------- 记录 DownBlock 的位置索引 -----------
        # 对 depth=4, num_res_blocks=2:
        # in_blocks:
        #   0: i=1, k=0 ResBlock (32x32, 256)
        #   1: i=1, k=1 ResBlock (32x32, 256)
        #   2: i=1 DownBlock      (32→16, 256)
        #   3: i=2, k=0 ResBlock (16x16, 512)
        #   4: i=2, k=1 ResBlock (16x16, 512)
        #   5: i=2 DownBlock      (16→8, 512)
        #   6: i=3, k=0 ResBlock ( 8x8, 1024)
        #   7: i=3, k=1 ResBlock ( 8x8, 1024)
        #
        # 我们希望：
        #   residuals[1] 加在 16x16 第一个 ResBlock 之后 → index = first_down_idx + 1
        #   residuals[2] 加在  8x8 第一个 ResBlock 之后 → index = second_down_idx + 1
        self.downblock_indices = []
        block_idx = 0
        for i in range(1, self.depth):
            block_idx += num_res_blocks  # ResBlocks
            if i < self.depth - 1:
                self.downblock_indices.append(block_idx)  # DownBlock 位置
                block_idx += 1
        
        # ----------- Middle ------------
        self.middle_block = SequentialEmb(
            ConvBlock(
                spatial_dims=spatial_dims,
                in_channels=hid_chs[-1],
                out_channels=hid_chs[-1],
                kernel_size=kernel_sizes[-1],
                stride=1,
                norm_name=norm_name,
                act_name=act_name,
                dropout=dropout,
                emb_channels=time_emb_dim
            ),
            Attention(
                spatial_dims=spatial_dims,
                in_channels=hid_chs[-1],
                out_channels=hid_chs[-1],
                num_heads=8,
                ch_per_head=hid_chs[-1]//8,
                depth=1,
                norm_name=norm_name,
                dropout=dropout,
                emb_dim=time_emb_dim,
                attention_type=use_attention[-1]
            ),
            ConvBlock(
                spatial_dims=spatial_dims,
                in_channels=hid_chs[-1],
                out_channels=hid_chs[-1],
                kernel_size=kernel_sizes[-1],
                stride=1,
                norm_name=norm_name,
                act_name=act_name,
                dropout=dropout,
                emb_channels=time_emb_dim
            )
        )

        # ------------ Decoder ----------
        out_blocks = [] 
        for i in range(1, self.depth):
            for k in range(num_res_blocks+1):
                seq_list = [] 
                out_channels=hid_chs[i-1 if k==0 else i]
                seq_list.append(
                    ConvBlock(
                        spatial_dims=spatial_dims,
                        in_channels=hid_chs[i]+hid_chs[i-1 if k==0 else i],
                        out_channels=out_channels,
                        kernel_size=kernel_sizes[i],
                        stride=1,
                        norm_name=norm_name,
                        act_name=act_name,
                        dropout=dropout,
                        emb_channels=time_emb_dim
                    )
                )
            
                seq_list.append(
                    Attention(
                        spatial_dims=spatial_dims,
                        in_channels=out_channels,
                        out_channels=out_channels,
                        num_heads=8,
                        ch_per_head=out_channels//8,
                        depth=1,
                        norm_name=norm_name,
                        dropout=dropout,
                        emb_dim=time_emb_dim,
                        attention_type=use_attention[i]
                    )
                )

                if (i >1) and k==0:
                    seq_list.append(
                        BasicUp(
                            spatial_dims=spatial_dims,
                            in_channels=out_channels,
                            out_channels=out_channels,
                            kernel_size=strides[i],
                            stride=strides[i],
                            learnable_interpolation=learnable_interpolation 
                        )
                    )
        
                out_blocks.append(SequentialEmb(*seq_list))
        self.out_blocks = nn.ModuleList(out_blocks)
        
        
        # --------------- Out-Convolution ----------------
        out_ch_hor = out_ch*2 if estimate_variance else out_ch
        self.outc = zero_module(UnetOutBlock(spatial_dims, hid_chs[0], out_ch_hor, dropout=None))
        if isinstance(deep_supervision, bool):
            deep_supervision = self.depth-2 if deep_supervision else 0 
        self.outc_ver = nn.ModuleList([
            zero_module(UnetOutBlock(spatial_dims, hid_chs[i]+hid_chs[i-1], out_ch, dropout=None) )
            for i in range(2, deep_supervision+2)
        ])
 

    def forward(self, x_t, t=None, condition=None, self_cond=None, control_residuals=None):
        """
        Forward pass with ControlNet support.
        
        Parameters
        ----------
        x_t : torch.Tensor
            Noisy latent, shape [B, C, H, W] (e.g., [B, 8, 32, 32])
        t : torch.Tensor
            Timestep, shape [B,]
        condition : torch.Tensor
            Conditioning labels, shape [B, num_labels]
        self_cond : torch.Tensor
            Self-conditioning input (optional)
        control_residuals : list of torch.Tensor or None
            List of 3 residual tensors from ControlNet:
            - residuals[0]: [B, 256, 32, 32] - 加到 in_conv 输出后
            - residuals[1]: [B, 512, 16, 16] - 加到 16x16 第一层 ResBlock 之后
            - residuals[2]: [B, 1024, 8, 8]  - 加到  8x8 第一层 ResBlock 之后
        """
        # -------- Time Embedding (Global) -----------
        if t is None:
            time_emb = None 
        else:
            time_emb = self.time_embedder(t) # [B, C]

        # -------- Condition Embedding (Global) -----------
        if (condition is None) or (self.cond_embedder is None):
            cond_emb = None  
        else:
            cond_emb = self.cond_embedder(condition) # [B, C]
        
        emb = save_add(time_emb, cond_emb)
       
        # ---------- Self-conditioning-----------
        if self.use_self_conditioning:
            self_cond =  torch.zeros_like(x_t) if self_cond is None else x_t 
            x_t = torch.cat([x_t, self_cond], dim=1)  
    
        # --------- Encoder --------------
        # in_conv: 输入卷积
        x = [self.in_conv(x_t)]
        
        # 添加第一个 control_residual (32x32, 256ch)
        if control_residuals is not None and len(control_residuals) > 0:
            x[0] = x[0] + control_residuals[0]
        
        # 通过 in_blocks
        for i in range(len(self.in_blocks)):
            x.append(self.in_blocks[i](x[i], emb))
            
            # ---- 在正确的尺度注入 control_residuals[1]/[2] ----
            if control_residuals is not None:
                # 16x16 注入点：第一个 DownBlock 之后的第一个 ResBlock
                if (
                    len(self.downblock_indices) > 0
                    and i == self.downblock_indices[0] + 1
                    and len(control_residuals) > 1
                ):
                    # 当前 x[-1] 形状: [B, 512, 16, 16]
                    x[-1] = x[-1] + control_residuals[1]

                # 8x8 注入点：第二个 DownBlock 之后的第一个 ResBlock
                if (
                    len(self.downblock_indices) > 1
                    and i == self.downblock_indices[1] + 1
                    and len(control_residuals) > 2
                ):
                    # 当前 x[-1] 形状: [B, 1024, 8, 8]
                    x[-1] = x[-1] + control_residuals[2]

        # ---------- Middle --------------
        h = self.middle_block(x[-1], emb)
        
        # -------- Decoder -----------
        y_ver = []
        for i in range(len(self.out_blocks), 0, -1):
            h = torch.cat([h, x.pop()], dim=1)

            depth, j = i//(self.num_res_blocks+1), i%(self.num_res_blocks+1)-1
            y_ver.append(self.outc_ver[depth-1](h)) if (len(self.outc_ver)>=depth>0) and (j==0) else None 

            h = self.out_blocks[i-1](h, emb)

        # ---------Out-Convolution ------------
        y = self.outc(h)

        return y, y_ver[::-1]
