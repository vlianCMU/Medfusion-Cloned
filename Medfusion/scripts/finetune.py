#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
典型“微调 UNet”脚本：finetune.py

用法示例（与训练时的设备一致，这里示例用 2、3 号卡）：
    python finetune.py \
        --sd_ckpt "/data1/lhy/medfusion-main/runs/2025_09_10_141511/lightning_logs/version_0/checkpoints/epoch=23-step=5500.ckpt" \
        --vae_ckpt "/data1/lhy/medfusion-main/VAE/runs/2025_09_10_193617/epoch=8-step=82950.ckpt" \
        --csv "/data1/lhy/medfusion-main/merged_final.csv" \
        --devices 2 3 \
        --batch_size 32 \
        --max_epochs 10 \
        --lr 1e-5

注意：
- 该脚本默认只训练 UNet（noise_estimator），其余模块全部冻结。
- 如果 DiffusionPipeline 的实现与训练时完全一致，strict=False 可消除极少量 Key 不匹配的影响。
"""

from pathlib import Path
from datetime import datetime
import argparse
import types

import torch
import torch.nn as nn
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

# ====== 你项目里的模块 ======
from medical_diffusion.data.datamodules import SimpleDataModule
from medical_diffusion.data.datasets import FundusEyeDiseaseDataset
from medical_diffusion.models.pipelines import DiffusionPipeline
from medical_diffusion.models.estimators import UNet
from medical_diffusion.models.noise_schedulers import GaussianNoiseScheduler
from medical_diffusion.models.embedders import TimeEmbbeding, FundusDiseaseEmbedder, LabelEmbedder
from medical_diffusion.models.embedders.latent_embedders import VAE

# ========== 参数 ==========
def get_args():
    parser = argparse.ArgumentParser(description="Fine-tune UNet only for Stable Diffusion-like pipeline")

    # 必填/默认路径（按你提供的）
    parser.add_argument("--sd_ckpt", type=str,
                        default="/data1/lhy/medfusion-main/new_runs/2025_10_14_125958/lightning_logs/version_0/checkpoints/epoch=97-step=17600.ckpt",
                        help="已训练好的 Stable Diffusion pipeline checkpoint 路径")
    parser.add_argument("--vae_ckpt", type=str,
                        default="/data1/lhy/medfusion-main/VAE/runs/2025_09_10_193617/epoch=8-step=82950.ckpt",
                        help="已训练好的 VAE checkpoint 路径（作为 latent_embedder 加载并冻结）")
    parser.add_argument("--csv", type=str,
                        default="/data1/lhy/medfusion-main/finetune.csv",
                        help="FundusEyeDiseaseDataset 的 CSV 路径")

    # 训练配置
    parser.add_argument("--devices", type=int, nargs="+", default=[0,1,2,3,4,5,6,7], help="GPU 设备编号列表")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=55)
    parser.add_argument("--min_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-5, help="UNet 微调学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)

    # 训练日志与采样
    parser.add_argument("--log_every_n_steps", type=int, default=200)
    parser.add_argument("--sample_every_n_steps", type=int, default=200)

    return parser.parse_args()


# ========== 工具函数 ==========
def freeze_module(m: nn.Module):
    if m is None:
        return
    for p in m.parameters():
        p.requires_grad = False


def build_datamodule(csv_path: str, batch_size: int, num_workers: int) -> SimpleDataModule:
    ds = FundusEyeDiseaseDataset(
        csv_path=csv_path,
        crawler_ext="jpg",
        image_resize=256,
        augment_horizontal_flip=True,
        augment_vertical_flip=True,
    )
    dm = SimpleDataModule(
        ds_train=ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )
    return dm


def build_pipeline(vae_ckpt: str, sample_every_n_steps: int) -> DiffusionPipeline:
    # 与你训练时的一致配置（根据你给的代码）
    # cond_embedder = FundusDiseaseEmbedder
    cond_embedder = LabelEmbedder
    cond_embedder_kwargs = {
        "emb_dim": 1024,
        # "num_diseases": 6,
        "num_classes": 6, ###lhy
    }

    time_embedder = TimeEmbbeding
    time_embedder_kwargs = {
        "emb_dim": 1024,
    }

    noise_estimator = UNet
    noise_estimator_kwargs = {
        "in_ch": 8,
        "out_ch": 8,
        "spatial_dims": 2,
        "hid_chs":       [256, 256, 512, 1024],
        "kernel_sizes":  [3, 3, 3, 3],
        "strides":       [1, 2, 2, 2],
        "time_embedder": time_embedder,
        "time_embedder_kwargs": time_embedder_kwargs,
        "cond_embedder": cond_embedder,
        "cond_embedder_kwargs": cond_embedder_kwargs,
        "deep_supervision": False,
        "use_res_block": True,
        "use_attention": "none",
    }

    noise_scheduler = GaussianNoiseScheduler
    noise_scheduler_kwargs = {
        "timesteps": 1000,
        "beta_start": 0.002,
        "beta_end": 0.02,
        "schedule_strategy": "scaled_linear",
    }

    # 只用作 latent 编码/解码，微调不训练
    latent_embedder = VAE
    latent_embedder_checkpoint = vae_ckpt

    pipeline = DiffusionPipeline(
        noise_estimator=noise_estimator,
        noise_estimator_kwargs=noise_estimator_kwargs,
        noise_scheduler=noise_scheduler,
        noise_scheduler_kwargs=noise_scheduler_kwargs,
        latent_embedder=latent_embedder,
        latent_embedder_checkpoint=latent_embedder_checkpoint,
        estimator_objective="x_T",
        estimate_variance=False,
        use_self_conditioning=False,
        use_ema=True,
        classifier_free_guidance_dropout=0.0,  # 微调时通常设为 0
        do_input_centering=False,
        clip_x0=False,
        sample_every_n_steps=sample_every_n_steps,
    )
    return pipeline


def inject_unet_only_optim(pipeline: DiffusionPipeline, lr: float, weight_decay: float, max_epochs: int):
    """在实例上注入只优化 UNet 的 configure_optimizers（带 Cosine Scheduler）。"""
    def configure_optimizers_unet_only(self):
        unet_params = [p for p in self.noise_estimator.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            unet_params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999)
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs, eta_min=lr * 0.1
        )
        return [optimizer], [scheduler]
    pipeline.configure_optimizers = types.MethodType(configure_optimizers_unet_only, pipeline)



def main():
    args = get_args()
    seed_everything(args.seed)

    # 运行目录
    current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    run_dir = Path.cwd() / "finetune" / f"{current_time}_finetune_unet"
    run_dir.mkdir(parents=True, exist_ok=True)

    # 数据
    dm = build_datamodule(args.csv, args.batch_size, args.num_workers)

    # 模型
    pipeline = build_pipeline(args.vae_ckpt, args.sample_every_n_steps)

    # 从已训练好的全模型 ckpt 加载 state_dict（不恢复优化器）
    print(f"[Info] Loading pretrained pipeline weights from: {args.sd_ckpt}")
    ckpt = torch.load(args.sd_ckpt, map_location="cpu")
    missing, unexpected = pipeline.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
    if missing:
        print("[Warn] Missing keys:", missing)
    if unexpected:
        print("[Warn] Unexpected keys:", unexpected)

    # 冻结除 UNet 外所有模块（只训练 UNet）
    if hasattr(pipeline, "latent_embedder"):
        freeze_module(pipeline.latent_embedder)
    if hasattr(pipeline, "cond_embedder"):
        freeze_module(pipeline.cond_embedder)
    if hasattr(pipeline, "time_embedder"):
        freeze_module(pipeline.time_embedder)
    # UNet 参与训练
    for p in pipeline.noise_estimator.parameters():
        p.requires_grad = True

    # 只给 UNet 配优化器
    inject_unet_only_optim(pipeline, lr=args.lr, weight_decay=args.weight_decay, max_epochs=args.max_epochs)

    # Callback（不做验证时，常以训练 loss 作为监控指标）
    to_monitor = "train/loss"
    checkpoint_cb = ModelCheckpoint(
        filename="{epoch}-{step}",
        monitor=to_monitor,
        every_n_train_steps=args.log_every_n_steps,
        save_last=True,
        save_top_k=3,
        mode="min",
    )

    # Trainer
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    strategy = DDPStrategy(find_unused_parameters=False) if accelerator == "gpu" and len(args.devices) > 1 else "auto"

    trainer = Trainer(
        accelerator=accelerator,
        devices=args.devices,
        strategy=strategy,
        default_root_dir=str(run_dir),
        gradient_clip_val=1.0,
        callbacks=[checkpoint_cb],
        enable_checkpointing=True,
        check_val_every_n_epoch=1,
        log_every_n_steps=args.log_every_n_steps,
        limit_val_batches=0,        # 微调阶段通常可不做验证；若需要验证，改为 >0 并提供 val 集
        min_epochs=args.min_epochs,
        max_epochs=args.max_epochs,
        num_sanity_val_steps=0,
    )

    # 训练
    trainer.fit(pipeline, datamodule=dm)

    # 结束信息
    print(f"[Done] Fine-tune finished. Logs/ckpts at: {trainer.logger.log_dir if trainer.logger else run_dir}")


if __name__ == "__main__":
    main()
