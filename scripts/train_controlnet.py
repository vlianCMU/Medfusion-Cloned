from pathlib import Path
from datetime import datetime

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from medical_diffusion.data.datamodules import SimpleDataModule
from medical_diffusion.data.datasets import FundusControlNetDataset
from medical_diffusion.models.pipelines import DiffusionPipeline
from medical_diffusion.models.estimators import UNet
from medical_diffusion.models.noise_schedulers import GaussianNoiseScheduler
from medical_diffusion.models.embedders import TimeEmbbeding, SynFundusConditionEmbedder
from medical_diffusion.models.embedders.latent_embedders import VAE
from medical_diffusion.models.estimators.controlnet import ControlNet


def load_pretrained_unet_from_ckpt(
    pipeline: DiffusionPipeline, ckpt_path: str, strict: bool = True
):
    """
    从之前的 DiffusionPipeline 的 checkpoint 中提取 noise_estimator(UNet) 的权重，
    加载到当前 pipeline.noise_estimator 里，并将其冻结（不参与 ControlNet 训练）。
    """
    print(f"[INFO] Loading pretrained UNet from ckpt: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    state_dict = ckpt.get("state_dict", ckpt)

    unet_state_dict = {
        k.replace("noise_estimator.", ""): v
        for k, v in state_dict.items()
        if k.startswith("noise_estimator.")
    }

    missing, unexpected = pipeline.noise_estimator.load_state_dict(
        unet_state_dict, strict=strict
    )

    print("[INFO] Pretrained UNet loaded into pipeline.noise_estimator")
    if not strict:
        if missing:
            print("[WARN] Missing keys in UNet state_dict:", missing)
        if unexpected:
            print("[WARN] Unexpected keys in UNet state_dict:", unexpected)

    for p in pipeline.noise_estimator.parameters():
        p.requires_grad = False
    pipeline.noise_estimator.train()
    print("[INFO] Frozen UNet parameters (requires_grad = False)")


if __name__ == "__main__":
    # ------------ Dataset ----------------
    ds = FundusControlNetDataset(
        csv_path="/data1/lhy/SynFundus-1M_annotation_15w.csv",
        image_dir="/data1/lhy/AutoMorph_copy2/Results/M0/images",
        vessel_dir="/data1/lhy/AutoMorph_copy2/Results/M2/binary_vessel/raw_binary",
        disc_cup_dir="/data1/lhy/AutoMorph_copy2/Results/M2/optic_disc_cup/raw",
        crawler_ext="png",
        image_resize=256,
        augment_horizontal_flip=False,
        augment_vertical_flip=False,
        image_column="img_path",
    )

    dm = SimpleDataModule(
        ds_train=ds,
        batch_size=256,
        num_workers=16,
        pin_memory=True,
    )

    # ------------ Directories ------------
    current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    path_run_dir = Path.cwd() / "controlnet_runs_new_new" / str(current_time)
    path_run_dir.mkdir(parents=True, exist_ok=True)

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    # ------------ Embeddings ------------
    cond_embedder_kwargs = {"emb_dim": 1024, "num_labels": 15}
    time_embedder_kwargs = {"emb_dim": 1024}

    # ------------ Diffusion UNet checkpoint ------------
    diffusion_ckpt = "/data1/lhy/medfusion-main/new_run/2025_11_05_074811/lightning_logs/version_0/checkpoints/last.ckpt"

    # ------------ UNet config (必须与之前 Diffusion 训练时一致) ------------
    noise_estimator_kwargs = {
        "in_ch": 8,
        "out_ch": 8,
        "spatial_dims": 2,
        "hid_chs": [256, 256, 512, 1024],
        "kernel_sizes": [3, 3, 3, 3],
        "strides": [1, 2, 2, 2],
        "time_embedder": TimeEmbbeding,
        "time_embedder_kwargs": time_embedder_kwargs,
        "cond_embedder": SynFundusConditionEmbedder,
        "cond_embedder_kwargs": cond_embedder_kwargs,
        "deep_supervision": False,
        "use_res_block": True,
        "use_attention": "none",
    }

    # ------------ ControlNet config（严格只用 ControlNet 支持的参数）------------
    controlnet_kwargs = {
        "in_ch": 3,  # control 图像 (vessel / disc-cup 等叠加)
        "spatial_dims": 2,
        "hid_chs": [256, 256, 512, 1024],
        "kernel_sizes": [1, 3, 3, 3],
        "strides": [1, 2, 2, 2],
        "time_embedder": TimeEmbbeding,
        "time_embedder_kwargs": time_embedder_kwargs,
        "cond_embedder": SynFundusConditionEmbedder,
        "cond_embedder_kwargs": cond_embedder_kwargs,
    }

    # ------------ Noise Scheduler ------------
    noise_scheduler_kwargs = {
        "timesteps": 1000,
        "beta_start": 0.002,
        "beta_end": 0.02,
        "schedule_strategy": "scaled_linear",
    }

    # ------------ Latent VAE ------------
    latent_embedder_checkpoint = (
        "/data1/lhy/medfusion-main/VAE/runs/2025_11_11_151232/last.ckpt"
    )

    # ------------ Initialize Pipeline ------------
    pipeline = DiffusionPipeline(
        noise_estimator=UNet,
        noise_estimator_kwargs=noise_estimator_kwargs,
        noise_scheduler=GaussianNoiseScheduler,
        noise_scheduler_kwargs=noise_scheduler_kwargs,
        latent_embedder=VAE,
        latent_embedder_checkpoint=latent_embedder_checkpoint,
        estimator_objective="x_T",
        estimate_variance=False,
        use_self_conditioning=False,
        use_ema=False,
        classifier_free_guidance_dropout=0.1,
        do_input_centering=False,
        clip_x0=False,
        sample_every_n_steps=2000,
        controlnet=ControlNet,
        controlnet_kwargs=controlnet_kwargs,
        controlnet_scale=1.0,
        controlnet_cond_dropout=0.0,
    )

    # ------------ 加载预训练 Diffusion UNet 权重 & 冻结 ------------
    load_pretrained_unet_from_ckpt(pipeline, diffusion_ckpt, strict=True)

    # ------------ Training ------------
    checkpointing = ModelCheckpoint(
        filename="{epoch}-{step}",
        monitor="train/loss",
        every_n_train_steps=2000,
        save_last=True,
        save_top_k=5,
        mode="min",
    )

    trainer = Trainer(
        accelerator=accelerator,
        devices=[0,1,3,4],  # 你也可以改成 [0,1,2,3] 或 "auto"
        strategy="ddp",
        default_root_dir=str(path_run_dir),
        callbacks=[checkpointing],
        min_epochs=1000,
        max_epochs=1005,
        limit_val_batches=0,
        num_sanity_val_steps=0,
    )

    trainer.fit(pipeline, datamodule=dm)

    pipeline.save_best_checkpoint(trainer.logger.log_dir, checkpointing.best_model_path)
