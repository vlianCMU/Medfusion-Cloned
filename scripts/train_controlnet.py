from pathlib import Path
from datetime import datetime

import torch
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from medical_diffusion.data.datamodules import SimpleDataModule
from medical_diffusion.data.datasets import FundusControlNetDataset
from medical_diffusion.models.pipelines import DiffusionPipeline
from medical_diffusion.models.estimators import UNet, ControlNet
from medical_diffusion.models.noise_schedulers import GaussianNoiseScheduler
from medical_diffusion.models.embedders import TimeEmbbeding, SynFundusConditionEmbedder
from medical_diffusion.models.embedders.latent_embedders import VAE


if __name__ == "__main__":
    # ------------ Load Data ----------------
    ds = FundusControlNetDataset(
        csv_path='/data1/lhy/GENFUND/genfund_conditions.csv',
        image_dir='/data1/lhy/AutoMorph/Results/M0/images',
        vessel_dir='/data1/lhy/AutoMorph/Results/M2/binary_vessel/raw_binary',
        disc_cup_dir='/data1/lhy/AutoMorph/Results/M2/optic_disc_cup/raw',
        crawler_ext='png',
        image_resize=256,
        augment_horizontal_flip=False,
        augment_vertical_flip=False,
        image_column='img_path',
    )

    dm = SimpleDataModule(
        ds_train=ds,
        batch_size=32,
        num_workers=16,
        pin_memory=True,
    )

    current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    path_run_dir = Path.cwd() / 'controlnet_runs' / str(current_time)
    path_run_dir.mkdir(parents=True, exist_ok=True)
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'

    # ------------ Initialize Model ------------
    cond_embedder = SynFundusConditionEmbedder
    cond_embedder_kwargs = {
        'emb_dim': 1024,
        'num_labels': 15
    }

    time_embedder = TimeEmbbeding
    time_embedder_kwargs = {
        'emb_dim': 1024
    }

    noise_estimator = UNet
    noise_estimator_kwargs = {
        'in_ch': 8,
        'out_ch': 8,
        'spatial_dims': 2,
        'hid_chs': [256, 256, 512, 1024],
        'kernel_sizes': [3, 3, 3, 3],
        'strides': [1, 2, 2, 2],
        'time_embedder': time_embedder,
        'time_embedder_kwargs': time_embedder_kwargs,
        'cond_embedder': cond_embedder,
        'cond_embedder_kwargs': cond_embedder_kwargs,
        'deep_supervision': False,
        'use_res_block': True,
        'use_attention': 'none',
    }

    controlnet_kwargs = {
        'in_ch': 3,  # RGB control 图
        'spatial_dims': noise_estimator_kwargs['spatial_dims'],
        'hid_chs': noise_estimator_kwargs['hid_chs'],
        'kernel_sizes': [1, 3, 3, 3],
        'strides': noise_estimator_kwargs['strides'],
        'time_embedder': time_embedder,
        'time_embedder_kwargs': time_embedder_kwargs,
        'cond_embedder': cond_embedder,
        'cond_embedder_kwargs': cond_embedder_kwargs,
    }

    # ------------ Initialize Noise ------------
    noise_scheduler = GaussianNoiseScheduler
    noise_scheduler_kwargs = {
        'timesteps': 1000,
        'beta_start': 0.002,
        'beta_end': 0.02,
        'schedule_strategy': 'scaled_linear'
    }

    # ------------ Initialize Latent Space  ------------
    latent_embedder = VAE
    latent_embedder_checkpoint = '/data1/lhy/medfusion-main/VAE/runs/2025_11_11_151232/last.ckpt'

    # ------------ Initialize Pipeline ------------
    pipeline = DiffusionPipeline(
        noise_estimator=noise_estimator,
        noise_estimator_kwargs=noise_estimator_kwargs,
        noise_scheduler=noise_scheduler,
        noise_scheduler_kwargs=noise_scheduler_kwargs,
        latent_embedder=latent_embedder,
        latent_embedder_checkpoint=latent_embedder_checkpoint,
        estimator_objective='x_T',
        estimate_variance=False,
        use_self_conditioning=False,
        classifier_free_guidance_dropout=0.2,
        do_input_centering=False,
        clip_x0=False,
        sample_every_n_steps=2000,
        controlnet=ControlNet,
        controlnet_kwargs=controlnet_kwargs,
        controlnet_cond_dropout=0.05,
        controlnet_scale=1.0,
    )

    print(pipeline.noise_estimator)

    # -------------- Training Initialization ---------------
    to_monitor = "train/loss"
    save_and_sample_every = 2000

    checkpointing = ModelCheckpoint(
        filename='{epoch}-{step}',
        monitor=to_monitor,
        every_n_train_steps=save_and_sample_every,
        save_last=True,
        save_top_k=5,
        mode='min',
    )

    trainer = Trainer(
        accelerator=accelerator,
        devices=[1,2],
        strategy='ddp',
        accumulate_grad_batches=1,
        default_root_dir=str(path_run_dir),
        callbacks=[checkpointing],
        enable_checkpointing=True,
        check_val_every_n_epoch=1,
        log_every_n_steps=save_and_sample_every,
        auto_lr_find=False,
        limit_val_batches=0,
        min_epochs=100,
        max_epochs=105,
        num_sanity_val_steps=0,
    )

    # ---------------- Execute Training ----------------
    trainer.fit(pipeline, datamodule=dm)

    # ------------- Save path to best model -------------
    pipeline.save_best_checkpoint(trainer.logger.log_dir, checkpointing.best_model_path)