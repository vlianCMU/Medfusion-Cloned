"""
Diffusion Pipeline with ControlNet Support

这个版本的 DiffusionPipeline 完全支持 ControlNet：
- ControlNet 处理 256x256 的分割图
- 产生 3 个 residuals 注入到 UNet2 的编码器
- 不对 control 图像进行 VAE 编码
"""

from pathlib import Path 
from tqdm import tqdm

import torch 
import torch.nn.functional as F 
from torchvision.utils import save_image 
import streamlit as st

from medical_diffusion.models import BasicModel
from medical_diffusion.utils.train_utils import EMAModel
from medical_diffusion.utils.math_utils import kl_gaussians


class DiffusionPipeline(BasicModel):
    def __init__(self, 
        noise_scheduler,
        noise_estimator,
        latent_embedder=None,
        controlnet=None,
        noise_scheduler_kwargs={},
        noise_estimator_kwargs={},
        latent_embedder_checkpoint='',
        controlnet_kwargs=None,
        estimator_objective = 'x_T', # 'x_T' or 'x_0'
        estimate_variance=False,
        use_self_conditioning=False,
        classifier_free_guidance_dropout=0.5,
        controlnet_cond_dropout=0.0,
        controlnet_scale=1.0,  # 可以是标量或列表
        num_samples = 4,
        do_input_centering = True,
        clip_x0=True,
        use_ema = False,
        ema_kwargs = {},
        optimizer=torch.optim.AdamW, 
        optimizer_kwargs={'lr':1e-4},
        lr_scheduler= None,
        lr_scheduler_kwargs={}, 
        loss=torch.nn.L1Loss,
        loss_kwargs={},
        sample_every_n_steps = 1000
        ):
        super().__init__(optimizer, optimizer_kwargs, lr_scheduler, lr_scheduler_kwargs)
        self.loss_fct = loss(**loss_kwargs)
        self.sample_every_n_steps=sample_every_n_steps

        noise_estimator_kwargs['estimate_variance'] = estimate_variance
        noise_estimator_kwargs['use_self_conditioning'] = use_self_conditioning

        controlnet_kwargs = controlnet_kwargs or {}

        self.noise_scheduler = noise_scheduler(**noise_scheduler_kwargs)
        self.noise_estimator = noise_estimator(**noise_estimator_kwargs)
        self.controlnet = controlnet(**controlnet_kwargs) if controlnet is not None else None
        
        # 冻结 ControlNet 的条件嵌入器（如果使用与 UNet 相同的嵌入器）
        # 这样可以节省内存，因为嵌入器参数是共享的
        if self.controlnet is not None:
            # 冻结 latent embedder
            for param in self.controlnet.parameters():
                param.requires_grad = True  # ControlNet 需要训练
        
        with torch.no_grad():
            if latent_embedder is not None:
                self.latent_embedder = latent_embedder.load_from_checkpoint(latent_embedder_checkpoint)
                for param in self.latent_embedder.parameters():
                    param.requires_grad = False
            else:
                self.latent_embedder = None 

        self.estimator_objective = estimator_objective
        self.use_self_conditioning = use_self_conditioning
        self.num_samples = num_samples
        self.classifier_free_guidance_dropout = classifier_free_guidance_dropout
        self.controlnet_cond_dropout = controlnet_cond_dropout
        self.do_input_centering = do_input_centering
        self.estimate_variance = estimate_variance
        self.clip_x0 = clip_x0
        self.controlnet_scale = controlnet_scale

        self.use_ema = use_ema
        if use_ema:
            self.ema_model = EMAModel(self.noise_estimator, **ema_kwargs)


    def _encode_latent(self, x):
        """将图像编码到 latent 空间"""
        if self.latent_embedder is not None:
            self.latent_embedder.eval()
            with torch.no_grad():
                x = self.latent_embedder.encode(x)
        if self.do_input_centering:
            x = 2 * x - 1
        return x

    def _prepare_control(self, control, t, condition):
        """
        准备 ControlNet 的控制残差。
        
        重要：control 图像 (256x256) 不需要 VAE 编码！
        ControlNet 直接处理原始分割图，然后通过自己的编码器
        产生与 latent 空间 UNet 匹配的 residuals。
        
        Parameters
        ----------
        control : torch.Tensor
            控制图像，shape [B, 3, 256, 256]
        t : torch.Tensor
            时间步，shape [B,]
        condition : torch.Tensor
            条件标签，shape [B, num_labels]
        
        Returns
        -------
        residuals : list of torch.Tensor or None
            控制残差列表，每个元素的 shape 分别为:
            - [B, 256, 32, 32]
            - [B, 512, 16, 16]
            - [B, 1024, 8, 8]
        """
        if (self.controlnet is None) or (control is None):
            return None

        # dropout only disables conditioning
        if self.training and self.controlnet_cond_dropout > 0:
            if torch.rand(1).item() < self.controlnet_cond_dropout:
                return None

        # ❗ ensure grad is tracked (no no_grad!)
        residuals = self.controlnet(control, t, condition)

        # apply scale
        if isinstance(self.controlnet_scale, (tuple, list)):
            scaled = [
                r * float(self.controlnet_scale[min(i, len(self.controlnet_scale)-1)])
                for i, r in enumerate(residuals)
            ]
        else:
            scaled = [r * float(self.controlnet_scale) for r in residuals]

        return scaled


    def _step(self, batch: dict, batch_idx: int, state: str, step: int, optimizer_idx: int):
        """训练/验证步骤"""
        results = {}
        
        # 编码源图像到 latent 空间
        x_0 = self._encode_latent(batch['source'])
        labels = batch.get('labels', None)
        control = batch.get('control', None)
        condition = labels if labels is not None else None

        # 采样噪声
        with torch.no_grad():
            x_t, x_T, t = self.noise_scheduler.sample(x_0) 
                
        # 选择模型
        if self.use_ema and (state != 'train'):
            noise_estimator = self.ema_model.averaged_model
        else:
            noise_estimator = self.noise_estimator

        # 准备 control residuals
        control_residuals = self._prepare_control(control, t, condition)

        # Self-conditioning
        self_cond = None
        if self.use_self_conditioning:

            # ❗ 禁止 control 分支参与 self-conditioning
            with torch.no_grad():
                pred_sc, _ = noise_estimator(
                    x_t, t, condition=condition, self_cond=None, control_residuals=None
                )

            if self.estimate_variance:
                pred_sc, _ = pred_sc.chunk(2, dim=1)

            if self.estimator_objective == "x_T":
                self_cond = self.noise_scheduler.estimate_x_0(
                    x_t, pred_sc, t=t, clip_x0=self.clip_x0
                )
            else:
                self_cond = self.noise_scheduler.estimate_x_T(
                    x_t, pred_sc, t=t, clip_x0=self.clip_x0
                )

        # Classifier-free guidance dropout
        if self.classifier_free_guidance_dropout > 0:
            drop_mask = torch.rand(condition.shape[0], device=condition.device) < self.classifier_free_guidance_dropout
            condition = condition.clone()
            condition[drop_mask] = 0

        # 模型预测
        pred, pred_vertical = noise_estimator(x_t, t, condition, self_cond, control_residuals)
        if self.estimate_variance:
            pred, pred_var = pred.chunk(2, dim=1)

        # 计算目标
        if self.estimator_objective == "x_T":
            target = x_T
        elif self.estimator_objective == "x_0":
            target = x_0
        else:
            raise NotImplementedError()

        # 计算损失
        interpolation_mode = 'area'
        loss = 0
        weights = [1/2**i for i in range(1 + len(pred_vertical))]
        tot_weight = sum(weights)
        weights = [w/tot_weight for w in weights]

        # MSE/L1 Loss
        loss += self.loss_fct(pred, target) * weights[0]

        # Variance Loss (如果启用)
        if self.estimate_variance:
            var_scale = (pred_var + 1) / 2
            pred_logvar = self.noise_scheduler.estimate_variance_t(t, x_t.ndim, log=True, var_scale=var_scale)

            if self.estimator_objective == 'x_T':
                pred_x_0 = self.noise_scheduler.estimate_x_0(x_t, x_T, t, clip_x0=self.clip_x0)
            elif self.estimator_objective == "x_0":
                pred_x_0 = pred

            with torch.no_grad():
                pred_mean = self.noise_scheduler.estimate_mean_t(x_t, pred_x_0, t)
                true_mean = self.noise_scheduler.estimate_mean_t(x_t, x_0, t)
                true_logvar = self.noise_scheduler.estimate_variance_t(t, x_t.ndim, log=True, var_scale=0)
            
            kl_loss = torch.mean(kl_gaussians(true_mean, true_logvar, pred_mean, pred_logvar), dim=list(range(1, x_0.ndim)))
            nnl_loss = torch.mean(F.gaussian_nll_loss(pred_x_0, x_0, torch.exp(pred_logvar), reduction='none'), dim=list(range(1, x_0.ndim)))
            var_loss = torch.mean(torch.where(t == 0, nnl_loss, kl_loss))
            loss += var_loss
            
            results['variance_scale'] = torch.mean(var_scale)
            results['variance_loss'] = var_loss

        # Deep Supervision
        for i, pred_i in enumerate(pred_vertical): 
            if pred_i is not None:
                target_i = F.interpolate(target, size=pred_i.shape[2:], mode=interpolation_mode, align_corners=None)
                loss += self.loss_fct(pred_i, target_i) * weights[i + 1]

        # Logging
        self.log(f"{state}/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        results['loss'] = loss
        
        # 定期采样
        if (state == 'train') and (step != 0) and (step % self.sample_every_n_steps == 0):
            log_step = step // self.sample_every_n_steps
            path_out = Path(self.logger.log_dir) / 'images'
            path_out.mkdir(parents=True, exist_ok=True)
            
            # 使用 control 条件生成样本（如果可用）
            with torch.no_grad():
                sample_control = control[:self.num_samples] if control is not None else None
                sample_condition = condition[:self.num_samples] if condition is not None else None
                latent_shape = x_0.shape[1:]
                sample_img = self.sample(
                    self.num_samples, 
                    latent_shape, 
                    condition=sample_condition, 
                    control=sample_control
                )
            
            def depth2batch(image):
                return (image if image.ndim < 5 else torch.swapaxes(image[0], 0, 1))
            images = depth2batch(sample_img)[:32]
            save_image(images, path_out / f'sample_{log_step}.png', normalize=True)
        
        return loss

    def forward(self, x_t, t, condition=None, self_cond=None, guidance_scale=1.0, cold_diffusion=False, un_cond=None, control_residuals=None):
        """前向传播（采样时使用）"""
        if self.use_ema:
            noise_estimator = self.ema_model.averaged_model
        else:
            noise_estimator = self.noise_estimator

        # Classifier-free guidance
        if (condition is not None) and (guidance_scale != 1.0):
            pred_uncond, _ = noise_estimator(x_t, t, condition=un_cond, self_cond=self_cond, control_residuals=control_residuals)
            pred_cond, _ = noise_estimator(x_t, t, condition=condition, self_cond=self_cond, control_residuals=control_residuals)
            pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

            if self.estimate_variance:
                pred_uncond, pred_var_uncond = pred_uncond.chunk(2, dim=1)
                pred_cond, pred_var_cond = pred_cond.chunk(2, dim=1)
                pred_var = pred_var_uncond + guidance_scale * (pred_var_cond - pred_var_uncond)
        else:
            pred, _ = noise_estimator(x_t, t, condition=condition, self_cond=self_cond, control_residuals=control_residuals)
            if self.estimate_variance:
                pred, pred_var = pred.chunk(2, dim=1)

        if self.estimate_variance:
            pred_var_scale = pred_var / 2 + 0.5
            pred_var_value = pred_var
        else:
            pred_var_scale = 0
            pred_var_value = None

        if self.estimator_objective == 'x_0':
            x_t_prior, x_0 = self.noise_scheduler.estimate_x_t_prior_from_x_0(x_t, t, pred, clip_x0=self.clip_x0, var_scale=pred_var_scale, cold_diffusion=cold_diffusion)
            x_T = self.noise_scheduler.estimate_x_T(x_t, x_0=pred, t=t, clip_x0=self.clip_x0)
            self_cond = x_T
        elif self.estimator_objective == 'x_T':
            x_t_prior, x_0 = self.noise_scheduler.estimate_x_t_prior_from_x_T(x_t, t, pred, clip_x0=self.clip_x0, var_scale=pred_var_scale, cold_diffusion=cold_diffusion)
            x_T = pred
            self_cond = x_0
        else:
            raise ValueError("Unknown Objective")
        
        return x_t_prior, x_0, x_T, self_cond


    @torch.no_grad()
    def denoise(self, x_t, steps=None, condition=None, use_ddim=True, control=None, **kwargs):
        """去噪循环"""
        self_cond = None

        if use_ddim:
            steps = self.noise_scheduler.timesteps if steps is None else steps
            timesteps_array = torch.linspace(0, self.noise_scheduler.T-1, steps, dtype=torch.long, device=x_t.device)
        else:
            timesteps_array = self.noise_scheduler.timesteps_array[slice(0, steps)]
            
        st_prog_bar = st.progress(0)
        for i, t in tqdm(enumerate(reversed(timesteps_array))):
            st_prog_bar.progress((i + 1) / len(timesteps_array))

            # 准备 control residuals
            control_residuals = self._prepare_control(control, t.expand(x_t.shape[0]), condition)
            
            # UNet 预测
            x_t, x_0, x_T, self_cond = self(x_t, t.expand(x_t.shape[0]), condition, self_cond=self_cond, control_residuals=control_residuals, **kwargs)
            self_cond = self_cond if self.use_self_conditioning else None
        
            if use_ddim and (steps - i - 1 > 0):
                t_next = timesteps_array[steps - i - 2]
                alpha = self.noise_scheduler.alphas_cumprod[t]
                alpha_next = self.noise_scheduler.alphas_cumprod[t_next]
                sigma = kwargs.get('eta', 1) * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                c = (1 - alpha_next - sigma ** 2).sqrt()
                noise = torch.randn_like(x_t)
                x_t = x_0 * alpha_next.sqrt() + c * x_T + sigma * noise

        # 解码到图像空间
        if self.latent_embedder is not None:
            x_t = self.latent_embedder.decode(x_t)
        
        return x_t

    @torch.no_grad()
    def sample(self, num_samples, img_size, condition=None, control=None, **kwargs):
        """生成样本"""
        template = torch.zeros((num_samples, *img_size), device=self.device)
        x_T = self.noise_scheduler.x_final(template)
        x_0 = self.denoise(x_T, condition=condition, control=control, **kwargs)
        return x_0

    @torch.no_grad()
    def interpolate(self, img1, img2, i=None, condition=None, lam=0.5, **kwargs):
        """插值生成"""
        assert img1.shape == img2.shape, "Image 1 and 2 must have equal shape"

        t = self.noise_scheduler.T - 1 if i is None else i
        t = torch.full(img1.shape[:1], i, device=img1.device)

        img1_t = self.noise_scheduler.estimate_x_t(img1, t=t, clip_x0=self.clip_x0)
        img2_t = self.noise_scheduler.estimate_x_t(img2, t=t, clip_x0=self.clip_x0)

        img = (1 - lam) * img1_t + lam * img2_t
        img = self.denoise(img, i, condition, **kwargs)
        return img

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.ema_model.step(self.noise_estimator)
    
    def configure_optimizers(self):
        """配置优化器 - 同时优化 UNet 和 ControlNet"""
        # 收集所有需要训练的参数
        params = list(self.noise_estimator.parameters())
        if self.controlnet is not None:
            params += list(self.controlnet.parameters())
        
        optimizer = self.optimizer(params, **self.optimizer_kwargs)
        
        if self.lr_scheduler is not None:
            lr_scheduler = {
                'scheduler': self.lr_scheduler(optimizer, **self.lr_scheduler_kwargs),
                'interval': 'step',
                'frequency': 1
            }
            return [optimizer], [lr_scheduler]
        else:
            return [optimizer]

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "train", self.global_step, 0)

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "val", self.global_step, 0)

    def save_best_checkpoint(self, log_dir, best_model_path):
        """保存最佳模型路径到文件"""
        path_out = Path(log_dir) / 'best_model.txt'
        with open(path_out, 'w') as f:
            f.write(str(best_model_path))