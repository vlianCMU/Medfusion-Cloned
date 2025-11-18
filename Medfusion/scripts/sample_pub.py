import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
import math
from typing import List, Optional, Dict
from torchvision import utils
from medical_diffusion.models.pipelines import DiffusionPipeline

# ===================== User Config =====================
CKPT_PATH = "/data1/lhy/medfusion-main/new_pub/2025_11_11_193039/lightning_logs/version_0/checkpoints/last.ckpt"
IMG_SIZE = (256, 256)
LATENT_SHAPE = (8, 32, 32)
N_SAMPLES = 16
STEPS = 150
USE_DDIM = True
GUIDANCE_SCALE = 1.0  # >1 时启用 classifier-free guidance

# 条件标签定义（与你训练时一致）
LABELS = [
    "青光眼",
    "糖尿病性视网膜病变",
    "年龄相关性黄斑变性",
    "白内障",
    "视网膜静脉阻塞",
    "左右眼",  # 0=右, 1=左
]

# ===================== 条件预设 =====================
COND_PRESETS: List[Dict] = [
    # 无条件生成（用于 classifier-free guidance）
    {"label": "unconditional", "cond_vec": [0.0] * len(LABELS)},
    
    # 单一疾病 + 右眼
    *[
        {
            "label": f"{name}_左眼",
            "cond_vec": [1.0 if i == j else 0.0 for j in range(len(LABELS) - 1)] + [1.0],  # 最后一维是左眼=1
        }
        for i, name in enumerate(LABELS[:-1])
    ],
    
    # 单一疾病 + 左眼
    *[
        {
            "label": f"{name}_右眼",
            "cond_vec": [1.0 if i == j else 0.0 for j in range(len(LABELS) - 1)] + [0.0],  # 最后一维是右眼=0
        }
        for i, name in enumerate(LABELS[:-1])
    ]
]

# =======================================================

def ensure_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_batch_cond_vec(vec: List[float], batch: int, device: torch.device) -> torch.Tensor:
    """将标签向量扩展为 batch"""
    t = torch.tensor(vec, dtype=torch.float32, device=device)
    if t.dim() == 1:
        t = t.unsqueeze(0)
    return t.repeat(batch, 1)

def save_grid(t: torch.Tensor, out_path: Path, nrow: int) -> None:
    """保存图像网格"""
    t = t.clamp(0, 1)
    utils.save_image(t, str(out_path), nrow=nrow, normalize=True, scale_each=True)

def main():
    current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    path_out = Path.cwd() / "generation_pub" / current_time
    path_out.mkdir(parents=True, exist_ok=True)
    
    device = ensure_device()
    torch.manual_seed(0)
    
    print(f"Loading pipeline from {CKPT_PATH} ...")
    pipeline = DiffusionPipeline.load_from_checkpoint(CKPT_PATH)
    pipeline.to(device)
    pipeline.eval()
    
    # 无条件嵌入（用于 classifier-free guidance）
    uncond_vec = to_batch_cond_vec([0.0] * len(LABELS), N_SAMPLES, device)
    
    def run_sample(cond_vec: torch.Tensor, label: str, use_cfg: bool = True):
        """生成样本"""
        print(f"\nGenerating → {label}")
        print(f"  Condition vec shape: {cond_vec.shape}, dtype: {cond_vec.dtype}")
        print(f"  Condition vec: {cond_vec[0].tolist()}")
        
        with torch.no_grad():
            un_cond = uncond_vec if use_cfg and GUIDANCE_SCALE > 1.0 else None
            
            samples = pipeline.sample(
                N_SAMPLES,
                LATENT_SHAPE,
                guidance_scale=GUIDANCE_SCALE,
                condition=cond_vec,
                un_cond=un_cond,
                steps=STEPS,
                use_ddim=USE_DDIM,
            )
            
            samples = (samples + 1.0) / 2.0
            save_grid(samples, path_out / f"{label}.png", nrow=int(math.sqrt(N_SAMPLES)))
    
    # 执行所有预设
    for preset in COND_PRESETS:
        cond_vec = to_batch_cond_vec(preset["cond_vec"], N_SAMPLES, device)
        run_sample(cond_vec, preset["label"], use_cfg=(GUIDANCE_SCALE > 1.0))
    
    print(f"\n✅ All done. Images saved to: {path_out}")

if __name__ == "__main__":
    main()
