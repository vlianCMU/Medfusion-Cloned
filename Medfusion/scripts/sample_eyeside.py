import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
import math
from typing import List, Optional, Dict
from torchvision import utils
from medical_diffusion.models.pipelines import DiffusionPipeline

# ===================== User Config =====================
CKPT_PATH = "/data1/lhy/medfusion-main/new_runs/2025_10_30_051150/lightning_logs/version_0/checkpoints/last.ckpt"
IMG_SIZE = (256, 256)
LATENT_SHAPE = (8, 32, 32)
N_SAMPLES = 16
STEPS = 150
USE_DDIM = True
GUIDANCE_SCALE = 1.0

# 确保与训练时的疾病数量一致
DISEASES = [
    "青光眼",
    "糖尿病性视网膜病变", 
    "年龄相关性黄斑变性",
    "白内障",
    "视网膜静脉阻塞",
]

# 眼别选项：0=左，1=右
EYE_SIDES = [0, 1]

# Define presets
COND_PRESETS: List[Dict] = [
    # 无条件生成（用于classifier-free guidance）
    {"label": "unconditional", "disease_vec": [0.0] * len(DISEASES), "eye_side": 0},
    
    # 单一疾病 + 右眼
    *[
        {
            "label": f"{name}_右眼",
            "disease_vec": [1.0 if i == j else 0.0 for j in range(len(DISEASES))],
            "eye_side": 0,
        }
        for i, name in enumerate(DISEASES)
    ],
    
    # 单一疾病 + 左眼
    *[
        {
            "label": f"{name}_左眼",
            "disease_vec": [1.0 if i == j else 0.0 for j in range(len(DISEASES))],
            "eye_side": 1,
        }
        for i, name in enumerate(DISEASES)
    ],
    
    # 多疾病组合示例
    {
        "label": "青光眼+AMD_左眼",
        "disease_vec": [1.0, 0.0, 1.0, 0.0, 0.0],  # 糖网和白内障
        "eye_side": 1,
    },
]

# =======================================================

def ensure_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_batch_disease_vec(vec: Optional[List[float]], batch: int, device: torch.device) -> Optional[torch.Tensor]:
    """将疾病向量扩展为batch"""
    if vec is None:
        return None
    # 确保是float32类型，与训练时一致
    t = torch.tensor(vec, dtype=torch.float32, device=device)
    if t.dim() == 1:
        t = t.unsqueeze(0)
    return t.repeat(batch, 1)

def to_batch_eye_side(side: Optional[int], batch: int, device: torch.device) -> Optional[torch.Tensor]:
    """将眼别标签扩展为batch"""
    if side is None:
        return None
    # 确保是long类型，用于Embedding层
    return torch.full((batch,), int(side), dtype=torch.long, device=device)

def save_grid(t: torch.Tensor, out_path: Path, nrow: int) -> None:
    """保存图像网格"""
    t = t.clamp(0, 1)
    utils.save_image(t, str(out_path), nrow=nrow, normalize=True, scale_each=True)

def main():
    current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    path_out = Path.cwd() / "generation_eyeside" / current_time
    path_out.mkdir(parents=True, exist_ok=True)
    
    device = ensure_device()
    torch.manual_seed(0)
    
    print(f"Loading pipeline from {CKPT_PATH} ...")
    pipeline = DiffusionPipeline.load_from_checkpoint(CKPT_PATH)
    pipeline.to(device)
    pipeline.eval()
    
    # 获取无条件的embedding（用于classifier-free guidance）
    uncond_disease_vec = to_batch_disease_vec([0.0] * len(DISEASES), N_SAMPLES, device)
    uncond_eye_side = to_batch_eye_side(0, N_SAMPLES, device)
    
    def run_sample(disease_vec, eye_side, label, use_cfg=True):
        """生成样本"""
        print(f"\nGenerating → {label}")
        print(f"  Disease vec shape: {disease_vec.shape}, dtype: {disease_vec.dtype}")
        print(f"  Eye side shape: {eye_side.shape}, dtype: {eye_side.dtype}")
        print(f"  Disease vec: {disease_vec[0].tolist()}")
        print(f"  Eye side: {eye_side[0].item()}")
        
        with torch.no_grad():
            # 根据是否使用classifier-free guidance设置un_cond
            un_cond = (uncond_disease_vec, uncond_eye_side) if use_cfg and GUIDANCE_SCALE > 1.0 else None
            
            samples = pipeline.sample(
                N_SAMPLES,
                LATENT_SHAPE,
                guidance_scale=GUIDANCE_SCALE,
                condition=(disease_vec, eye_side),
                un_cond=un_cond,
                steps=STEPS,
                use_ddim=USE_DDIM,
            )
            
            # 归一化到[0, 1]
            samples = (samples + 1.0) / 2.0
            save_grid(samples, path_out / f"{label}.png", nrow=int(math.sqrt(N_SAMPLES)))
    
    # 生成所有预设条件的样本
    for preset in COND_PRESETS:
        label = preset["label"]
        disease_vec = to_batch_disease_vec(preset["disease_vec"], N_SAMPLES, device)
        eye_side = to_batch_eye_side(preset["eye_side"], N_SAMPLES, device)
        
        run_sample(disease_vec, eye_side, label, use_cfg=(GUIDANCE_SCALE > 1.0))
    
    print(f"\n✅ All done. Images saved to: {path_out}")

if __name__ == "__main__":
    main()