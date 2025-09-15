from pathlib import Path
from datetime import datetime
import math
from typing import List, Optional, Dict

import torch
from torchvision import utils

# Import your diffusion pipeline (same as in training)
from medical_diffusion.models.pipelines import DiffusionPipeline

"""
Multi-label / multi-class sampling script for your Fundus diffusion model.
Fix: pipeline.sample requires `img_size` instead of `sample_shape`.
"""

# ===================== User Config =====================
CKPT_PATH = "/data1/lhy/medfusion-main/runs/2025_09_11_052246/lightning_logs/version_0/checkpoints/epoch=2-step=1400.ckpt"

# Output image size (decoded image size). Training used 256.
IMG_SIZE = (256, 256)
LATENT_SHAPE = (8, 32, 32)

N_SAMPLES = 16
STEPS = 150
USE_DDIM = True
GUIDANCE_SCALE = 1.0

MODE = "multilabel"  # "multilabel" or "singleclass"

DISEASES = [
    "青光眼",
    "糖尿病性视网膜病变",
    "年龄相关性黄斑变性",
    "病理性近视",
    "白内障",
    "视网膜静脉阻塞",
    "正常眼底",
]

COND_PRESETS: List[Dict] = [
    {"label": "unconditional", "vec": None},
    *[{"label": f"{name}", "vec": [1 if i == j else 0 for j in range(len(DISEASES))]} for i, name in enumerate(DISEASES)],
    {"label": "glaucoma+dr", "vec": [1,1,0,0,0,0,0]},
    {"label": "dr+amd", "vec": [0,1,1,0,0,0,0]},
]

SINGLECLASS_LIST = [0, 1, 2, 3, 4, 5, 6, None]

# =======================================================


def ensure_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_batch(vec: Optional[List[int]], batch: int, device: torch.device) -> Optional[torch.Tensor]:
    if vec is None:
        return None
    t = torch.tensor(vec, dtype=torch.float32, device=device).unsqueeze(0)
    t = t.repeat(batch, 1)
    return t


def to_batch_singleclass(label: Optional[int], batch: int, device: torch.device) -> Optional[torch.Tensor]:
    if label is None:
        return None
    return torch.full((batch,), int(label), dtype=torch.long, device=device)


def save_grid(t: torch.Tensor, out_path: Path, nrow: int) -> None:
    t = t.clamp(0, 1)
    utils.save_image(t, str(out_path), nrow=nrow, normalize=True, scale_each=True)


def main():
    current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    path_out = Path.cwd() / "generation" / current_time
    path_out.mkdir(parents=True, exist_ok=True)

    device = ensure_device()
    torch.manual_seed(0)

    pipeline = DiffusionPipeline.load_from_checkpoint(CKPT_PATH)
    pipeline.to(device)
    pipeline.eval()

    def run_sample(_condition: Optional[torch.Tensor], _uncond: Optional[torch.Tensor], label: str):
        with torch.no_grad():
            samples = pipeline.sample(
                N_SAMPLES,
                LATENT_SHAPE,
                guidance_scale=GUIDANCE_SCALE,
                condition=_condition,
                un_cond=_uncond,
                steps=STEPS,
                use_ddim=USE_DDIM,
            )
            samples = (samples + 1.0) / 2.0
            save_grid(samples, path_out / f"{label}.png", nrow=int(math.sqrt(N_SAMPLES)))

    if MODE == "multilabel":
        for preset in COND_PRESETS:
            label = preset["label"]
            cond_vec = preset["vec"]
            cond = to_batch(cond_vec, N_SAMPLES, device)
            uncond = None
            run_sample(cond, uncond, label)

    elif MODE == "singleclass":
        for cls in SINGLECLASS_LIST:
            label = "unconditional" if cls is None else f"class_{int(cls)}"
            cond = to_batch_singleclass(cls, N_SAMPLES, device)
            uncond = None
            run_sample(cond, uncond, label)

    else:
        raise ValueError("MODE must be 'multilabel' or 'singleclass'")

    print(f"Done. Images saved to: {path_out}")


if __name__ == "__main__":
    main()
