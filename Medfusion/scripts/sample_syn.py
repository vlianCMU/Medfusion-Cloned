import math
from pathlib import Path
from datetime import datetime
from typing import List, Dict

import torch
from torchvision import utils
from medical_diffusion.models.pipelines import DiffusionPipeline

# ===================== User Config =====================
CKPT_PATH = "/data1/lhy/medfusion-main/new_run/2025_11_05_074811/lightning_logs/version_0/checkpoints/last.ckpt"
LATENT_SHAPE = (8, 32, 32)
N_SAMPLES = 16
STEPS = 150
USE_DDIM = True
GUIDANCE_SCALE = 4.0  # >1 时启用 CFG

# === 必须与训练/数据集一致的字段顺序 ===
LABEL_COLUMNS = [
    'dr_grade', 'eye_side',
    'is_amd', 'is_aon', 'is_crp', 'is_dm', 'is_dme',
    'is_em', 'is_gc', 'is_htr', 'is_pm', 'is_rvo',
    'is_fundus', 'is_optic_disc_readable', 'is_retinal_region_readable'
]
# 除去前2个和最后3个，剩下的是“其它二分类疾病”
OTHER_DISEASES = LABEL_COLUMNS[2:-3]

# （可选）美化输出名字
HUMAN_READABLE = {
    'is_amd': 'AMD',
    'is_aon': 'AON',
    'is_crp': 'CRP',
    'is_dm': 'DM',
    'is_dme': 'DME',
    'is_em': 'EM',
    'is_gc': '青光眼',
    'is_htr': 'HTR',
    'is_pm': 'PM',
    'is_rvo': 'RVO',
    'normal': '正常',
    'eye0': '右眼',
    'eye1': '左眼',
}

QUALITY_TAIL = [1.0, 1.0, 1.0]  # 最后三个固定为 1

# =======================================================

def ensure_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_batch_cond_vec(vec: List[float], batch: int, device: torch.device) -> torch.Tensor:
    t = torch.tensor(vec, dtype=torch.float32, device=device)
    if t.dim() == 1:
        t = t.unsqueeze(0)
    return t.repeat(batch, 1)

def save_grid(t: torch.Tensor, out_path: Path, nrow: int) -> None:
    t = t.clamp(0, 1)
    utils.save_image(t, str(out_path), nrow=nrow, normalize=True, scale_each=True)

def main():
    current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    path_out = Path.cwd() / "generation_synfundus_single" / current_time
    path_out.mkdir(parents=True, exist_ok=True)

    device = ensure_device()
    torch.manual_seed(0)

    print(f"Loading pipeline from {CKPT_PATH} ...")
    pipeline = DiffusionPipeline.load_from_checkpoint(CKPT_PATH)
    pipeline.to(device)
    pipeline.eval()

    uncond_vec = to_batch_cond_vec([0.0] * len(LABEL_COLUMNS), N_SAMPLES, device)

    def run_sample(cond_vec: torch.Tensor, label: str, use_cfg: bool = True):
        print(f"\nGenerating → {label}")
        print(f"  Condition vec[0]: {cond_vec[0].tolist()}")
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

    # ===== 构造并生成：每只眼单独生成 若干“类别” =====
    # 1) 正常（dr_grade=0，所有疾病=0）
    for eye_side in [0, 1]:  # 0=右眼, 1=左眼
        cond = [0.0, float(eye_side)]  # dr_grade=0, eye_side
        cond += [0.0] * len(OTHER_DISEASES)  # 其它疾病全 0
        cond += QUALITY_TAIL
        label = f"{HUMAN_READABLE['normal']}_{HUMAN_READABLE[f'eye{eye_side}']}"
        run_sample(to_batch_cond_vec(cond, N_SAMPLES, device), label, use_cfg=(GUIDANCE_SCALE > 1.0))

    # 2) DR：作为独立类别，dr_grade ∈ {1,2,3,4}；不与其它疾病叠加
    for eye_side in [0, 1]:
        for dr in [1, 2, 3, 4]:
            cond = [float(dr), float(eye_side)]          # 指定 dr_grade、eye_side
            cond += [0.0] * len(OTHER_DISEASES)          # 其它疾病全 0
            cond += QUALITY_TAIL
            label = f"DR{dr}_{HUMAN_READABLE[f'eye{eye_side}']}"
            run_sample(to_batch_cond_vec(cond, N_SAMPLES, device), label, use_cfg=(GUIDANCE_SCALE > 1.0))

    # 3) 其它二分类疾病：逐个激活为 1；dr_grade=0
    for eye_side in [0, 1]:
        for i, dis_key in enumerate(OTHER_DISEASES):
            disease_vec = [0.0] * len(OTHER_DISEASES)
            disease_vec[i] = 1.0
            cond = [0.0, float(eye_side)] + disease_vec + QUALITY_TAIL  # dr=0，其它病 one-hot
            human_name = HUMAN_READABLE.get(dis_key, dis_key)
            label = f"{human_name}_{HUMAN_READABLE[f'eye{eye_side}']}"
            run_sample(to_batch_cond_vec(cond, N_SAMPLES, device), label, use_cfg=(GUIDANCE_SCALE > 1.0))

    print(f"\n✅ All done. Images saved to: {path_out}")

if __name__ == "__main__":
    main()
