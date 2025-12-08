import argparse
import math
import shutil
from pathlib import Path
from typing import List, Sequence
import torch
from torchvision import utils as vutils
from PIL import Image
import numpy as np
from datetime import datetime

from medical_diffusion.data.datasets import FundusControlNetDataset
from medical_diffusion.models.pipelines import DiffusionPipeline


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate disease variants for normal fundus images using ControlNet.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ckpt", default="/data1/lhy/medfusion-main/controlnet_runs_new_new/2025_12_01_131134/lightning_logs/version_0/checkpoints/last.ckpt")
    parser.add_argument("--csv", default="/data1/lhy/GENFUND_600/genfund_conditions.csv")
    parser.add_argument("--image-dir", default="/data1/lhy/AutoMorph_copy/Results/M0/images")
    parser.add_argument("--vessel-dir", default="/data1/lhy/AutoMorph_copy/Results/M2/binary_vessel/raw_binary")
    parser.add_argument("--disc-cup-dir", default="/data1/lhy/AutoMorph_copy/Results/M2/optic_disc_cup/raw")
    parser.add_argument("--output", default="controlnet_pairs_new", help="Where to save paired samples")
    parser.add_argument("--latent-channels", type=int, default=8, help="Latent channels expected by the VAE/UNet")
    parser.add_argument("--latent-height", type=int, default=32, help="Latent height")
    parser.add_argument("--latent-width", type=int, default=32, help="Latent width")
    parser.add_argument("--label-columns", nargs="*", default=None, help="Override label columns used for conditioning")
    parser.add_argument("--image-column", default="img_path", help="CSV column that stores the image filename")
    parser.add_argument("--image-size", type=int, default=256, help="Image resize used during training")
    parser.add_argument("--num-pairs", type=int, default=40, help="How many normal images to process")
    parser.add_argument("--steps", type=int, default=150, help="Diffusion steps for sampling")
    parser.add_argument("--guidance-scale", type=float, default=1.0, help="Classifier-free guidance scale")
    parser.add_argument("--controlnet-scale", type=float, default=1.0, help="ControlNet influence scale")
    return parser.parse_args()


def _generate_disease_conditions():
    """
    生成所有疾病组合的条件
    返回: List[dict], 每个 dict 包含 'name' 和 'labels' (15维向量)
    
    label顺序: dr_grade, eye_side, is_amd, is_aon, is_crp, is_dm, is_dme, 
               is_em, is_gc, is_htr, is_pm, is_rvo, 
               is_fundus, is_optic_disc_readable, is_retinal_region_readable
    """
    disease_conditions = []
    
    # 2. DR grade 变体 (dr_grade: 1-4)
    for dr in range(1, 5):
        disease_conditions.append({
            'name': f'dr_grade_{dr}',
            'labels': [dr, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
        })
    
    # 3. 单一疾病变体
    diseases = [
        ('is_amd', 2),    # 年龄相关性黄斑变性
        ('is_aon', 3),    # 视神经萎缩
        ('is_crp', 4),    # 中心性浆液性脉络膜视网膜病变
        ('is_dm', 5),     # 糖尿病
        ('is_dme', 6),    # 糖尿病性黄斑水肿
        ('is_em', 7),     # 黄斑前膜
        ('is_gc', 8),     # 青光眼
        ('is_htr', 9),    # 高血压视网膜病变
        ('is_pm', 10),    # 病理性近视
        ('is_rvo', 11),   # 视网膜静脉阻塞
    ]
    
    for disease_name, disease_idx in diseases:
        labels = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
        labels[disease_idx] = 1
        disease_conditions.append({
            'name': disease_name,
            'labels': labels
        })
    
    
    return disease_conditions


def save_tensor_as_image(tensor: torch.Tensor, path: Path):
    """保存张量为图像"""
    # 确保是 [C, H, W] 格式
    if tensor.dim() == 4:
        tensor = tensor[0]
    
    # 反归一化 (从 [-1, 1] 到 [0, 1])
    tensor = (tensor + 1) / 2
    tensor = tensor.clamp(0, 1)
    
    # 转换为 numpy
    img_np = tensor.cpu().permute(1, 2, 0).numpy()
    img_np = (img_np * 255).astype(np.uint8)
    
    # 保存
    if img_np.shape[2] == 1:
        img_np = img_np[:, :, 0]
    Image.fromarray(img_np).save(path)


def main():
    args = _parse_args()
    device = _device()
    
    print(f"Loading dataset from {args.csv}...")
    # 加载数据集
    dataset = FundusControlNetDataset(
        csv_path=args.csv,
        image_dir=args.image_dir,
        vessel_dir=args.vessel_dir,
        disc_cup_dir=args.disc_cup_dir,
        crawler_ext="png",
        image_resize=args.image_size,
        image_column=args.image_column,
        label_columns=args.label_columns,
    )
    
    print(f"Loading model from {args.ckpt}...")
    # 加载模型
    pipeline: DiffusionPipeline = DiffusionPipeline.load_from_checkpoint(
        args.ckpt, 
        map_location=device
    )
    pipeline.eval()
    pipeline.to(device)
    
    # 设置 controlnet_scale
    pipeline.controlnet_scale = args.controlnet_scale
    
    latent_shape = (args.latent_channels, args.latent_height, args.latent_width)
    current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    out_root = Path(args.output) / current_time
    out_root.mkdir(parents=True, exist_ok=True)
    
    # 生成所有疾病条件
    disease_conditions = _generate_disease_conditions()
    
    print(f"Total disease conditions to generate: {len(disease_conditions)}")
    print(f"Processing {args.num_pairs} images...")
    
    for idx in range(min(args.num_pairs, len(dataset))):
        sample = dataset[idx]
        img_path = Path(sample['img_path'])
        image_id = img_path.stem
        
        print(f"\nProcessing image {idx + 1}/{args.num_pairs}: {image_id}")
        
        # 创建输出目录
        out_dir = out_root / image_id
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存原始图像
        source_img = sample['source']
        save_tensor_as_image(source_img, out_dir / 'original.png')
        
        # 保存控制图
        control_img = sample['control']
        save_tensor_as_image(control_img, out_dir / 'control.png')
        
        # 准备控制张量
        control = control_img.unsqueeze(0).to(device)
        
        # 为每个疾病条件生成图像
        for cond in disease_conditions:
            cond_name = cond['name']
            labels = torch.tensor([cond['labels']], dtype=torch.float32, device=device)
            
            print(f"  Generating: {cond_name}")
            
            # 生成图像
            with torch.no_grad():
                generated = pipeline.sample(
                    num_samples=1,
                    img_size=latent_shape,
                    condition=labels,
                    control=control,
                    steps=args.steps,
                    guidance_scale=args.guidance_scale,
                )
            
            # 保存生成的图像
            save_tensor_as_image(generated, out_dir / f'{cond_name}.png')
        
        print(f"  Saved to {out_dir}")
    
    print(f"\nDone! Results saved to {out_root}")


if __name__ == "__main__":
    main()