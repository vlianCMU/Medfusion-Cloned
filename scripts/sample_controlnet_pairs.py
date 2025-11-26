import argparse
import math
import shutil
from pathlib import Path
from typing import List, Sequence
import torch
from torchvision import utils as vutils
from medical_diffusion.data.datasets import FundusControlNetDataset
from medical_diffusion.models.pipelines import DiffusionPipeline


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate disease variants for normal fundus images using ControlNet.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ckpt", default="/data1/lhy/medfusion-main/controlnet_runs/2025_11_26_092013/lightning_logs/version_0/checkpoints/last.ckpt")
    parser.add_argument("--csv", default="/data1/lhy/GENFUND_600/genfund_conditions.csv")
    parser.add_argument("--image-dir", default="/data1/lhy/AutoMorph_copy/Results/M0/images")
    parser.add_argument("--vessel-dir", default="/data1/lhy/AutoMorph_copy/Results/M2/binary_vessel/raw_binary")
    parser.add_argument("--disc-cup-dir", default="/data1/lhy/AutoMorph_copy/Results/M2/optic_disc_cup/raw")
    parser.add_argument("--output", default="controlnet_pairs", help="Where to save paired samples.")
    parser.add_argument("--latent-channels", type=int, default=8, help="Latent channels expected by the VAE/UNet.")
    parser.add_argument("--latent-height", type=int, default=32, help="Latent height.")
    parser.add_argument("--latent-width", type=int, default=32, help="Latent width.")
    parser.add_argument("--label-columns", nargs="*", default=None, help="Override label columns used for conditioning.")
    parser.add_argument("--image-column", default="img_path", help="CSV column that stores the image filename.")
    parser.add_argument("--image-size", type=int, default=256, help="Image resize used during training (applied to control maps).")
    parser.add_argument("--num-pairs", type=int, default=40, help="How many normal images to process.")
    parser.add_argument("--steps", type=int, default=150, help="Diffusion steps for sampling.")
    parser.add_argument("--guidance-scale", type=float, default=1.0, help="Classifier-free guidance scale.")
    return parser.parse_args()


def _generate_disease_conditions():
    """
    生成所有疾病组合的条件
    返回: List[dict], 每个 dict 包含 'name' 和 'labels' (15维向量)
    """
    disease_conditions = []
    
    # 固定的标签 (is_fundus=1, is_optic_disc_readable=1, is_retinal_region_readable=1)
    # label顺序: dr_grade, eye_side, is_amd, is_aon, is_crp, is_dm, is_dme, 
    #           is_em, is_gc, is_htr, is_pm, is_rvo, 
    #           is_fundus, is_optic_disc_readable, is_retinal_region_readable
    
    # 1. 保留原始 normal (所有疾病=0)
    disease_conditions.append({
        'name': 'normal',
        'labels': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
        # dr_grade=0, 所有疾病=0, 固定标签=1
    })
    
    # 2. DR grade 变体 (dr_grade: 1-4)
    for dr in range(1, 5):
        disease_conditions.append({
            'name': f'dr_grade_{dr}',
            'labels': [dr, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
        })
    
    # 3. 单一疾病变体
    diseases = [
        ('is_amd', 2),
        ('is_aon', 3),
        ('is_crp', 4),
        ('is_dm', 5),
        ('is_dme', 6),
        ('is_em', 7),
        ('is_gc', 8),
        ('is_htr', 9),
        ('is_pm', 10),
        ('is_rvo', 11),
    ]
    
    for disease_name, disease_idx in diseases:
        labels = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
        labels[disease_idx] = 1
        disease_conditions.append({
            'name': disease_name,
            'labels': labels
        })
    
    # 4. 如果需要,可以添加组合疾病
    # 例: DR + DM
    disease_conditions.append({
        'name': 'dr3_dm',
        'labels': [3, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1]
    })
    
    # 例: DR + DME
    disease_conditions.append({
        'name': 'dr3_dme',
        'labels': [3, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1]
    })
    
    return disease_conditions


def main():
    args = _parse_args()
    device = _device()
    
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
    
    # 加载模型
    pipeline: DiffusionPipeline = DiffusionPipeline.load_from_checkpoint(args.ckpt, map_location=device)
    pipeline.eval()
    pipeline.to(device)
    
    latent_shape = (args.latent_channels, args.latent_height, args.latent_width)
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)
    
    # 生成所有疾病条件
    disease_conditions = _generate_disease_conditions()
    
    print(f"Total disease conditions to generate: {len(disease_conditions)}")
    print(f"Processing {min(args.num_pairs, len(dataset))} normal images...")
    
    # 对每张 normal 图生成一整套疾病变体
    for i in range(min(args.num_pairs, len(dataset))):
        sample = dataset[i]
        img_name = Path(sample['img_path']).stem
        
        # 为当前图像创建文件夹
        img_folder = out_root / img_name
        img_folder.mkdir(parents=True, exist_ok=True)
        
        # 拷贝原始 normal 图像
        src_img = Path(sample['img_path'])
        dst_img = img_folder / f"{img_name}.png"
        if src_img.exists():
            shutil.copy2(src_img, dst_img)
            print(f"\n[{i+1}/{min(args.num_pairs, len(dataset))}] Processing {img_name}")
            print(f"  ✓ Copied original image to {dst_img}")
        else:
            print(f"\n[{i+1}/{min(args.num_pairs, len(dataset))}] Processing {img_name}")
            print(f"  ⚠ Original image not found: {src_img}")
        
        # 获取 control
        control = sample["control"].unsqueeze(0).to(device)
        
        # 获取原始的 eye_side (index=1)
        original_eye_side = sample["labels"][1].item()
        
        # 为当前图像生成所有疾病变体
        for disease_cond in disease_conditions:
            disease_name = disease_cond['name']
            disease_labels = disease_cond['labels'].copy()
            
            # 保持原始的 eye_side
            disease_labels[1] = original_eye_side
            
            # 转换为 tensor
            condition = torch.tensor([disease_labels], dtype=torch.float32).to(device)
            
            # 生成图像
            noise = torch.randn((1, *latent_shape), device=device)  # 使用随机噪声
            x_T = pipeline.noise_scheduler.x_final(noise)
            
            with torch.no_grad():
                generated_img = pipeline.denoise(
                    x_T,
                    condition=condition,
                    control=control,
                    steps=args.steps,
                    use_ddim=True,
                    guidance_scale=args.guidance_scale,
                )
            
            # 保存生成的图像
            generated_img = (generated_img + 1.0) / 2.0
            generated_img = generated_img.clamp(0, 1)
            
            save_path = img_folder / f"{disease_name}.png"
            vutils.save_image(
                generated_img, 
                save_path, 
                nrow=1, 
                normalize=False
            )
            
            print(f"  ✓ Generated {disease_name}")
        
        print(f"  Done! All variants saved to {img_folder}")
    
    print(f"\n{'='*60}")
    print(f"✅ All done! Results saved to: {out_root}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()