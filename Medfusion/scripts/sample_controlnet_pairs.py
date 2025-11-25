import argparse
import math
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
        description="Sample healthy–diseased fundus pairs using a ControlNet-trained checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ckpt", required=True, help="Path to the Lightning checkpoint containing the ControlNet-augmented pipeline.")
    parser.add_argument("--csv", required=True, help="CSV with conditioning labels and image paths.")
    parser.add_argument("--image-dir", required=True, help="Directory with fundus images.")
    parser.add_argument("--vessel-dir", required=True, help="Directory with vessel segmentation PNGs.")
    parser.add_argument("--disc-cup-dir", default=None, help="Directory with optic disc/cup segmentation PNGs (optional).")
    parser.add_argument("--output", default="controlnet_pairs", help="Where to save paired samples.")

    parser.add_argument("--latent-channels", type=int, default=8, help="Latent channels expected by the VAE/UNet.")
    parser.add_argument("--latent-height", type=int, default=32, help="Latent height.")
    parser.add_argument("--latent-width", type=int, default=32, help="Latent width.")

    parser.add_argument("--label-columns", nargs="*", default=None, help="Override label columns used for conditioning.")
    parser.add_argument(
        "--disease-columns",
        nargs="*",
        default=None,
        help="Subset of label columns to zero out for the healthy counterpart (defaults to all label columns).",
    )
    parser.add_argument("--image-column", default="img_path", help="CSV column that stores the image filename.")
    parser.add_argument("--image-size", type=int, default=256, help="Image resize used during training (applied to control maps).")
    parser.add_argument("--num-pairs", type=int, default=8, help="How many rows to sample from the CSV.")
    parser.add_argument("--steps", type=int, default=1000, help="Diffusion steps for sampling.")
    parser.add_argument("--guidance-scale", type=float, default=4.0, help="Classifier-free guidance scale.")
    parser.add_argument("--use-ddim", action="store_true", help="Enable DDIM sampling (eta=1).")
    return parser.parse_args()


def _prepare_conditions(labels: torch.Tensor, disease_idx: Sequence[int]):
    labels = labels.unsqueeze(0)
    healthy = labels.clone()
    if len(disease_idx):
        healthy[:, disease_idx] = 0.0
    return healthy, labels


def main():
    args = _parse_args()
    device = _device()

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

    label_columns: List[str] = dataset.label_columns
    disease_columns = args.disease_columns or label_columns
    disease_idx = [label_columns.index(col) for col in disease_columns]

    pipeline: DiffusionPipeline = DiffusionPipeline.load_from_checkpoint(args.ckpt, map_location=device)
    pipeline.eval()
    pipeline.to(device)

    latent_shape = (args.latent_channels, args.latent_height, args.latent_width)
    out_root = Path(args.output)
    (out_root / "healthy").mkdir(parents=True, exist_ok=True)
    (out_root / "diseased").mkdir(parents=True, exist_ok=True)

    for i in range(min(args.num_pairs, len(dataset))):
        sample = dataset[i]
        control = sample["control"].unsqueeze(0).to(device)
        healthy_cond, disease_cond = _prepare_conditions(sample["labels"].to(device), disease_idx)

        noise = torch.zeros((1, *latent_shape), device=device)
        x_T = pipeline.noise_scheduler.x_final(noise)

        with torch.no_grad():
            healthy_img = pipeline.denoise(
                x_T.clone(),
                condition=healthy_cond,
                control=control,
                steps=args.steps,
                use_ddim=args.use_ddim,
                guidance_scale=args.guidance_scale,
            )
            disease_img = pipeline.denoise(
                x_T.clone(),
                condition=disease_cond,
                control=control,
                steps=args.steps,
                use_ddim=args.use_ddim,
                guidance_scale=args.guidance_scale,
            )

        for img, split in ((healthy_img, "healthy"), (disease_img, "diseased")):
            img = (img + 1.0) / 2.0
            img = img.clamp(0, 1)
            name = f"{Path(sample['img_path']).stem}.png"
            vutils.save_image(img, out_root / split / name, nrow=int(math.sqrt(img.shape[0])), normalize=True, scale_each=True)

        print(f"Saved pair for {sample['img_path']}")

    print(f"Done. Results saved to: {out_root}")


if __name__ == "__main__":
    main()
