#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import os
os.environ["TORCH_HOME"] = os.environ.get("TORCH_HOME", "/tmp/torchhome")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score  # ← 新增
import matplotlib.pyplot as plt

from datetime import datetime
import pytz
from math import erf, sqrt  # 用于正态分布CDF

# 你的项目
from medical_diffusion.models.pipelines import DiffusionPipeline

import inspect

def estimate_x_t_compat(scheduler, x0, t_tensor, clip_flag: bool):
    """兼容不同版本的 GaussianNoiseScheduler.estimate_x_t 签名"""
    try:
        sig = inspect.signature(scheduler.estimate_x_t)
        if "clip_x0" in sig.parameters:
            return scheduler.estimate_x_t(x0, t=t_tensor, clip_x0=clip_flag)
        else:
            return scheduler.estimate_x_t(x0, t=t_tensor)
    except (TypeError, ValueError):
        try:
            return scheduler.estimate_x_t(x0, t=t_tensor, clip_x0=clip_flag)
        except TypeError:
            return scheduler.estimate_x_t(x0, t=t_tensor)

CLS_NAMES = ['anodr','bmilddr','cmoderatedr','dseveredr','eproliferativedr']
# CLS_NAMES = ['anormal','bsuspectglaucoma','cglaucoma']
# CLS_NAMES = ['hypertension','cataract','other', 'amd', 'normal', 'diabetes', 'myopia', 'glaucoma']
# CLS_NAMES = ['ageDegeneration','hypertension','cataract','other', 'normal', 'diabetes', 'myopia', 'glaucoma']
CLS2IDX = {c:i for i,c in enumerate(CLS_NAMES)}

# ---------------------- Dataset ----------------------
class APTOSDataset(Dataset):
    def __init__(self, split_dir, img_size=256):
        self.paths, self.labels = [], []
        for c in CLS_NAMES:
            cdir = Path(split_dir)/c
            for p in sorted(cdir.glob("*.png")): ## 这里可能要改 lhy
                self.paths.append(str(p)); self.labels.append(CLS2IDX[c])
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),   # [0,1]
        ])
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        x = self.transform(img)
        y = self.labels[idx]
        return {"image": x, "label": y, "path": self.paths[idx]}

# ---------------------- Classifier ----------------------
class UNetBottleneckClassifier(nn.Module):
    """
    在 noise_estimator.middle_block 注册 forward hook，截获 bottleneck 特征；
    对每个指定的时间步 t，前向一次 UNet，取 middle_block 输出 → 1x1 Conv 降到256 → GAP → LN；
    将两个时间步的向量拼接后送入 MLP 分类头。
    """
    def __init__(self, pipeline, t_steps=(50,500), mode="freeze", bottleneck_ch=None):
        super().__init__()
        assert mode in ("freeze","finetune")
        self.pipeline = pipeline.eval() if mode=="freeze" else pipeline.train()
        self.t_steps = tuple(int(x) for x in t_steps)
        self.mode = mode

        self._last_feat = None
        self._hook = self.pipeline.noise_estimator.middle_block.register_forward_hook(self._hook_fn)

        self.reduce_conv = None
        self.out_dim = 256

        self.head = nn.Sequential(
            nn.Linear(self.out_dim * len(self.t_steps), 512),
            nn.LayerNorm(512),  # 替换为LayerNorm提升稳定性
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),    # 增加dropout防止过拟合
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, len(CLS_NAMES))
        )

        if mode=="freeze":
            for p in self.pipeline.noise_estimator.parameters():
                p.requires_grad = False

        self._bottleneck_ch_fixed = bottleneck_ch

    def _hook_fn(self, mod, inp, out):
        if torch.is_tensor(out):
            self._last_feat = out
        elif isinstance(out, (list, tuple)) and torch.is_tensor(out[0]):
            self._last_feat = out[0]
        else:
            self._last_feat = None

    @torch.no_grad()
    def _encode_to_latent(self, x):
        if self.pipeline.latent_embedder is not None:
            self.pipeline.latent_embedder.eval()
            z = self.pipeline.latent_embedder.encode(x)
        else:
            z = x
        if getattr(self.pipeline, "do_input_centering", False):
            z = 2*z - 1
        return z

    def _ensure_reduce_conv(self, ch, device):
        if self.reduce_conv is None:
            self.reduce_conv = nn.Sequential(
                nn.Conv2d(ch, self.out_dim, kernel_size=1, bias=False),
                nn.GroupNorm(32, self.out_dim),
                nn.SiLU()
            ).to(device)

    @torch.no_grad()
    def warmup_build(self, sample_images):
        device = next(self.pipeline.parameters()).device
        x = sample_images.to(device)
        z0 = self._encode_to_latent(x)
        t0 = torch.full((x.size(0),), int(self.t_steps[0]), dtype=torch.long, device=device)
        x_t = estimate_x_t_compat(
            self.pipeline.noise_scheduler, z0, t0, getattr(self.pipeline, "clip_x0", True)
        )
        with torch.no_grad():
            _pred, _ = self.pipeline.noise_estimator(x_t, t0, condition=None, self_cond=None)
        assert self._last_feat is not None, "未捕获到 middle_block 特征，请检查 hook。"
        ch = self._last_feat.size(1)
        if self._bottleneck_ch_fixed is not None and self._bottleneck_ch_fixed != ch:
            raise ValueError(f"给定 bottleneck_ch={self._bottleneck_ch_fixed} 与实际 {ch} 不一致。")
        self._ensure_reduce_conv(ch, device)
        self._last_feat = None

    def forward(self, x):
        device = x.device
        z0 = self._encode_to_latent(x)
        vecs = []
        for t_val in self.t_steps:
            t_tensor = torch.full((x.size(0),), int(t_val), dtype=torch.long, device=device)
            x_t = estimate_x_t_compat(
                self.pipeline.noise_scheduler, z0, t_tensor, getattr(self.pipeline, "clip_x0", True)
            )
            self._last_feat = None
            if self.mode == "freeze":
                with torch.no_grad():
                    _pred, _ = self.pipeline.noise_estimator(x_t, t_tensor, condition=None, self_cond=None)
            else:
                _pred, _ = self.pipeline.noise_estimator(x_t, t_tensor, condition=None, self_cond=None)

            feat = self._last_feat
            assert feat is not None, "未捕获到 middle_block 特征，请检查 hook。"
            if self.reduce_conv is None:
                self._ensure_reduce_conv(feat.size(1), device)
            feat256 = self.reduce_conv(feat)
            gap = F.adaptive_avg_pool2d(feat256, 1).flatten(1)
            gap = F.normalize(gap, p=2, dim=1) 
            vecs.append(gap)
        z = torch.cat(vecs, dim=1)
        logits = self.head(z)
        return logits

    def remove_hook(self):
        if self._hook is not None:
            self._hook.remove()
            self._hook = None

# ---------------------- Utils (args写入 & AUC p值) ----------------------
def write_args_txt(out_dir: Path, args: argparse.Namespace):
    out_dir.mkdir(parents=True, exist_ok=True)
    args_path = out_dir / "args.txt"
    with args_path.open("w", encoding="utf-8") as f:
        for k in sorted(vars(args).keys()):
            f.write(f"{k} = {getattr(args, k)}\n")

def norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))

def auc_var_hanley(auc: float, n_pos: int, n_neg: int) -> float:
    auc = min(max(auc, 1e-6), 1 - 1e-6)
    n_pos = max(int(n_pos), 1)
    n_neg = max(int(n_neg), 1)
    Q1 = auc / (2 - auc)
    Q2 = 2 * auc * auc / (1 + auc)
    var = (auc * (1 - auc) + (n_pos - 1) * (Q1 - auc * auc) + (n_neg - 1) * (Q2 - auc * auc)) / (n_pos * n_neg)
    return max(var, 1e-12)

def format_p_str(p: float) -> str:
    return "P < 0.001" if p < 0.001 else f"P = {p:.3f}"

# ---------------------- Train / Eval / Plot ----------------------
def train_one_epoch(model, loader, optim, device):
    model.train()
    ce = nn.CrossEntropyLoss()
    total, cnt = 0.0, 0
    for batch in tqdm(loader, desc="Train", leave=False):
        x = batch["image"].to(device)
        y = batch["label"].to(device)
        optim.zero_grad()
        logits = model(x)
        loss = ce(logits, y)
        loss.backward()
        optim.step()
        total += loss.item()*x.size(0); cnt += x.size(0)
    return total/cnt

@torch.no_grad()
def eval_one_epoch(model, loader, device):
    model.eval()
    ce = nn.CrossEntropyLoss()
    total, cnt = 0.0, 0
    all_probs, all_labels = [], []
    for batch in tqdm(loader, desc="Eval", leave=False):
        x = batch["image"].to(device)
        y = batch["label"].to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(y.cpu().numpy())
        loss = ce(logits, y)
        total += loss.item()*x.size(0); cnt += x.size(0)
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return total/cnt, all_probs, all_labels

def plot_and_save_losses(train_losses, val_losses, out_dir, tag=""):
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(train_losses, label="train_loss")
    plt.plot(val_losses, label="val_loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
    plt.title(f"Train/Val Loss {tag}")
    plt.tight_layout()
    plt.savefig(out_dir/f"loss_curve{'_'+tag if tag else ''}.png", dpi=150)
    plt.close()

def plot_roc_ovr_and_auc(probs, labels, out_dir, tag=""):
    """
    在测试集上画 OvR ROC：
    - 图例显示每类 AUC；
    - 标题显示 Macro AUC 和总体 P（Stouffer 合并各类 z 值；H0: AUC=0.5）。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    y_true = np.eye(len(CLS_NAMES))[labels]  # one-hot

    aucs, zs, ws = [], [], []
    plt.figure()
    for k in range(len(CLS_NAMES)):
        fpr, tpr, _ = roc_curve(y_true[:,k], probs[:,k])
        auc_k = roc_auc_score(y_true[:,k], probs[:,k])
        aucs.append(auc_k)
        n_pos = int(y_true[:,k].sum())
        n_neg = int(y_true.shape[0] - n_pos)
        var_k = auc_var_hanley(auc_k, n_pos, n_neg)
        z_k = (auc_k - 0.5) / sqrt(var_k)
        w_k = sqrt(max(n_pos * n_neg, 1))
        zs.append(z_k); ws.append(w_k)
        plt.plot(fpr, tpr, label=f"{CLS_NAMES[k]} (AUC={auc_k:.3f})")

    macro_auc = float(np.mean(aucs))
    # Stouffer 合并 z：z_comb = sum(w_i z_i)/sqrt(sum w_i^2)
    if len(zs) > 0:
        denom = sqrt(sum(w*w for w in ws)) if sum(w*w for w in ws) > 0 else 1.0
        z_comb = sum(w*z for w, z in zip(ws, zs)) / denom
        p_total = 2.0 * (1.0 - norm_cdf(abs(z_comb)))
    else:
        p_total = 1.0
    p_str = format_p_str(float(p_total))

    print(f"[ROC-OvR][TEST] Macro AUROC = {macro_auc:.4f}, {p_str}")

    plt.plot([0,1],[0,1],'--')
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title(f"ROC (OvR){' - '+tag if tag else ''} | Macro AUC={macro_auc:.3f}, {p_str}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir/f"roc_ovr{'_'+tag if tag else ''}.png", dpi=150)
    plt.close()
    return macro_auc, float(p_total)

# ===== 新增：混淆矩阵 + Macro Precision/Recall/F1  =====
def plot_confusion_matrix_with_prf(labels, probs, out_dir: Path, classes, tag=""):
    """
    使用测试集预测结果绘制混淆矩阵，并在图中写入 Macro Precision/Recall/F1（小数三位）。
    仅新增，不影响其它流程。
    """
    y_true = np.asarray(labels).astype(int)
    y_pred = np.asarray(probs).argmax(axis=1)

    # 混淆矩阵（按类别顺序保证轴对齐）
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))), normalize='true')

    # 计算 Macro 指标
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro    = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro        = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # 画图
    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('Count', rotation=90, va='center')

    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        ylabel='True label',
        xlabel='Predicted label',
        title=f"Confusion Matrix{' - '+tag if tag else ''}"
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # 每格写入数字
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:.2f}",
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    # 在图内写 Macro 指标（不遮挡格子，放在坐标轴内右侧空白区域）
    text_str = (
        f"Macro Precision = {precision_macro:.3f}\n"
        f"Macro Recall    = {recall_macro:.3f}\n"
        f"Macro F1        = {f1_macro:.3f}"
    )
    # 放在坐标轴外侧右边缘（仍在整张图中）
    ax.text(1.02, 0.5, text_str, transform=ax.transAxes, va='center', ha='left',
            fontsize=10, bbox=dict(boxstyle="round", facecolor="w", alpha=0.85))

    fig.tight_layout()
    fig.savefig(out_dir / f"confusion_matrix{'_'+tag if tag else ''}.png", dpi=150)
    plt.close(fig)

    # 控制台也打印一下，便于日志查看
    print(f"[CONFUSION][TEST] Macro Precision={precision_macro:.4f}  "
          f"Macro Recall={recall_macro:.4f}  Macro F1={f1_macro:.4f}")

    return float(precision_macro), float(recall_macro), float(f1_macro)

# ---------------------- Main ----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, choices=["freeze","finetune"], default="finetune",
                    help="freeze=冻结 UNet 只训练头；finetune=连同 UNet 一起微调")
    # ap.add_argument("--ckpt", type=str, default="/data1/lhy/medfusion-main/runs/2025_09_10_141511没加公开数据集/lightning_logs/version_0/checkpoints/epoch=23-step=5500.ckpt")
    # ap.add_argument("--ckpt", type=str, default="/data1/lhy/medfusion-main/runs/2025_09_11_052246加了公开数据集/lightning_logs/version_0/checkpoints/last.ckpt")
    # ap.add_argument("--ckpt", type=str, default="/data1/lhy/medfusion-main/runs/2025_09_15_152118没加公开数据集在典型上finetune/lightning_logs/version_0/checkpoints/last.ckpt")
    # ap.add_argument("--ckpt", type=str, default="/data1/lhy/medfusion-main/finetune/2025_10_15_030800_finetune_unet/lightning_logs/version_0/checkpoints/last.ckpt")
    ap.add_argument("--ckpt", type=str, default="/data1/lhy/medfusion-main/new_run/2025_11_05_074811/lightning_logs/version_0/checkpoints/last.ckpt")
    # ap.add_argument("--data_root", type=str, default="/data2/ophthalmology/APTOS2019")
    ap.add_argument("--data_root", type=str, default="/data2/ophthalmology/MESSIDOR2")
    # ap.add_argument("--data_root", type=str, default="/data2/ophthalmology/PAPILA")
    # ap.add_argument("--data_root", type=str, default="/data1/lhy/RETFound/AOD")
    # ap.add_argument("--data_root", type=str, default="/data1/lhy/RETFound/ODIR-5K")
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr_unet", type=float, default=2e-4, help="仅 finetune 模式使用")
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--t_steps", type=int, nargs=2, default=[10, 750])
    ap.add_argument("--out_dir", type=str, default="/data1/lhy/medfusion-main/classification_syn")
    ap.add_argument("--bottleneck-ch", type=int, default=1024,
                    help="若已知 middle_block 通道数（你的模型是 1024），可显式给出；否则脚本会 warm-up 自动探测。")
    ap.add_argument("--decay_at7", action="store_true",
                help="仅 freeze 模式使用；是否在 epoch 7 时将 lr_head 降 10 倍")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tz = pytz.timezone("Asia/Shanghai")
    time_str = datetime.now(tz).strftime("%Y%m%d%H%M%S")
    sub_dir = Path(args.out_dir) / time_str

    out_dir = Path(sub_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_args_txt(out_dir, args)

    # 1) 加载 pipeline
    pipeline = DiffusionPipeline.load_from_checkpoint(args.ckpt, map_location=device)
    pipeline = pipeline.to(device)

    # 2) 数据
    train_ds = APTOSDataset(Path(args.data_root)/"train", img_size=args.img_size)
    val_ds   = APTOSDataset(Path(args.data_root)/"val",   img_size=args.img_size)
    test_ds  = APTOSDataset(Path(args.data_root)/"test",  img_size=args.img_size)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=8, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    # 3) 构建模型
    model = UNetBottleneckClassifier(
        pipeline=pipeline,
        t_steps=args.t_steps,
        mode=args.mode,
        bottleneck_ch=args.bottleneck_ch
    ).to(device)

    # 3.1 warm-up
    with torch.no_grad():
        warm_batch = next(iter(DataLoader(train_ds, batch_size=2, shuffle=True)))
        model.warmup_build(warm_batch["image"].to(device))

    # 4) 构建优化器
    if args.mode == "freeze":
        head_params = list(model.reduce_conv.parameters()) + list(model.head.parameters())
        optimizer = torch.optim.SGD(head_params, lr=args.lr_head, momentum=0.9, weight_decay=args.weight_decay)
        decay_at7 = True
        tag = "freeze"
    else:
        head_params = list(model.reduce_conv.parameters()) + list(model.head.parameters())
        unet_params = list(model.pipeline.noise_estimator.parameters())
        optimizer = torch.optim.SGD([
            {"params": unet_params, "lr": args.lr_unet},
            {"params": head_params, "lr": args.lr_head},
        ], momentum=0.9, weight_decay=args.weight_decay)
        decay_at7 = True
        tag = "finetune"

    # 5) 训练
    best_val = 1e9
    train_losses, val_losses = [], []
    best_path = out_dir/f"{tag}_unet_best.pth"
    for epoch in range(1, args.epochs+1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_probs, val_labels = eval_one_epoch(model, val_loader, device)
        train_losses.append(tr_loss); val_losses.append(val_loss)
        print(f"[{tag}][Epoch {epoch}] train_loss={tr_loss:.4f}  val_loss={val_loss:.4f}")

        if decay_at7 and epoch == 20:
            for pg in optimizer.param_groups: pg["lr"] *= 0.25

        if val_loss < best_val:
            best_val = val_loss
            out_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), best_path)

    # 6) 损失曲线（保持不变）
    plot_and_save_losses(train_losses, val_losses, out_dir, tag=tag)

    # 7) Test（在测试集上评估并作图）
    model.load_state_dict(torch.load(best_path, map_location=device))
    test_loss, test_probs, test_labels = eval_one_epoch(model, test_loader, device)
    test_macro_auc = roc_auc_score(
        np.eye(len(CLS_NAMES))[test_labels], test_probs, average="macro", multi_class="ovr"
    )
    print(f"[{tag}][TEST] loss={test_loss:.4f}  macro_AUROC(OvR)={test_macro_auc:.4f}")

    # 绘制混淆矩阵并在图中写 Macro Precision/Recall/F1（新增）
    plot_confusion_matrix_with_prf(
        test_labels, test_probs, out_dir, CLS_NAMES, tag=tag
    )

    # 在测试集上画 ROC，并在标题中显示 Macro AUC 与总体 P 值（保持不变）
    plot_roc_ovr_and_auc(test_probs, test_labels, out_dir, tag=tag)

    # 清理 hook（可选）
    model.remove_hook()

if __name__ == "__main__":
    main()