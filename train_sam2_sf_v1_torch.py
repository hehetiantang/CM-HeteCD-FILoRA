# train_sam2_sf_v1_torch.py
# Put this file in the root directory of your local CM-HeteCD-FILoRA project.
# It trains filora/cm_hetecd_sam2_sf_v1_torch.py with official SAM2 image encoder.
#
# Expected default dataset structure:
#   DATA_ROOT/
#     train/
#       A/ or optical/ or opt/
#       B/ or sar/
#       label/ or labels/ or mask/
#     val/
#       A/ or optical/ or opt/
#       B/ or sar/
#       label/ or labels/ or mask/
#     test/
#       A/ or optical/ or opt/
#       B/ or sar/
#       label/ or labels/ or mask/
#
# If your folders are different, use explicit args:
#   --train_opt_dir ... --train_sar_dir ... --train_label_dir ...
#   --val_opt_dir ...   --val_sar_dir ...   --val_label_dir ...
#   --test_opt_dir ...  --test_sar_dir ...  --test_label_dir ...

import argparse
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from filora.cm_hetecd_sam2_sf_v1_torch import (
    SAM2LoRASFChangeDetectorV1,
    ce_dice_loss,
    first_version_auxiliary_losses,
)


IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


# =============================
# 你可以直接在这里填写路径
# =============================
# 如果你的数据结构是：
# DATA_ROOT/train/A, DATA_ROOT/train/B, DATA_ROOT/train/label
# DATA_ROOT/val/A,   DATA_ROOT/val/B,   DATA_ROOT/val/label
# DATA_ROOT/test/A,  DATA_ROOT/test/B,  DATA_ROOT/test/label
# 那么只需要改 DATA_ROOT。
DATA_ROOT = "/data/sjh/data/XiongAn"

# SAM2 路径。sam2_cfg 是相对于 third_party/sam2 的 config 路径；
# sam2_ckpt 建议写绝对路径。
SAM2_REPO_DIR = "/data/sjh/study/test/third_party/sam2"
SAM2_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"
SAM2_CKPT = "/data/sjh/study/test/third_party/sam2/checkpoints/sam2.1_hiera_tiny.pt"

# 输出目录
SAVE_DIR = "./output_sam2_sf_v1"

# 如果你的文件夹不是 A/B/label，就在这里显式填写；否则保持 None。
TRAIN_OPT_DIR = None
TRAIN_SAR_DIR = None
TRAIN_LABEL_DIR = None
VAL_OPT_DIR = None
VAL_SAR_DIR = None
VAL_LABEL_DIR = None
TEST_OPT_DIR = None
TEST_SAR_DIR = None
TEST_LABEL_DIR = None

# 常用训练参数
IMG_SIZE = 512
BATCH_SIZE = 2
EPOCHS = 100
NUM_WORKERS = 4
LR = 5e-4
DECODER_DIM = 96
LORA_R = 4
LORA_ALPHA = 8


def set_seed(seed: int = 32767):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _find_first_existing(root: Path, names: List[str]) -> Optional[Path]:
    for name in names:
        p = root / name
        if p.exists() and p.is_dir():
            return p
    return None


def autodetect_split_dirs(data_root: str, split: str) -> Tuple[Path, Path, Path]:
    split_root = Path(data_root) / split
    if not split_root.exists():
        raise FileNotFoundError(f"Split directory not found: {split_root}")

    opt_dir = _find_first_existing(split_root, ["A", "a", "optical", "opt", "imageA", "img_a", "t1"])
    sar_dir = _find_first_existing(split_root, ["B", "b", "sar", "SAR", "imageB", "img_b", "t2"])
    lab_dir = _find_first_existing(split_root, ["label", "labels", "mask", "masks", "gt", "GT"])

    missing = []
    if opt_dir is None:
        missing.append("optical/A")
    if sar_dir is None:
        missing.append("sar/B")
    if lab_dir is None:
        missing.append("label")
    if missing:
        raise FileNotFoundError(
            f"Cannot autodetect {missing} under {split_root}. "
            "Use explicit --*_opt_dir --*_sar_dir --*_label_dir args."
        )
    return opt_dir, sar_dir, lab_dir


def list_image_files(folder: Path) -> Dict[str, Path]:
    files = {}
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            files[p.stem] = p
    return files


class HeteCDTorchDataset(Dataset):
    def __init__(
        self,
        opt_dir: str,
        sar_dir: str,
        label_dir: str,
        img_size: int = 512,
        augment: bool = False,
    ):
        self.opt_dir = Path(opt_dir)
        self.sar_dir = Path(sar_dir)
        self.label_dir = Path(label_dir)
        self.img_size = int(img_size)
        self.augment = augment

        opt_files = list_image_files(self.opt_dir)
        sar_files = list_image_files(self.sar_dir)
        lab_files = list_image_files(self.label_dir)

        names = sorted(set(opt_files.keys()) & set(sar_files.keys()) & set(lab_files.keys()))
        if not names:
            raise RuntimeError(
                f"No matched file stems found among:\n"
                f"  opt={self.opt_dir}\n  sar={self.sar_dir}\n  label={self.label_dir}"
            )

        self.items = [(opt_files[n], sar_files[n], lab_files[n], n) for n in names]

    def __len__(self):
        return len(self.items)

    @staticmethod
    def _read_rgb(path: Path) -> Image.Image:
        return Image.open(path).convert("RGB")

    @staticmethod
    def _read_label(path: Path) -> Image.Image:
        return Image.open(path).convert("L")

    def __getitem__(self, idx: int):
        opt_path, sar_path, lab_path, name = self.items[idx]
        opt = self._read_rgb(opt_path)
        sar = self._read_rgb(sar_path)
        lab = self._read_label(lab_path)

        size = (self.img_size, self.img_size)
        opt = opt.resize(size, Image.BILINEAR)
        sar = sar.resize(size, Image.BILINEAR)
        lab = lab.resize(size, Image.NEAREST)

        if self.augment:
            if random.random() < 0.5:
                opt = opt.transpose(Image.FLIP_LEFT_RIGHT)
                sar = sar.transpose(Image.FLIP_LEFT_RIGHT)
                lab = lab.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() < 0.5:
                opt = opt.transpose(Image.FLIP_TOP_BOTTOM)
                sar = sar.transpose(Image.FLIP_TOP_BOTTOM)
                lab = lab.transpose(Image.FLIP_TOP_BOTTOM)

        opt_np = np.asarray(opt).astype("float32") / 255.0
        sar_np = np.asarray(sar).astype("float32") / 255.0
        lab_np = np.asarray(lab)
        lab_np = (lab_np > 0).astype("int64")

        opt_t = torch.from_numpy(opt_np).permute(2, 0, 1).contiguous()
        sar_t = torch.from_numpy(sar_np).permute(2, 0, 1).contiguous()
        lab_t = torch.from_numpy(lab_np).contiguous()

        return {
            "optical": opt_t,
            "sar": sar_t,
            "label": lab_t,
            "name": name,
        }


@torch.no_grad()
def evaluate(model, loader, device: str):
    model.eval()
    tp = fp = fn = tn = 0
    for batch in tqdm(loader, desc="Eval", ncols=120):
        optical = batch["optical"].to(device, non_blocking=True)
        sar = batch["sar"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)

        logits = model(optical, sar, return_aux=False)
        pred = torch.argmax(logits, dim=1)

        pred_pos = pred == 1
        pred_neg = pred == 0
        lab_pos = label == 1
        lab_neg = label == 0

        tp += torch.logical_and(pred_pos, lab_pos).sum().item()
        fp += torch.logical_and(pred_pos, lab_neg).sum().item()
        fn += torch.logical_and(pred_neg, lab_pos).sum().item()
        tn += torch.logical_and(pred_neg, lab_neg).sum().item()

    eps = 1e-7
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    oa = (tp + tn) / (tp + tn + fp + fn + eps)
    return {"iou": iou, "f1": f1, "precision": precision, "recall": recall, "oa": oa}


def build_datasets(args):
    if args.train_opt_dir and args.train_sar_dir and args.train_label_dir:
        train_dirs = (args.train_opt_dir, args.train_sar_dir, args.train_label_dir)
    else:
        train_dirs = autodetect_split_dirs(args.data_root, "train")

    if args.val_opt_dir and args.val_sar_dir and args.val_label_dir:
        val_dirs = (args.val_opt_dir, args.val_sar_dir, args.val_label_dir)
    else:
        val_dirs = autodetect_split_dirs(args.data_root, "val")

    if args.test_opt_dir and args.test_sar_dir and args.test_label_dir:
        test_dirs = (args.test_opt_dir, args.test_sar_dir, args.test_label_dir)
    else:
        test_root = Path(args.data_root) / "test"
        test_dirs = autodetect_split_dirs(args.data_root, "test") if test_root.exists() else val_dirs

    train_set = HeteCDTorchDataset(*map(str, train_dirs), img_size=args.img_size, augment=True)
    val_set = HeteCDTorchDataset(*map(str, val_dirs), img_size=args.img_size, augment=False)
    test_set = HeteCDTorchDataset(*map(str, test_dirs), img_size=args.img_size, augment=False)
    return train_set, val_set, test_set


def parse_args():
    parser = argparse.ArgumentParser("SAM2-LoRA Spatial-Frequency Change Detection V1")

    parser.add_argument("--data_root", type=str, default=DATA_ROOT)
    parser.add_argument("--save_dir", type=str, default=SAVE_DIR)

    parser.add_argument("--sam2_repo_dir", type=str, default=SAM2_REPO_DIR)
    parser.add_argument("--sam2_cfg", type=str, default=SAM2_CFG)
    parser.add_argument("--sam2_ckpt", type=str, default=SAM2_CKPT)

    parser.add_argument("--img_size", type=int, default=IMG_SIZE)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--decoder_dim", type=int, default=DECODER_DIM)
    parser.add_argument("--lora_r", type=int, default=LORA_R)
    parser.add_argument("--lora_alpha", type=int, default=LORA_ALPHA)
    parser.add_argument("--lora_dropout", type=float, default=0.0)

    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--spa_weight", type=float, default=0.03)
    parser.add_argument("--freq_weight", type=float, default=0.02)
    parser.add_argument("--orth_weight", type=float, default=0.01)

    parser.add_argument("--device", type=str, default="cuda:3")
    parser.add_argument("--seed", type=int, default=32767)
    parser.add_argument("--amp", action="store_true")

    # Explicit folder overrides.
    parser.add_argument("--train_opt_dir", type=str, default=TRAIN_OPT_DIR)
    parser.add_argument("--train_sar_dir", type=str, default=TRAIN_SAR_DIR)
    parser.add_argument("--train_label_dir", type=str, default=TRAIN_LABEL_DIR)
    parser.add_argument("--val_opt_dir", type=str, default=VAL_OPT_DIR)
    parser.add_argument("--val_sar_dir", type=str, default=VAL_SAR_DIR)
    parser.add_argument("--val_label_dir", type=str, default=VAL_LABEL_DIR)
    parser.add_argument("--test_opt_dir", type=str, default=TEST_OPT_DIR)
    parser.add_argument("--test_sar_dir", type=str, default=TEST_SAR_DIR)
    parser.add_argument("--test_label_dir", type=str, default=TEST_LABEL_DIR)

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"

    train_set, val_set, test_set = build_datasets(args)
    print(f"Train samples: {len(train_set)}, Val samples: {len(val_set)}, Test samples: {len(test_set)}")

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = SAM2LoRASFChangeDetectorV1(
        sam2_repo_dir=args.sam2_repo_dir,
        sam2_cfg=args.sam2_cfg,
        sam2_ckpt=args.sam2_ckpt,
        img_size=(args.img_size, args.img_size),
        num_classes=args.num_classes,
        decoder_dim=args.decoder_dim,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        normalize_input=True,
        device=device,
    ).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params) / 1e6:.2f} M")
    print("Trainable parameter names:")
    for name, p in model.named_parameters():
        if p.requires_grad:
            print(name, tuple(p.shape))

    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.startswith("cuda"))

    best_iou = -1.0
    best_path = os.path.join(args.save_dir, "best_model.pth")
    last_path = os.path.join(args.save_dir, "last_model.pth")

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch}/{args.epochs}]", ncols=140)
        running = []

        for batch in pbar:
            optical = batch["optical"].to(device, non_blocking=True)
            sar = batch["sar"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=args.amp and device.startswith("cuda")):
                out = model(optical, sar, return_aux=True)
                logits = out["logits"]
                loss_cd = ce_dice_loss(logits, label)
                aux_losses = first_version_auxiliary_losses(
                    out["aux"],
                    lambda_spa=args.spa_weight,
                    lambda_freq=args.freq_weight,
                    lambda_orth=args.orth_weight,
                )
                loss = loss_cd + aux_losses["loss_aux"]

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_value = float(loss.detach().cpu())
            running.append(loss_value)
            pbar.set_postfix(
                loss=f"{loss_value:.4f}",
                cd=f"{float(loss_cd.detach().cpu()):.4f}",
                aux=f"{float(aux_losses['loss_aux'].detach().cpu()):.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

        scheduler.step()

        val_metrics = evaluate(model, val_loader, device)
        print(
            f"[VAL] epoch={epoch} loss={np.mean(running):.4f} "
            f"IoU={val_metrics['iou']:.4f} F1={val_metrics['f1']:.4f} "
            f"P={val_metrics['precision']:.4f} R={val_metrics['recall']:.4f} OA={val_metrics['oa']:.4f}"
        )

        torch.save({"model": model.state_dict(), "epoch": epoch, "args": vars(args)}, last_path)
        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "args": vars(args)}, best_path)
            print(f"[SAVE] best model saved to {best_path}, IoU={best_iou:.4f}")

    print("Loading best model for final test:", best_path)
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    test_metrics = evaluate(model, test_loader, device)
    print(
        f"[TEST] IoU={test_metrics['iou']:.4f} F1={test_metrics['f1']:.4f} "
        f"P={test_metrics['precision']:.4f} R={test_metrics['recall']:.4f} OA={test_metrics['oa']:.4f}"
    )


if __name__ == "__main__":
    main()
