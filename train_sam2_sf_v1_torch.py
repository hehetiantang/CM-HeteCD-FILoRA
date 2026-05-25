# train_sam2_sf_v1_torch.py
# Put this file in the root directory of your local CM-HeteCD-FILoRA project.
# It trains filora/cm_hetecd_sam2_sf_v1_torch.py with official SAM2 image encoder.
#
# HeteCD-style train augmentation enabled by default:
#   random horizontal flip
#   random vertical flip
#   scale random crop, scale in [1.0, 1.2]
#   random Gaussian blur
#   ColorJitter, brightness/contrast/saturation/hue = 0.3
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
from PIL import Image, ImageFilter, ImageEnhance

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
BATCH_SIZE = 1
EPOCHS = 100
NUM_WORKERS = 4
LR = 5e-4
WEIGHT_DECAY = 1e-4
DECODER_DIM = 96
LORA_R = 4
LORA_ALPHA = 8


# =============================
# HeteCD-style augmentation config
# =============================
# 对应 HeteCD 中启用的增强：
# hflip / vflip / scale_random_crop / random_blur / color_jitter
AUG_HFLIP = True
AUG_VFLIP = True
AUG_SCALE_RANDOM_CROP = True
AUG_RANDOM_BLUR = True
AUG_COLOR_JITTER = True

# HeteCD 代码里有 random rotation，但默认没有启用。这里也默认关闭。
AUG_RANDOM_ROT = False

# HeteCD scale_random_crop 常用范围：1.0 ~ 1.2
SCALE_MIN = 1.0
SCALE_MAX = 1.2

# Gaussian blur
BLUR_PROB = 0.5
BLUR_RADIUS_MIN = 0.1
BLUR_RADIUS_MAX = 2.0

# ColorJitter: brightness/contrast/saturation/hue = 0.3
COLOR_JITTER_PROB = 1.0
COLOR_BRIGHTNESS = 0.3
COLOR_CONTRAST = 0.3
COLOR_SATURATION = 0.3
COLOR_HUE = 0.3

# 为了先按照 HeteCD 的方式，默认 Optical 和 SAR 都做 ColorJitter。
# 如果后续发现 SAR 被扰动后性能下降，可以改成 False。
COLOR_JITTER_ON_SAR = True


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


def color_jitter_pil(
    img: Image.Image,
    brightness: float = 0.3,
    contrast: float = 0.3,
    saturation: float = 0.3,
    hue: float = 0.3,
) -> Image.Image:
    """PIL implementation of ColorJitter-like transform."""
    if brightness > 0:
        factor = random.uniform(max(0.0, 1.0 - brightness), 1.0 + brightness)
        img = ImageEnhance.Brightness(img).enhance(factor)

    if contrast > 0:
        factor = random.uniform(max(0.0, 1.0 - contrast), 1.0 + contrast)
        img = ImageEnhance.Contrast(img).enhance(factor)

    if saturation > 0:
        factor = random.uniform(max(0.0, 1.0 - saturation), 1.0 + saturation)
        img = ImageEnhance.Color(img).enhance(factor)

    if hue > 0:
        hue_factor = random.uniform(-hue, hue)
        hsv = img.convert("HSV")
        h, s, v = hsv.split()
        h_np = np.asarray(h).astype(np.uint16)
        delta = int(hue_factor * 255)
        h_np = ((h_np + delta) % 255).astype(np.uint8)
        h = Image.fromarray(h_np, mode="L")
        img = Image.merge("HSV", (h, s, v)).convert("RGB")

    return img


def scale_random_crop_pil(
    opt: Image.Image,
    sar: Image.Image,
    lab: Image.Image,
    img_size: int,
    scale_min: float = 1.0,
    scale_max: float = 1.2,
):
    """
    Resize to a random scale in [1.0, 1.2], then random crop back to img_size.
    Optical/SAR/label use the same crop to keep pixel alignment.
    """
    scale = random.uniform(scale_min, scale_max)
    scaled_size = max(img_size, int(round(img_size * scale)))

    if scaled_size == img_size:
        return opt, sar, lab

    opt = opt.resize((scaled_size, scaled_size), Image.BILINEAR)
    sar = sar.resize((scaled_size, scaled_size), Image.BILINEAR)
    lab = lab.resize((scaled_size, scaled_size), Image.NEAREST)

    max_x = scaled_size - img_size
    max_y = scaled_size - img_size
    left = random.randint(0, max_x)
    top = random.randint(0, max_y)
    box = (left, top, left + img_size, top + img_size)

    return opt.crop(box), sar.crop(box), lab.crop(box)


class HeteCDTorchDataset(Dataset):
    def __init__(
        self,
        opt_dir: str,
        sar_dir: str,
        label_dir: str,
        img_size: int = 512,
        augment: bool = False,
        aug_hflip: bool = True,
        aug_vflip: bool = True,
        aug_random_rot: bool = False,
        aug_scale_random_crop: bool = True,
        aug_random_blur: bool = True,
        aug_color_jitter: bool = True,
        color_jitter_on_sar: bool = True,
        scale_min: float = 1.0,
        scale_max: float = 1.2,
        blur_prob: float = 0.5,
        blur_radius_min: float = 0.1,
        blur_radius_max: float = 2.0,
        color_jitter_prob: float = 1.0,
        color_brightness: float = 0.3,
        color_contrast: float = 0.3,
        color_saturation: float = 0.3,
        color_hue: float = 0.3,
    ):
        self.opt_dir = Path(opt_dir)
        self.sar_dir = Path(sar_dir)
        self.label_dir = Path(label_dir)
        self.img_size = int(img_size)
        self.augment = augment

        self.aug_hflip = aug_hflip
        self.aug_vflip = aug_vflip
        self.aug_random_rot = aug_random_rot
        self.aug_scale_random_crop = aug_scale_random_crop
        self.aug_random_blur = aug_random_blur
        self.aug_color_jitter = aug_color_jitter
        self.color_jitter_on_sar = color_jitter_on_sar
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.blur_prob = blur_prob
        self.blur_radius_min = blur_radius_min
        self.blur_radius_max = blur_radius_max
        self.color_jitter_prob = color_jitter_prob
        self.color_brightness = color_brightness
        self.color_contrast = color_contrast
        self.color_saturation = color_saturation
        self.color_hue = color_hue

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
            # 1. Random horizontal flip
            if self.aug_hflip and random.random() < 0.5:
                opt = opt.transpose(Image.FLIP_LEFT_RIGHT)
                sar = sar.transpose(Image.FLIP_LEFT_RIGHT)
                lab = lab.transpose(Image.FLIP_LEFT_RIGHT)

            # 2. Random vertical flip
            if self.aug_vflip and random.random() < 0.5:
                opt = opt.transpose(Image.FLIP_TOP_BOTTOM)
                sar = sar.transpose(Image.FLIP_TOP_BOTTOM)
                lab = lab.transpose(Image.FLIP_TOP_BOTTOM)

            # 3. Optional random 90/180/270 rotation. HeteCD has this code, but default was off.
            if self.aug_random_rot:
                k = random.randint(0, 3)
                if k == 1:
                    opt = opt.transpose(Image.ROTATE_90)
                    sar = sar.transpose(Image.ROTATE_90)
                    lab = lab.transpose(Image.ROTATE_90)
                elif k == 2:
                    opt = opt.transpose(Image.ROTATE_180)
                    sar = sar.transpose(Image.ROTATE_180)
                    lab = lab.transpose(Image.ROTATE_180)
                elif k == 3:
                    opt = opt.transpose(Image.ROTATE_270)
                    sar = sar.transpose(Image.ROTATE_270)
                    lab = lab.transpose(Image.ROTATE_270)

            # 4. Scale random crop
            if self.aug_scale_random_crop:
                opt, sar, lab = scale_random_crop_pil(
                    opt,
                    sar,
                    lab,
                    img_size=self.img_size,
                    scale_min=self.scale_min,
                    scale_max=self.scale_max,
                )

            # 5. Random Gaussian blur. Same probability and radius for optical/SAR.
            if self.aug_random_blur and random.random() < self.blur_prob:
                radius = random.uniform(self.blur_radius_min, self.blur_radius_max)
                opt = opt.filter(ImageFilter.GaussianBlur(radius=radius))
                sar = sar.filter(ImageFilter.GaussianBlur(radius=radius))

            # 6. ColorJitter. HeteCD applies color transform to both input images.
            if self.aug_color_jitter and random.random() < self.color_jitter_prob:
                opt = color_jitter_pil(
                    opt,
                    brightness=self.color_brightness,
                    contrast=self.color_contrast,
                    saturation=self.color_saturation,
                    hue=self.color_hue,
                )
                if self.color_jitter_on_sar:
                    sar = color_jitter_pil(
                        sar,
                        brightness=self.color_brightness,
                        contrast=self.color_contrast,
                        saturation=self.color_saturation,
                        hue=self.color_hue,
                    )

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

    train_set = HeteCDTorchDataset(
        *map(str, train_dirs),
        img_size=args.img_size,
        augment=True,
        aug_hflip=args.aug_hflip,
        aug_vflip=args.aug_vflip,
        aug_random_rot=args.aug_random_rot,
        aug_scale_random_crop=args.aug_scale_random_crop,
        aug_random_blur=args.aug_random_blur,
        aug_color_jitter=args.aug_color_jitter,
        color_jitter_on_sar=args.color_jitter_on_sar,
        scale_min=args.scale_min,
        scale_max=args.scale_max,
        blur_prob=args.blur_prob,
        blur_radius_min=args.blur_radius_min,
        blur_radius_max=args.blur_radius_max,
        color_jitter_prob=args.color_jitter_prob,
        color_brightness=args.color_brightness,
        color_contrast=args.color_contrast,
        color_saturation=args.color_saturation,
        color_hue=args.color_hue,
    )
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

    # HeteCD-style augmentation switches.
    parser.add_argument("--aug_hflip", action="store_true", default=AUG_HFLIP)
    parser.add_argument("--no_aug_hflip", action="store_false", dest="aug_hflip")
    parser.add_argument("--aug_vflip", action="store_true", default=AUG_VFLIP)
    parser.add_argument("--no_aug_vflip", action="store_false", dest="aug_vflip")
    parser.add_argument("--aug_random_rot", action="store_true", default=AUG_RANDOM_ROT)
    parser.add_argument("--aug_scale_random_crop", action="store_true", default=AUG_SCALE_RANDOM_CROP)
    parser.add_argument("--no_aug_scale_random_crop", action="store_false", dest="aug_scale_random_crop")
    parser.add_argument("--aug_random_blur", action="store_true", default=AUG_RANDOM_BLUR)
    parser.add_argument("--no_aug_random_blur", action="store_false", dest="aug_random_blur")
    parser.add_argument("--aug_color_jitter", action="store_true", default=AUG_COLOR_JITTER)
    parser.add_argument("--no_aug_color_jitter", action="store_false", dest="aug_color_jitter")
    parser.add_argument("--color_jitter_on_sar", action="store_true", default=COLOR_JITTER_ON_SAR)
    parser.add_argument("--no_color_jitter_on_sar", action="store_false", dest="color_jitter_on_sar")
    parser.add_argument("--scale_min", type=float, default=SCALE_MIN)
    parser.add_argument("--scale_max", type=float, default=SCALE_MAX)
    parser.add_argument("--blur_prob", type=float, default=BLUR_PROB)
    parser.add_argument("--blur_radius_min", type=float, default=BLUR_RADIUS_MIN)
    parser.add_argument("--blur_radius_max", type=float, default=BLUR_RADIUS_MAX)
    parser.add_argument("--color_jitter_prob", type=float, default=COLOR_JITTER_PROB)
    parser.add_argument("--color_brightness", type=float, default=COLOR_BRIGHTNESS)
    parser.add_argument("--color_contrast", type=float, default=COLOR_CONTRAST)
    parser.add_argument("--color_saturation", type=float, default=COLOR_SATURATION)
    parser.add_argument("--color_hue", type=float, default=COLOR_HUE)

    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)

    parser.add_argument("--spa_weight", type=float, default=0.03)
    parser.add_argument("--freq_weight", type=float, default=0.02)
    parser.add_argument("--orth_weight", type=float, default=0.01)

    parser.add_argument("--device", type=str, default="cuda")
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
    print(
        "Augmentation: "
        f"hflip={args.aug_hflip}, vflip={args.aug_vflip}, rot={args.aug_random_rot}, "
        f"scale_crop={args.aug_scale_random_crop}, blur={args.aug_random_blur}, "
        f"color_jitter={args.aug_color_jitter}, color_jitter_on_sar={args.color_jitter_on_sar}"
    )

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

    # Before training, you can test the SAM2 feature shapes by setting DEBUG_FEATURE_SHAPES=True below.
    DEBUG_FEATURE_SHAPES = False

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

    if DEBUG_FEATURE_SHAPES:
        model.eval()
        with torch.no_grad():
            dummy = torch.rand(1, 3, args.img_size, args.img_size, device=device)
            opt_feats = model.optical_encoder(dummy)
            sar_feats = model.sar_encoder(dummy)
            print("Optical feature shapes:", [tuple(f.shape) for f in opt_feats])
            print("SAR feature shapes:", [tuple(f.shape) for f in sar_feats])
        return

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params) / 1e6:.2f} M")

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
                lr=f"{optimizer.param_groups[0]['lr']:.6e}",
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
