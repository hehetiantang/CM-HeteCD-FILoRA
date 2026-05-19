import os
import sys


def _prepare_paddle_gpu_env():
    """
    Configure cuDNN / NCCL library paths before importing Paddle.
    LD_LIBRARY_PATH must be set before the Python process loads Paddle,
    so this function re-executes the current script once.
    """
    conda_prefix = os.environ.get("CONDA_PREFIX", "/home/sjh/anaconda3/envs/paddle")
    py_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"

    cudnn_dir = os.path.join(
        conda_prefix,
        "lib",
        py_ver,
        "site-packages",
        "nvidia",
        "cudnn",
        "lib",
    )

    nccl_dir = os.path.join(
        conda_prefix,
        "lib",
        py_ver,
        "site-packages",
        "nvidia",
        "nccl",
        "lib",
    )

    conda_lib_dir = os.path.join(conda_prefix, "lib")

    # Optional: create local symlink inside conda env, no sudo needed.
    cudnn_so = os.path.join(cudnn_dir, "libcudnn.so")
    cudnn_so8 = os.path.join(cudnn_dir, "libcudnn.so.8")
    if os.path.exists(cudnn_so8) and not os.path.exists(cudnn_so):
        try:
            os.symlink("libcudnn.so.8", cudnn_so)
        except FileExistsError:
            pass

    nccl_so = os.path.join(nccl_dir, "libnccl.so")
    nccl_so2 = os.path.join(nccl_dir, "libnccl.so.2")
    if os.path.exists(nccl_so2) and not os.path.exists(nccl_so):
        try:
            os.symlink("libnccl.so.2", nccl_so)
        except FileExistsError:
            pass

    lib_paths = [
        cudnn_dir,
        nccl_dir,
        conda_lib_dir,
    ]

    lib_paths = [p for p in lib_paths if os.path.exists(p)]

    old_ld = os.environ.get("LD_LIBRARY_PATH", "")
    old_parts = [p for p in old_ld.split(":") if p]

    new_parts = []
    for p in lib_paths + old_parts:
        if p not in new_parts:
            new_parts.append(p)

    new_ld = ":".join(new_parts)

    # Re-exec only once. This is necessary because LD_LIBRARY_PATH is read
    # by the dynamic loader when the process starts.
    if os.environ.get("PADDLE_GPU_ENV_READY") != "1":
        os.environ["LD_LIBRARY_PATH"] = new_ld
        os.environ["PADDLE_GPU_ENV_READY"] = "1"

        # 如果你想强制单卡，可以取消下面一行注释。
        # os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

        os.execv(sys.executable, [sys.executable] + sys.argv)


_prepare_paddle_gpu_env()

import random
import os
import numpy as np
import paddle
import logging
import argparse
from functools import partial


# from cd_models.fccdn import FCCDN
# from cd_models.stanet import STANet
# from cd_models.p2v import P2V
# from cd_models.msfgnet import MSFGNet
# from cd_models.fc_siam_conc import FCSiamConc
# from cd_models.dsamnet import DSAMNet
# from cd_models.snunet import SNUNet
# from cd_models.f3net import F3Net
# from paddleseg.models import UNet
# from cd_models.replkcd import CD_RLKNet
# from cd_models.efisam import EFISam
# from paddleseg.models import UNet
# from cd_models.cienet.ciescd import CIENetTinyViT
# from sfinet.model import SFFNet_BCD
from filora.cm_hetecd_filora import CMHeteCDFILoRA_BCD

from core.bcdwork import Work


# dataset_name = "LEVIR_CD"
# dataset_name = "LEVIR_CDP"
# dataset_name = "GVLM_CD"
# dataset_name = "MacaoCD"
# dataset_name = "SYSU_CD"
# dataset_name = "WHU_BCD"
# dataset_name = "S2Looking"
# dataset_name = "CLCD"
dataset_name = "XiongAn"

dataset_path = 'data/sjh/data/{}'.format(dataset_name)

pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser(
        description="CM-HeteCD-FILoRA for XiongAn heterogeneous change detection"
    )

    parser.add_argument("--model", type=str, default="CMHeteCDFILoRA_BCD")
    parser.add_argument("--root", type=str, default="./output")
    parser.add_argument("--data_root", type=str, default="/data/sjh/data")
    parser.add_argument("--dataset", type=str, default="XiongAn")

    parser.add_argument("--input_mode", type=str, default="hetecd")

    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--device", type=str, default="gpu:1")

    parser.add_argument("--num_classes", type=int, default=2)

    # XiongAn: Optical RGB 3 channels, SAR selected HH/VV/HV 3 channels
    parser.add_argument("--in_chans_t1", type=int, default=3)
    parser.add_argument("--in_chans_t2", type=int, default=3)

    parser.add_argument("--base_dim", type=int, default=32)
    parser.add_argument("--decoder_dim", type=int, default=128)

    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--fca_weight", type=float, default=2.0)
    parser.add_argument("--fca_temperature", type=float, default=4.0)

    parser.add_argument("--img_ab_concat", type=bool, default=True)
    parser.add_argument("--en_load_edge", type=bool, default=False)
    parser.add_argument("--test", type=bool, default=True)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    print("main")

    pil_logger = logging.getLogger("PIL")
    pil_logger.setLevel(logging.INFO)

    args = parse_args()

    random.seed(32767)
    os.environ["PYTHONHASHSEED"] = str(32767)
    np.random.seed(32767)
    paddle.seed(32767)

    paddle.device.set_device(args.device)

    model = CMHeteCDFILoRA_BCD(
        img_size=args.img_size,
        num_cls=args.num_classes,
        in_chans_t1=args.in_chans_t1,
        in_chans_t2=args.in_chans_t2,
        base_dim=args.base_dim,
        decoder_dim=args.decoder_dim,
    )

    Work(model, args)
  


