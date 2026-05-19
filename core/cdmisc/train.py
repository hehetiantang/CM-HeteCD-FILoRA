# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0

import os
import time
import logging
from tqdm import tqdm

import numpy as np
import paddle
import paddle.nn.functional as F
import paddle.optimizer

from .val import evaluate
from .predict import test


def _prepare_label(labels):
    """
    Prepare BCD labels.

    Internal label shape:
        [B, H, W]

    Label value:
        0: unchanged
        1: changed
    """

    if not isinstance(labels, paddle.Tensor):
        labels = paddle.to_tensor(labels)

    # [H, W] -> [1, H, W]
    if len(labels.shape) == 2:
        labels = labels.unsqueeze(0)

    # [B, 1, H, W] -> [B, H, W]
    if len(labels.shape) == 4 and labels.shape[1] == 1:
        labels = labels.squeeze(1)

    # [B, H, W, 1] -> [B, H, W]
    elif len(labels.shape) == 4 and labels.shape[-1] == 1:
        labels = labels.squeeze(-1)

    # [B, 2, H, W] one-hot -> [B, H, W]
    elif len(labels.shape) == 4 and labels.shape[1] == 2:
        labels = paddle.argmax(labels, axis=1)

    # [B, H, W, 3] RGB label -> use first channel
    elif len(labels.shape) == 4 and labels.shape[-1] == 3:
        labels = labels[:, :, :, 0]

    # Binary change detection:
    # non-zero pixels are regarded as changed.
    labels = (labels > 0).astype("int64")

    return labels


def bcd_loss(logits, labels, dice_weight=1.0, eps=1e-6):
    """
    Binary change detection loss.

    For CMHeteCDFILoRA_BCD:
        logits: [B, 2, H, W]
        labels: [B, H, W]

    Loss:
        CrossEntropy + Dice Loss
    """

    labels = _prepare_label(labels)

    if isinstance(logits, (tuple, list)):
        logits = logits[0]

    if isinstance(logits, dict):
        logits = logits["logits"]

    # Resize logits to label size if needed.
    if logits.shape[-2:] != labels.shape[-2:]:
        logits = F.interpolate(
            logits,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=True
        )

    # Case 1: two-class output [B, 2, H, W]
    if len(logits.shape) == 4 and logits.shape[1] == 2:
        labels_ce = labels.unsqueeze(1)

        ce_loss = F.cross_entropy(
            input=logits,
            label=labels_ce,
            axis=1
        )

        probs = F.softmax(logits, axis=1)
        change_prob = probs[:, 1, :, :]
        change_label = (labels == 1).astype("float32")

        intersection = paddle.sum(change_prob * change_label)
        union = paddle.sum(change_prob) + paddle.sum(change_label)

        dice_loss = 1.0 - (2.0 * intersection + eps) / (union + eps)

        return ce_loss + dice_weight * dice_loss

    # Case 2: single-channel output [B, 1, H, W]
    elif len(logits.shape) == 4 and logits.shape[1] == 1:
        target = labels.astype("float32").unsqueeze(1)

        bce_loss = F.binary_cross_entropy_with_logits(
            logit=logits,
            label=target
        )

        probs = F.sigmoid(logits)

        intersection = paddle.sum(probs * target)
        union = paddle.sum(probs) + paddle.sum(target)

        dice_loss = 1.0 - (2.0 * intersection + eps) / (union + eps)

        return bce_loss + dice_weight * dice_loss

    else:
        raise ValueError(
            "Unsupported logits shape for BCD loss. "
            f"Got logits shape: {logits.shape}. "
            "Expected [B, 2, H, W] or [B, 1, H, W]."
        )


def _l2_normalize_feature(x, eps=1e-6):
    """
    L2 normalize feature along channel dimension.

    x: [B, C, H, W]
    """
    norm = paddle.sqrt(paddle.sum(x * x, axis=1, keepdim=True)) + eps
    return x / norm


def _masked_mean(x, mask, eps=1e-6):
    """
    x:    [B, H, W]
    mask: [B, H, W]
    """
    return paddle.sum(x * mask) / (paddle.sum(mask) + eps)


def hetecd_fca_loss(
        feats_opt,
        feats_sar,
        labels,
        temperature=4.0,
        changed_weight=0.0,
        margin=0.2,
        eps=1e-6
):
    """
    HeteCD-style FCA loss: Feature Consistency Alignment Loss.

    Purpose:
        Align optical and SAR features in unchanged regions.

    Inputs:
        feats_opt: list of optical features.
                   Each feature shape: [B, C, H, W]

        feats_sar: list of SAR features.
                   Each feature shape: [B, C, H, W]

        labels:    [B, H, W] or [B, 1, H, W]
                   0 = unchanged
                   1 = changed

    Main idea:
        Unchanged pixels:
            optical feature and SAR feature should be consistent.

        Changed pixels:
            by default, we do not force alignment.
            changed_weight can be set > 0 if you want weak separation.
    """

    labels = _prepare_label(labels)

    if feats_opt is None or feats_sar is None:
        return paddle.to_tensor(0.0, dtype="float32")

    if not isinstance(feats_opt, (list, tuple)):
        feats_opt = [feats_opt]

    if not isinstance(feats_sar, (list, tuple)):
        feats_sar = [feats_sar]

    total_fca_loss = paddle.to_tensor(0.0, dtype="float32")
    valid_scales = 0

    temp = max(float(temperature), eps)

    for fo, fs in zip(feats_opt, feats_sar):
        # fo, fs: [B, C, H, W]
        h, w = fo.shape[2], fo.shape[3]

        if fs.shape[-2:] != fo.shape[-2:]:
            fs = F.interpolate(
                fs,
                size=(h, w),
                mode="bilinear",
                align_corners=True
            )

        # Resize label to current feature scale.
        label_small = labels.astype("float32").unsqueeze(1)
        label_small = F.interpolate(
            label_small,
            size=(h, w),
            mode="nearest"
        )
        label_small = label_small.squeeze(1).astype("int64")

        unchanged_mask = (label_small == 0).astype("float32")
        changed_mask = (label_small == 1).astype("float32")

        fo_norm = _l2_normalize_feature(fo, eps)
        fs_norm = _l2_normalize_feature(fs, eps)

        # Cosine similarity: [B, H, W], range roughly [-1, 1]
        cos_sim = paddle.sum(fo_norm * fs_norm, axis=1)

        # FCA for unchanged regions:
        # larger similarity -> smaller loss.
        unchanged_loss_map = (1.0 - cos_sim) / temp
        unchanged_loss = _masked_mean(
            unchanged_loss_map,
            unchanged_mask,
            eps=eps
        )

        # Optional weak separation for changed regions.
        # Default changed_weight = 0.0, so changed regions are not forced.
        if changed_weight > 0:
            changed_loss_map = F.relu(cos_sim - margin)
            changed_loss = _masked_mean(
                changed_loss_map,
                changed_mask,
                eps=eps
            )
            scale_loss = unchanged_loss + changed_weight * changed_loss
        else:
            scale_loss = unchanged_loss

        total_fca_loss = total_fca_loss + scale_loss
        valid_scales += 1

    if valid_scales == 0:
        return paddle.to_tensor(0.0, dtype="float32")

    return total_fca_loss / valid_scales


def _forward_model(model, data, return_aux=False):
    """
    Forward wrapper.

    Compatible with:

    1. data["img"] = concat(A, B)
       model(data["img"], return_aux=True/False)

    2. data["img_a"], data["img_b"]
       model(data["img_a"], data["img_b"], return_aux=True/False)

    3. data["A"], data["B"]
       model(data["A"], data["B"], return_aux=True/False)
    """

    if "img_a" in data and "img_b" in data:
        img_a = data["img_a"].astype("float32")
        img_b = data["img_b"].astype("float32")

        if return_aux:
            pred = model(img_a, img_b, return_aux=True)
        else:
            pred = model(img_a, img_b)

    elif "A" in data and "B" in data:
        img_a = data["A"].astype("float32")
        img_b = data["B"].astype("float32")

        if return_aux:
            pred = model(img_a, img_b, return_aux=True)
        else:
            pred = model(img_a, img_b)

    else:
        images = data["img"].astype("float32")

        if return_aux:
            pred = model(images, return_aux=True)
        else:
            pred = model(images)

    return pred


def _to_float(x):
    """
    Convert Paddle scalar tensor to Python float.
    """
    if isinstance(x, paddle.Tensor):
        return float(x.detach().cpu().numpy())
    return float(x)


def train(model, train_dataset, val_dataset, test_dataset, args):
    """
    Launch training for binary change detection with optional FCA loss.

    Total loss:
        loss_total = seg_loss + fca_weight * fca_loss

    When args.fca_weight > 0:
        model will be called with return_aux=True.
    """

    logger = getattr(args, "logger", None)

    if logger is not None:
        logger.info("start train")
    else:
        print("start train")

    os.makedirs(args.save_dir, exist_ok=True)

    use_fca = getattr(args, "fca_weight", 0.0) > 0.0

    if use_fca:
        msg = (
            "[FCA] enabled. "
            f"fca_weight={getattr(args, 'fca_weight', 0.0)}, "
            f"fca_temperature={getattr(args, 'fca_temperature', 4.0)}"
        )
    else:
        msg = "[FCA] disabled. Set --fca_weight > 0 to enable."

    if logger is not None:
        logger.info(msg)
    else:
        print(msg)

    # Cosine learning rate scheduler
    lr_scheduler = paddle.optimizer.lr.CosineAnnealingDecay(
        learning_rate=args.lr,
        T_max=max(1, args.iters),
        last_epoch=-1
    )

    optimizer = paddle.optimizer.Adam(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        weight_decay=getattr(args, "weight_decay", 0.0)
    )

    best_mean_iou = -1.0
    best_model_iter = -1

    batch_start = time.time()

    for epoch in range(1, args.iters + 1):
        model.train()

        avg_loss_list = []
        avg_seg_loss_list = []
        avg_fca_loss_list = []

        progress_bar = tqdm(
            train_dataset,
            desc=f"Epoch [{epoch}/{args.iters}]",
            ncols=120
        )

        for data in progress_bar:
            labels = data["label"]
            labels = _prepare_label(labels)

            if use_fca:
                out = _forward_model(model, data, return_aux=True)

                if isinstance(out, dict):
                    pred = out["logits"]
                    fca_opt = out.get("fca_opt", None)
                    fca_sar = out.get("fca_sar", None)
                else:
                    pred = out
                    fca_opt = None
                    fca_sar = None

                if hasattr(model, "loss") and callable(model.loss):
                    seg_loss = model.loss(pred, labels)
                else:
                    seg_loss = bcd_loss(pred, labels)

                fca_loss_value = hetecd_fca_loss(
                    feats_opt=fca_opt,
                    feats_sar=fca_sar,
                    labels=labels,
                    temperature=getattr(args, "fca_temperature", 4.0),
                    changed_weight=getattr(args, "fca_changed_weight", 0.0),
                    margin=getattr(args, "fca_margin", 0.2)
                )

                loss_total = seg_loss + getattr(args, "fca_weight", 0.0) * fca_loss_value

            else:
                pred = _forward_model(model, data, return_aux=False)

                if isinstance(pred, dict):
                    pred = pred["logits"]

                if isinstance(pred, (tuple, list)):
                    pred = pred[0]

                if hasattr(model, "loss") and callable(model.loss):
                    seg_loss = model.loss(pred, labels)
                else:
                    seg_loss = bcd_loss(pred, labels)

                fca_loss_value = paddle.to_tensor(0.0, dtype="float32")
                loss_total = seg_loss

            loss_total.backward()
            optimizer.step()
            optimizer.clear_grad()

            lr = optimizer.get_lr()

            if isinstance(lr_scheduler, paddle.optimizer.lr.LRScheduler):
                lr_scheduler.step()

            loss_value = _to_float(loss_total)
            seg_loss_value = _to_float(seg_loss)
            fca_loss_float = _to_float(fca_loss_value)

            avg_loss_list.append(loss_value)
            avg_seg_loss_list.append(seg_loss_value)
            avg_fca_loss_list.append(fca_loss_float)

            if use_fca:
                progress_bar.set_postfix(
                    total=f"{loss_value:.4f}",
                    seg=f"{seg_loss_value:.4f}",
                    fca=f"{fca_loss_float:.4f}",
                    lr=f"{lr:.6f}"
                )
            else:
                progress_bar.set_postfix(
                    loss=f"{loss_value:.4f}",
                    lr=f"{lr:.6f}"
                )

        batch_cost = time.time() - batch_start

        avg_loss = float(np.mean(avg_loss_list)) if len(avg_loss_list) > 0 else 0.0
        avg_seg_loss = float(np.mean(avg_seg_loss_list)) if len(avg_seg_loss_list) > 0 else 0.0
        avg_fca_loss = float(np.mean(avg_fca_loss_list)) if len(avg_fca_loss_list) > 0 else 0.0

        train_num = getattr(args, "traindata_num", None)
        if train_num is not None and train_num > 0:
            ips = train_num / max(batch_cost, 1e-6)
        else:
            ips = 0.0

        if use_fca:
            train_msg = (
                "[TRAIN] iter: {}/{}, total_loss: {:.4f}, seg_loss: {:.4f}, "
                "fca_loss: {:.4f}, fca_weight: {:.4f}, lr: {:.6f}, "
                "batch_cost: {:.2f}, ips: {:.4f} samples/sec"
            ).format(
                epoch,
                args.iters,
                avg_loss,
                avg_seg_loss,
                avg_fca_loss,
                getattr(args, "fca_weight", 0.0),
                lr,
                batch_cost,
                ips
            )
        else:
            train_msg = (
                "[TRAIN] iter: {}/{}, loss: {:.4f}, lr: {:.6f}, "
                "batch_cost: {:.2f}, ips: {:.4f} samples/sec"
            ).format(
                epoch,
                args.iters,
                avg_loss,
                lr,
                batch_cost,
                ips
            )

        if logger is not None:
            logger.info(train_msg)
        else:
            print(train_msg)

        # Save last model
        if epoch == args.iters:
            last_model_path = os.path.join(args.save_dir, "last_model.pdparams")
            paddle.save(model.state_dict(), last_model_path)

            if logger is not None:
                logger.info(f"[SAVE] last model saved to {last_model_path}")
            else:
                print(f"[SAVE] last model saved to {last_model_path}")

        # Validation
        mean_iou = evaluate(model, val_dataset, args)

        # Save best model
        if mean_iou > best_mean_iou:
            best_mean_iou = mean_iou
            best_model_iter = epoch

            paddle.save(model.state_dict(), args.best_model_path)

            if logger is not None:
                logger.info(
                    "[SAVE] best model saved to {}, iter {}, max IoU {:.4f}".format(
                        args.best_model_path,
                        best_model_iter,
                        best_mean_iou
                    )
                )
            else:
                print(
                    "[SAVE] best model saved to {}, iter {}, max IoU {:.4f}".format(
                        args.best_model_path,
                        best_model_iter,
                        best_mean_iou
                    )
                )

        best_msg = "[TRAIN] best iter {}, max IoU {:.4f}".format(
            best_model_iter,
            best_mean_iou
        )

        if logger is not None:
            logger.info(best_msg)
        else:
            print(best_msg)

        batch_start = time.time()

    # Load best model before test
    if os.path.exists(args.best_model_path):
        state_dict = paddle.load(args.best_model_path)
        model.set_state_dict(state_dict)

        if logger is not None:
            logger.info(f"[TEST] loaded best model from {args.best_model_path}")
        else:
            print(f"[TEST] loaded best model from {args.best_model_path}")
    else:
        if logger is not None:
            logger.warning("[TEST] best model path not found. Test with current model.")
        else:
            print("[TEST] best model path not found. Test with current model.")

    test(model, test_dataset, args)

    logging.shutdown()

    # Sleep for a second to let dataloader release resources.
    time.sleep(1)
