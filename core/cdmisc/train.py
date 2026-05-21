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
            align_corners=True,
        )

    # Case 1: two-class output [B, 2, H, W]
    if len(logits.shape) == 4 and logits.shape[1] == 2:
        labels_ce = labels.unsqueeze(1)

        ce_loss = F.cross_entropy(
            input=logits,
            label=labels_ce,
            axis=1,
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
            label=target,
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


def _to_list(x):
    """
    Convert Tensor to list[Tensor].
    """
    if x is None:
        return None

    if isinstance(x, (list, tuple)):
        return list(x)

    return [x]


def _kl_divergence_with_temperature(x_t1, x_t2, axis, temperature=4.0):
    """
    KL divergence with temperature.

    KL(p_t1 || p_t2)
        = sum p_t1 * (log p_t1 - log p_t2)

    Args:
        x_t1: feature tensor from optical branch
        x_t2: feature tensor from SAR branch
        axis: dimension used to calculate softmax distribution
        temperature: temperature factor T
    """
    t = float(temperature)

    p_t1 = F.softmax(x_t1 / t, axis=axis)
    log_p_t1 = F.log_softmax(x_t1 / t, axis=axis)
    log_p_t2 = F.log_softmax(x_t2 / t, axis=axis)

    kl = paddle.sum(p_t1 * (log_p_t1 - log_p_t2), axis=axis)

    return kl


def hetecd_fca_loss(
    feats_opt,
    feats_sar,
    temperature=4.0,
    eps=1e-6,
):
    """
    FCA Loss following the original HeteCD paper.

    This version is different from the previous mask-based cosine FCA.

    Original idea:
        1. Align heterogeneous feature distributions by KL divergence.
        2. Calculate distribution consistency from spatial dimension.
        3. Calculate distribution consistency from channel dimension.
        4. No unchanged-region mask is used here.

    Args:
        feats_opt:
            Optical features. Tensor or list[Tensor].
            Each tensor shape: [B, C, H, W]

        feats_sar:
            SAR features. Tensor or list[Tensor].
            Each tensor shape: [B, C, H, W]

        temperature:
            Temperature factor T. Default is 4.0.

        eps:
            Small value for numerical safety.

    Returns:
        FCA loss.
    """
    if feats_opt is None or feats_sar is None:
        return paddle.to_tensor(0.0, dtype="float32")

    feats_opt = _to_list(feats_opt)
    feats_sar = _to_list(feats_sar)

    if feats_opt is None or feats_sar is None:
        return paddle.to_tensor(0.0, dtype="float32")

    total_fca_loss = paddle.to_tensor(0.0, dtype="float32")
    valid_scales = 0

    t = max(float(temperature), eps)

    for fo, fs in zip(feats_opt, feats_sar):
        if fo is None or fs is None:
            continue

        # fo, fs: [B, C, H, W]
        if len(fo.shape) != 4 or len(fs.shape) != 4:
            raise ValueError(
                "FCA features must be 4D tensors with shape [B, C, H, W]. "
                f"Got fo.shape={fo.shape}, fs.shape={fs.shape}."
            )

        b, c, h, w = fo.shape
        _, c_sar, h_sar, w_sar = fs.shape

        # Resize SAR feature to optical feature size if needed.
        if (h_sar, w_sar) != (h, w):
            fs = F.interpolate(
                fs,
                size=(h, w),
                mode="bilinear",
                align_corners=True,
            )

        if c_sar != c:
            raise ValueError(
                "Optical and SAR features must have the same channel number "
                "for HeteCD FCA KL alignment. "
                f"Got optical C={c}, SAR C={c_sar}."
            )

        hw = h * w

        # ------------------------------------------------------------
        # 1. Spatial-position distribution alignment
        # ------------------------------------------------------------
        # For each spatial position i, compare the channel distributions:
        #     p(X_t1^i) and p(X_t2^i)
        #
        # Original formula:
        #     T^2 / HW * sum_i KL(p(X_t1^i) || p(X_t2^i))
        #
        # Tensor transform:
        #     [B, C, H, W] -> [B, C, HW] -> [B, HW, C]
        # Softmax axis:
        #     channel dimension C
        # ------------------------------------------------------------
        fo_pos = paddle.reshape(fo, [b, c, hw])
        fs_pos = paddle.reshape(fs, [b, c, hw])

        fo_pos = paddle.transpose(fo_pos, [0, 2, 1])
        fs_pos = paddle.transpose(fs_pos, [0, 2, 1])

        spatial_kl = _kl_divergence_with_temperature(
            fo_pos,
            fs_pos,
            axis=-1,
            temperature=t,
        )

        spatial_loss = paddle.mean(spatial_kl)

        # ------------------------------------------------------------
        # 2. Channel distribution alignment
        # ------------------------------------------------------------
        # For each channel c, compare the spatial distributions:
        #     p(X_t1^c) and p(X_t2^c)
        #
        # Original formula:
        #     T^2 / C * sum_c KL(p(X_t1^c) || p(X_t2^c))
        #
        # Tensor transform:
        #     [B, C, H, W] -> [B, C, HW]
        # Softmax axis:
        #     spatial dimension HW
        # ------------------------------------------------------------
        fo_channel = paddle.reshape(fo, [b, c, hw])
        fs_channel = paddle.reshape(fs, [b, c, hw])

        channel_kl = _kl_divergence_with_temperature(
            fo_channel,
            fs_channel,
            axis=-1,
            temperature=t,
        )

        channel_loss = paddle.mean(channel_kl)

        # HeteCD uses T^2 as the scale factor.
        scale_loss = (t * t) * (spatial_loss + channel_loss)

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
    Launch training for binary change detection with optional HeteCD FCA loss.

    Total loss follows the original HeteCD paper:

        loss_total = seg_loss + alpha / epoch * fca_loss

    In this implementation:

        alpha = args.fca_weight

    Recommended setting:

        args.fca_weight = 2.0
        args.fca_temperature = 4.0
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
            "[FCA] enabled with HeteCD KL distribution alignment. "
            f"alpha={getattr(args, 'fca_weight', 0.0)}, "
            f"temperature={getattr(args, 'fca_temperature', 4.0)}, "
            "effective_weight = alpha / epoch"
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
        last_epoch=-1,
    )

    optimizer = paddle.optimizer.Adam(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        weight_decay=getattr(args, "weight_decay", 0.0),
    )

    best_mean_iou = -1.0
    best_model_iter = -1
    batch_start = time.time()

    for epoch in range(1, args.iters + 1):
        model.train()

        avg_loss_list = []
        avg_seg_loss_list = []
        avg_fca_loss_list = []
        avg_effective_fca_weight_list = []

        progress_bar = tqdm(
            train_dataset,
            desc=f"Epoch [{epoch}/{args.iters}]",
            ncols=120,
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
                    temperature=getattr(args, "fca_temperature", 4.0),
                )

                alpha = float(getattr(args, "fca_weight", 0.0))
                effective_fca_weight = alpha / float(max(epoch, 1))

                loss_total = seg_loss + effective_fca_weight * fca_loss_value

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
                effective_fca_weight = 0.0
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
            avg_effective_fca_weight_list.append(effective_fca_weight)

            if use_fca:
                progress_bar.set_postfix(
                    total=f"{loss_value:.4f}",
                    seg=f"{seg_loss_value:.4f}",
                    fca=f"{fca_loss_float:.4f}",
                    w=f"{effective_fca_weight:.4f}",
                    lr=f"{lr:.6f}",
                )
            else:
                progress_bar.set_postfix(
                    loss=f"{loss_value:.4f}",
                    lr=f"{lr:.6f}",
                )

        batch_cost = time.time() - batch_start

        avg_loss = float(np.mean(avg_loss_list)) if len(avg_loss_list) > 0 else 0.0
        avg_seg_loss = (
            float(np.mean(avg_seg_loss_list)) if len(avg_seg_loss_list) > 0 else 0.0
        )
        avg_fca_loss = (
            float(np.mean(avg_fca_loss_list)) if len(avg_fca_loss_list) > 0 else 0.0
        )
        avg_effective_fca_weight = (
            float(np.mean(avg_effective_fca_weight_list))
            if len(avg_effective_fca_weight_list) > 0
            else 0.0
        )

        train_num = getattr(args, "traindata_num", None)

        if train_num is not None and train_num > 0:
            ips = train_num / max(batch_cost, 1e-6)
        else:
            ips = 0.0

        if use_fca:
            train_msg = (
                "[TRAIN] iter: {}/{}, total_loss: {:.4f}, seg_loss: {:.4f}, "
                "fca_loss: {:.4f}, effective_fca_weight: {:.4f}, lr: {:.6f}, "
                "batch_cost: {:.2f}, ips: {:.4f} samples/sec"
            ).format(
                epoch,
                args.iters,
                avg_loss,
                avg_seg_loss,
                avg_fca_loss,
                avg_effective_fca_weight,
                lr,
                batch_cost,
                ips,
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
                ips,
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
                        best_mean_iou,
                    )
                )
            else:
                print(
                    "[SAVE] best model saved to {}, iter {}, max IoU {:.4f}".format(
                        args.best_model_path,
                        best_model_iter,
                        best_mean_iou,
                    )
                )

        best_msg = "[TRAIN] best iter {}, max IoU {:.4f}".format(
            best_model_iter,
            best_mean_iou,
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
