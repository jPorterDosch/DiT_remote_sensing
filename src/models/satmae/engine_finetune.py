# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math

import torch
from timm.data.mixup import Mixup
from timm.utils.metrics import accuracy
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from utils import NativeScalerWithGradNormCount


def adjust_learning_rate(optimizer: Optimizer, epoch: float, config) -> float:
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < config.warmup_epochs:
        lr = config.lr * epoch / config.warmup_epochs
    else:
        lr = config.min_lr + (config.lr - config.min_lr) * 0.5 * (
            1.0 + math.cos(math.pi * (epoch - config.warmup_epochs) / (config.epochs - config.warmup_epochs))
        )
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler: NativeScalerWithGradNormCount,
    max_norm: float = 0,
    mixup_fn: Mixup | None = None,
    # TODO: fill in entry point config object.
    config=None,
) -> dict[str, float]:
    # Temporary until we configure config properly.
    if config is None:
        raise ValueError("config must be provided")

    model.train()
    optimizer.zero_grad()

    total_loss = 0.0
    total_samples = 0

    for data_iter_step, samples in enumerate(data_loader):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % config.accum_iter == 0:
            adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, config)

        images, targets = samples["img"], samples["label"]
        images = images.to(device)
        targets = targets.to(device)

        if mixup_fn is not None:
            images, targets = mixup_fn(images, targets)

        with torch.amp.autocast(device_type=device.type):
            outputs = model(images)
            loss = criterion(outputs, targets)

        loss_value = loss.item()
        batch_size = images.shape[0]
        total_loss += loss_value * batch_size
        total_samples += batch_size

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            raise ValueError(f"Loss is {loss_value}, stopping training")

        loss = loss / config.accum_iter
        loss_scaler(
            loss,
            optimizer,
            clip_grad=max_norm,
            parameters=model.parameters(),
            create_graph=False,
            update_grad=(data_iter_step + 1) % config.accum_iter == 0,
        )
        if (data_iter_step + 1) % config.accum_iter == 0:
            optimizer.zero_grad()

        # TODO: replaced hardcoded min_lr (0.0) and max_lr (10.0) with config values. MUST ADD TO CONFIG
        # min_lr = min(
        #     config.min_lr, max(group["lr"] for group in optimizer.param_groups)
        # )
        max_lr = max(config.max_lr, max(group["lr"] for group in optimizer.param_groups))

    stats = {
        "loss": total_loss / max(total_samples, 1),
        "lr": max_lr,
    }

    print("Averaged stats:", stats)
    return stats


@torch.no_grad()
def evaluate(
    data_loader: DataLoader,
    model: torch.nn.Module,
    device: torch.device,
) -> dict[str, float]:
    criterion = torch.nn.CrossEntropyLoss()

    # switch to evaluation mode
    model.eval()

    total_loss = 0.0
    total_acc1 = 0.0
    total_acc5 = 0.0
    total_samples = 0

    for batch in data_loader:
        images, target = batch["img"], batch["label"]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type):
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        total_loss += loss.item() * batch_size
        total_acc1 += acc1.item() * batch_size
        total_acc5 += acc5.item() * batch_size
        total_samples += batch_size

    stats = {
        "loss": total_loss / max(total_samples, 1),
        "acc1": total_acc1 / max(total_samples, 1),
        "acc5": total_acc5 / max(total_samples, 1),
    }

    print("* Acc@1 {acc1:.3f} Acc@5 {acc5:.3f} loss {loss:.3f}".format(**stats))

    return stats
