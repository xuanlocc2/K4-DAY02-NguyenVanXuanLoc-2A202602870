"""model.py - tạo backbone, đóng băng, nhóm tham số, đếm params/GMAC.

Giao diện giữ đúng như khung:
    build_model(name, pretrained, num_classes, drop_rate, init) -> nn.Module
    freeze_backbone(model)                                        -> None
    param_groups(model, lr_backbone, lr_head, weight_decay)       -> list[dict]
    count_params(model) -> float (triệu)     count_gmacs(model, img_size) -> float
Thêm: set_train_mode(model) để giữ BN của backbone đóng băng ở eval.
"""
from __future__ import annotations

import timm
import torch
from torch import nn

SUGGESTED_BACKBONES = {
    "resnet50": "resnet50",
    "resnext50": "resnext50_32x4d",
    "convnext_tiny": "convnext_tiny",
    "deit_small": "deit_small_patch16_224",
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "efficientnet_b0": "efficientnet_b0",
    "mobilenetv3": "mobilenetv3_large_100",
}


def _head_params(model) -> list:
    return list(model.get_classifier().parameters())


def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune"):
    """init: scratch (không pretrained) | frozen (pretrained, chỉ train head) | finetune."""
    name = SUGGESTED_BACKBONES.get(name, name)
    if init not in ("scratch", "frozen", "finetune"):
        raise ValueError(f"init không hợp lệ: {init}")
    use_pre = pretrained and init != "scratch"
    model = timm.create_model(name, pretrained=use_pre, num_classes=num_classes, drop_rate=drop_rate)
    cfg = getattr(model, "pretrained_cfg", {}) or {}
    model.weight_tag = (cfg.get("hf_hub_id") or cfg.get("tag") or cfg.get("url") or "none") if use_pre else "none"
    model.arch_name = name
    model.is_frozen = False
    if init == "frozen":
        freeze_backbone(model)
    return model


def freeze_backbone(model) -> None:
    """requires_grad=False cho mọi tham số trừ head. BN backbone phải ở eval: dùng set_train_mode."""
    head = {id(p) for p in _head_params(model)}
    for p in model.parameters():
        p.requires_grad = id(p) in head
    model.is_frozen = True


def set_train_mode(model) -> None:
    """model.train(), nhưng nếu backbone đóng băng thì mọi module không thuộc head về eval
    (để BatchNorm không cập nhật running stats và dropout của backbone tắt)."""
    model.train()
    if getattr(model, "is_frozen", False):
        head_mods = set(model.get_classifier().modules())
        for m in model.modules():
            if m not in head_mods:
                m.training = False  # không dùng m.eval(): nó đệ quy và tắt luôn cả head


def param_groups(model, lr_backbone: float, lr_head: float, weight_decay: float):
    """3 nhóm: backbone ndim>1 (wd) | norm+bias backbone (wd=0) | head (lr_head, wd)."""
    head_ids = {id(p) for p in _head_params(model)}
    decay, no_decay, head = [], [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if id(p) in head_ids:
            head.append(p)
        elif p.ndim > 1:
            decay.append(p)
        else:
            no_decay.append(p)
    groups = [
        {"params": decay, "lr": lr_backbone, "weight_decay": weight_decay},
        {"params": no_decay, "lr": lr_backbone, "weight_decay": 0.0},
        {"params": head, "lr": lr_head, "weight_decay": weight_decay},
    ]
    return [g for g in groups if g["params"]]


def count_params(model) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


def count_gmacs(model, img_size: int = 224) -> float:
    """GMAC (= FLOPs / 2) bằng torch.utils.flop_counter (chỉ đếm conv/matmul/attention).
    Số có thể lệch vài % so với fvcore/ptflops."""
    from torch.utils.flop_counter import FlopCounterMode

    was_training = model.training
    model.eval()
    p = next(model.parameters())
    x = torch.zeros(1, 3, img_size, img_size, device=p.device, dtype=p.dtype)
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        model(x)
    model.train(was_training)
    return fc.get_total_flops() / 2 / 1e9
