"""losses.py - các hàm loss và trộn mẫu (Mixup, CutMix).

Giao diện giữ đúng như khung:
    build_criterion(kind, **kw)                 -> callable(logits, target) -> loss scalar
    class_weights(counts, beta)                 -> tensor trọng số lớp
    mix_batch(x, y, alpha, mode)                -> (x_mixed, (y_a, y_b, lam))
    mixed_loss(criterion, logits, targets)      -> loss scalar
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class LabelSmoothingCE(nn.Module):
    """q'(k) = (1-eps)*1[k==y] + eps/K. Tự cài đặt; eps=0 cho đúng CE."""

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.eps = smoothing

    def forward(self, logits, target):
        logp = F.log_softmax(logits.float(), dim=-1)
        nll = -logp.gather(1, target[:, None]).squeeze(1)
        smooth = -logp.mean(dim=-1)
        return ((1 - self.eps) * nll + self.eps * smooth).mean()


class FocalLoss(nn.Module):
    """FL = -alpha_t (1-p_t)^gamma log p_t, trung bình batch. gamma=0 và alpha=None => CE."""

    def __init__(self, gamma: float = 2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", None if alpha is None else torch.as_tensor(alpha, dtype=torch.float32))

    def forward(self, logits, target):
        logp = F.log_softmax(logits.float(), dim=-1).gather(1, target[:, None]).squeeze(1)
        pt = logp.exp()
        loss = -((1 - pt) ** self.gamma) * logp
        if self.alpha is not None:
            loss = loss * self.alpha[target]
        return loss.mean()


def class_weights(counts, beta: float = 0.0):
    """Trọng số lớp từ số ảnh TRAIN. beta=0: 1/n_c chuẩn hoá trung bình 1;
    beta>0: class-balanced (1-beta)/(1-beta^n_c) chuẩn hoá tổng = số lớp."""
    n = torch.as_tensor(np.asarray(counts, dtype=np.float64))
    w = 1.0 / n if beta == 0 else (1.0 - beta) / (1.0 - torch.pow(torch.tensor(beta, dtype=torch.float64), n))
    return (w * len(n) / w.sum()).float()


def build_criterion(kind: str = "ce", **kw):
    """kind: ce | ls (smoothing) | focal (gamma, alpha) | ce_weighted (weight)."""
    if kind == "ce":
        return nn.CrossEntropyLoss()
    if kind == "ls":
        return LabelSmoothingCE(kw.get("smoothing", 0.1))
    if kind == "focal":
        return FocalLoss(kw.get("gamma", 2.0), kw.get("alpha"))
    if kind == "ce_weighted":
        return nn.CrossEntropyLoss(weight=kw["weight"])
    raise ValueError(f"loss không hợp lệ: {kind}")


def _rand_box(h: int, w: int, lam: float):
    cut = np.sqrt(1.0 - lam)
    ch, cw = int(h * cut), int(w * cut)
    cy, cx = np.random.randint(h), np.random.randint(w)
    y1, y2 = np.clip(cy - ch // 2, 0, h), np.clip(cy + ch // 2, 0, h)
    x1, x2 = np.clip(cx - cw // 2, 0, w), np.clip(cx + cw // 2, 0, w)
    return y1, y2, x1, x2


def mix_batch(x, y, alpha: float = 1.0, mode: str = "cutmix"):
    """Trộn batch. CutMix: lam tính lại theo DIỆN TÍCH THỰC của hộp sau khi cắt biên."""
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device)
    if mode == "mixup":
        xm = lam * x + (1 - lam) * x[perm]
    elif mode == "cutmix":
        h, w = x.shape[-2:]
        y1, y2, x1, x2 = _rand_box(h, w, lam)
        xm = x.clone()
        xm[:, :, y1:y2, x1:x2] = x[perm][:, :, y1:y2, x1:x2]
        lam = 1.0 - (y2 - y1) * (x2 - x1) / (h * w)
    else:
        raise ValueError(f"mode không hợp lệ: {mode}")
    return xm, (y, y[perm], lam)


def mixed_loss(criterion, logits, targets):
    y_a, y_b, lam = targets
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
