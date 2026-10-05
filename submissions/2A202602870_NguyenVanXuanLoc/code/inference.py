"""inference.py - các phương pháp suy luận (Bước 3 của GUIDE.md).

Mọi hàm chạy ở chế độ eval, không gradient. Chọn phương pháp CHỈ dựa trên val; nhiệt độ T khớp trên VAL.
Quy ước view: loader của suy luận trả ảnh FULL 256x256 (build_transforms(False, 256)); mỗi view là hàm
biến batch 256 -> tensor đưa vào model (ví dụ center-crop 224 = đúng tiền xử lý val lúc train).

    predict_logits(model, loader, device, view=None) -> (filenames, y_true, logits[N, 9])
    predict_views(model, loader, device, views)      -> (filenames, y_true, {tên view: logits})
    aggregate_views(list_of_logits, space)           -> probs[N, 9]
    fit_temperature(val_logits, val_labels)          -> float T
    apply_temperature(logits, T)                     -> probs
    ensemble_probs(list_of_probs)                    -> probs
    fuse_conv_bn(model)                              -> model (BN đã gộp vào conv)
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# ---------------------------------------------------------------- views


def view_identity(x):
    return x


def view_hflip(x):
    """Lật ngang batch (N, C, H, W)."""
    return torch.flip(x, dims=[3])


def center_crop(x, crop: int):
    h, w = x.shape[-2:]
    t, l = (h - crop) // 2, (w - crop) // 2
    return x[..., t:t + crop, l:l + crop]


def views_multicrop(x, crop: int):
    """5 crop (4 góc + giữa) kích thước `crop`. Trả list 5 batch."""
    h, w = x.shape[-2:]
    return [x[..., :crop, :crop], x[..., :crop, w - crop:], x[..., h - crop:, :crop],
            x[..., h - crop:, w - crop:], center_crop(x, crop)]


def views_multiscale(x, sizes):
    """Resize toàn ảnh về từng kích thước (bilinear + antialias). Cần model chấp nhận kích thước đó."""
    return [x if s == x.shape[-1] else F.interpolate(x, size=(s, s), mode="bilinear",
                                                      antialias=True, align_corners=False) for s in sizes]


# ---------------------------------------------------------------- dự đoán


@torch.inference_mode()
def predict_views(model, loader, device, views: dict, amp: bool = True):
    """Duyệt loader MỘT lần, áp dụng từng view. Trả (filenames, y_true, {tên: logits float32 numpy})."""
    model.eval()
    names, ys, out = [], [], {k: [] for k in views}
    use_amp = amp and torch.device(device).type == "cuda"
    for x, y, fn in loader:
        x = x.to(device, non_blocking=True)
        for k, v in views.items():
            with torch.autocast(torch.device(device).type, dtype=torch.float16, enabled=use_amp):
                out[k].append(model(v(x)).float().cpu())
        ys.append(y)
        names += list(fn)
    return names, torch.cat(ys).numpy(), {k: torch.cat(v).numpy() for k, v in out.items()}


def predict_logits(model, loader, device, view=None, amp: bool = True):
    fn, y, d = predict_views(model, loader, device, {"v": view or view_identity}, amp)
    return fn, y, d["v"]


# ---------------------------------------------------------------- gộp / hiệu chuẩn


def _softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def aggregate_views(logits_per_view, space: str = "prob"):
    """space='prob': trung bình softmax; 'logit': trung bình logit rồi softmax. Trả (N, 9)."""
    if space == "prob":
        return np.mean([_softmax(l) for l in logits_per_view], axis=0)
    if space == "logit":
        return _softmax(np.mean(logits_per_view, axis=0))
    raise ValueError(space)


def ensemble_probs(list_of_probs):
    """Trung bình xác suất. Các mô hình phải cùng tập ảnh, cùng thứ tự file."""
    return np.mean(list_of_probs, axis=0)


def fit_temperature(val_logits, val_labels) -> float:
    """T > 0 cực tiểu NLL trên VAL (LBFGS trên log T). Không khớp trên test."""
    z = torch.as_tensor(val_logits, dtype=torch.float64)
    y = torch.as_tensor(val_labels, dtype=torch.long)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(z / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp())


def apply_temperature(logits, T: float):
    return _softmax(np.asarray(logits, dtype=np.float64) / T)


# ---------------------------------------------------------------- gộp BN


@torch.no_grad()
def _fuse(conv: nn.Conv2d, bn) -> nn.Conv2d:
    std = torch.sqrt(bn.running_var + bn.eps)
    scale = bn.weight / std if bn.weight is not None else 1.0 / std
    shift = (bn.bias if bn.bias is not None else 0.0) - bn.running_mean * scale
    fused = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding,
                      conv.dilation, conv.groups, bias=True).to(conv.weight.device, conv.weight.dtype)
    fused.weight.copy_(conv.weight * scale.reshape(-1, 1, 1, 1))
    b = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)
    fused.bias.copy_(b * scale + shift)
    return fused


def _is_bn(m) -> bool:
    return isinstance(m, nn.BatchNorm2d) or (hasattr(m, "running_mean") and hasattr(m, "act")
                                             and getattr(m, "running_mean", None) is not None)


@torch.no_grad()
def fuse_conv_bn(model):
    """Gộp (Conv2d, BN liền sau) thành một Conv có bias. Trả về BẢN SAO đã gộp (model gốc giữ nguyên).
    timm dùng BatchNormAct2d (BN+activation): thay bằng activation của nó. Kiến trúc không có BN2d
    (ViT/Swin/ConvNeXt dùng LayerNorm) thì trả bản sao không đổi (n_fused = 0).
    In sai số lớn nhất của logit trước/sau gộp."""
    model = copy.deepcopy(model).eval()
    ref = copy.deepcopy(model)
    n = 0
    for parent in list(model.modules()):
        items = list(parent._modules.items())
        for (n1, m1), (n2, m2) in zip(items, items[1:]):
            if isinstance(m1, nn.Conv2d) and _is_bn(m2):
                parent._modules[n1] = _fuse(m1, m2)
                parent._modules[n2] = m2.act if hasattr(m2, "act") and m2.act is not None else nn.Identity()
                n += 1
    model.n_fused = n
    if n:
        p = next(model.parameters())
        x = torch.randn(2, 3, 224, 224, device=p.device, dtype=p.dtype)
        err = (model(x) - ref(x)).abs().max().item()
        print(f"fuse_conv_bn: gộp {n} cặp, sai số logit lớn nhất = {err:.2e}")
        model.fuse_max_err = err
    else:
        print("fuse_conv_bn: không có cặp Conv+BN2d nào (kiến trúc dùng LayerNorm?), không áp dụng")
        model.fuse_max_err = 0.0
    return model


# ---------------------------------------------------------------- bộ view dùng chung (infer_all / finalize)


def view_set(name: str, crop: int = 224) -> dict:
    """Bộ view trên ảnh 256x256. center | flip | crop5 | crop5flip | full."""
    c = lambda x: center_crop(x, crop)  # noqa: E731
    cf = lambda x: view_hflip(center_crop(x, crop))  # noqa: E731
    if name == "center":
        return {"c": c}
    if name == "full":
        return {"full": view_identity}
    if name == "flip":
        return {"c": c, "f": cf}
    crops = {f"k{i}": (lambda x, i=i: views_multicrop(x, crop)[i]) for i in range(5)}
    if name == "crop5":
        return crops
    if name == "crop5flip":
        return {**crops, **{f"{k}f": (lambda x, v=v: view_hflip(v(x))) for k, v in crops.items()}}
    raise ValueError(name)
