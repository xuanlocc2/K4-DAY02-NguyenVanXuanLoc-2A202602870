"""train.py - vòng huấn luyện cho mọi thí nghiệm (B, T, F). MỘT hàm `run(cfg)` dùng chung.

    python train.py --set exp_id=B01 backbone=resnet50 seed=0
Chỉ số chọn checkpoint (macro-F1 val) tính bằng eval.compute_metrics của repo (cùng định nghĩa lúc chấm).
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

import dataset as D
import losses as L
import model as M
from evalpath import ev


@dataclass
class Config:
    # --- định danh ---
    exp_id: str = "T00"
    tag: str = ""                     # mô tả ngắn cho tên ảnh curves (mặc định = backbone)
    seed: int = 0
    fold: int = 0
    # --- mô hình ---
    backbone: str = "resnet50"
    init: str = "finetune"            # scratch | frozen | finetune
    drop_rate: float = 0.0
    # --- dữ liệu / augmentation ---
    img_size: int = 224
    aug: str = "basic"                # basic | color | trivial | randaug | flips
    sampler: str | None = None        # None | balanced
    mix: str | None = None            # None | mixup | cutmix
    mix_alpha: float = 1.0
    # --- loss ---
    loss: str = "ce"                  # ce | ls | focal | ce_weighted
    label_smoothing: float = 0.1      # dùng khi loss=ls
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None
    # --- tối ưu (công thức nền, GUIDE.md mục 1.4) ---
    epochs: int = 12
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: float | None = None
    amp: bool = True
    num_workers: int = 2
    # --- đường dẫn ---
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"
    pred_dir: str = "predictions"
    curves_dir: str = "curves"
    # --- chỉ bật ở Bước 4 (chung kết): ghi predictions trên TEST. Mặc định TẮT (quy tắc S4). ---
    save_test_predictions: bool = False


def run_dir(cfg: Config) -> Path:
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def curve_path(cfg: Config) -> Path:
    name = f"{cfg.exp_id}_{cfg.tag or cfg.backbone}" + (f"_seed{cfg.seed}" if cfg.seed else "")
    return Path(cfg.curves_dir) / f"{name}.png"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cudnn.benchmark=True để nhanh; không bit-exact (ghi trong báo cáo)
    torch.backends.cudnn.benchmark = True


def build_optimizer(model, cfg: Config):
    return torch.optim.AdamW(M.param_groups(model, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay))


def build_scheduler(optimizer, cfg: Config, steps_per_epoch: int):
    """Warmup tuyến tính rồi cosine về 0, cập nhật THEO BƯỚC (iteration)."""
    total = cfg.epochs * steps_per_epoch
    warm = max(1, int(cfg.warmup_epochs * steps_per_epoch))

    def f(step):
        if step < warm:
            return (step + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


class EMA:
    """W_ema <- d W_ema + (1-d) W, d tăng dần min(decay, (1+n)/(10+n)). Buffer (BN stats) sao chép từ model."""

    def __init__(self, model, decay: float):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay, self.n = decay, 0

    @torch.no_grad()
    def update(self, model) -> None:
        self.n += 1
        d = min(self.decay, (1 + self.n) / (10 + self.n))
        for e, p in zip(self.module.parameters(), model.parameters()):
            e.mul_(d).add_(p.detach(), alpha=1 - d)
        for eb, b in zip(self.module.buffers(), model.buffers()):
            eb.copy_(b)


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, cfg: Config,
                    device, ema: EMA | None = None) -> dict:
    M.set_train_mode(model)  # giữ BN backbone ở eval nếu đóng băng
    tot, n = torch.zeros((), device=device), 0
    use_amp = cfg.amp and device.type == "cuda"
    for x, y, _ in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if cfg.mix:
            x, targets = L.mix_batch(x, y, cfg.mix_alpha, cfg.mix)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(x)
        loss = L.mixed_loss(criterion, logits, targets) if cfg.mix else criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        if ema is not None:
            ema.update(model)
        tot += loss.detach() * len(y)
        n += len(y)
    return {"train_loss": float(tot) / n, "lr": optimizer.param_groups[0]["lr"]}


@torch.inference_mode()
def evaluate(model, loader, criterion, device, amp: bool = True):
    """Eval (model.eval, không gradient). Trả (filenames, y_true, logits[N,9], loss). Giữ thứ tự loader."""
    model.eval()
    names, ys, zs, tot = [], [], [], 0.0
    use_amp = amp and device.type == "cuda"
    for x, y, fn in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            z = model(x)
        z = z.float()
        tot += criterion(z, y).item() * len(y)
        names += list(fn)
        ys.append(y.cpu())
        zs.append(z.cpu())
    y_true, logits = torch.cat(ys).numpy(), torch.cat(zs).numpy()
    return names, y_true, logits, tot / len(y_true)


def softmax_np(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def split_metrics(y_true, logits) -> dict:
    p = softmax_np(logits.astype(np.float64))
    return ev.compute_metrics(y_true, p.argmax(1), p)


def plot_curves(history: list[dict], path: str | Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h = pd.DataFrame(history)
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    ax[0].plot(h.epoch, h.train_loss, "o-", label="train loss")
    ax[0].plot(h.epoch, h.val_loss, "s-", label="val loss (CE)")
    ax[0].set(xlabel="epoch", ylabel="loss", title="Loss")
    ax[1].plot(h.epoch, h.val_macro_f1, "s-", c="tab:green", label="val macro-F1")
    ax[1].plot(h.epoch, h.val_top1, "^--", c="tab:orange", label="val top-1")
    ax[1].set(xlabel="epoch", ylabel="metric", title="Val metrics")
    ax[2].plot(h.epoch, h.lr, "o-", c="tab:red", label="LR (cuối epoch)")
    ax[2].set(xlabel="epoch", ylabel="LR", title="LR (warmup + cosine, theo bước)")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend()
    fig.suptitle(title)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def build_loaders(cfg: Config, tr, va, te=None):
    ntr = dict(num_workers=cfg.num_workers)
    train_loader = D.make_loader(tr, cfg.images_dir, D.build_transforms(True, cfg.img_size, cfg.aug),
                                 cfg.batch_size, True, cfg.sampler, seed=cfg.seed, **ntr)
    ev_tf = D.build_transforms(False, cfg.img_size)
    val_loader = D.make_loader(va, cfg.images_dir, ev_tf, cfg.batch_size * 2, False, seed=cfg.seed, **ntr)
    test_loader = None if te is None else D.make_loader(te, cfg.images_dir, ev_tf, cfg.batch_size * 2,
                                                         False, seed=cfg.seed, **ntr)
    return train_loader, val_loader, test_loader


def make_criterion(cfg: Config, tr, device):
    if cfg.loss == "ls":
        return L.build_criterion("ls", smoothing=cfg.label_smoothing)
    if cfg.loss == "focal":
        return L.build_criterion("focal", gamma=cfg.focal_gamma)
    if cfg.loss == "ce_weighted":
        counts = tr["Label"].value_counts().sort_index().values  # chỉ từ TRAIN
        w = L.class_weights(counts, cfg.class_weight_beta or 0.0).to(device)
        return L.build_criterion("ce_weighted", weight=w)
    return L.build_criterion("ce")


def run(cfg: Config) -> dict:
    """Huấn luyện một cấu hình, lưu config/history/curves/checkpoint/logit/predictions. Trả dict tóm tắt."""
    set_seed(cfg.seed)
    rd = run_dir(cfg)
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=1), encoding="utf-8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tr, va, te = D.load_split(cfg.labels_dir, cfg.fold)
    chk = D.check_split(tr, va, te, cfg.images_dir)
    (rd / "split_check.json").write_text(json.dumps(chk, indent=1, default=str), encoding="utf-8")
    train_loader, val_loader, test_loader = build_loaders(cfg, tr, va, te if cfg.save_test_predictions else None)

    model = M.build_model(cfg.backbone, True, D.NUM_CLASSES, cfg.drop_rate, cfg.init).to(device)
    criterion = make_criterion(cfg, tr, device)
    val_criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")
    ema = EMA(model, cfg.ema_decay) if cfg.ema_decay else None
    eval_model = ema.module if ema else model

    history, best_f1, best_epoch, t_train = [], -1.0, -1, 0.0
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.perf_counter()
        info = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, cfg, device, ema)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        t_train += dt
        _, yv, zv, vloss = evaluate(eval_model, val_loader, val_criterion, device, cfg.amp)
        m = split_metrics(yv, zv)
        history.append({"epoch": epoch, **info, "val_loss": vloss, "val_macro_f1": m["macro_f1"],
                        "val_top1": m["top1"], "epoch_sec": dt})
        print(f"[{cfg.exp_id} s{cfg.seed}] ep{epoch:02d} train {info['train_loss']:.4f} val {vloss:.4f} "
              f"F1 {m['macro_f1']:.4f} top1 {m['top1']:.4f} ({dt:.0f}s)", flush=True)
        if m["macro_f1"] > best_f1 + 1e-12:  # hòa -> giữ epoch sớm hơn
            best_f1, best_epoch = m["macro_f1"], epoch
            torch.save(eval_model.state_dict(), rd / "best.pt")

    eval_model.load_state_dict(torch.load(rd / "best.pt", map_location=device))
    names, yv, zv, _ = evaluate(eval_model, val_loader, val_criterion, device, cfg.amp)
    np.save(rd / "val_logits.npy", zv)
    pv = softmax_np(zv.astype(np.float64))
    ev.save_predictions(pred_path(cfg, "val"), names, yv, pv)
    mv = ev.compute_metrics(yv, pv.argmax(1), pv)
    if cfg.save_test_predictions:  # đúng MỘT lần, sau khi checkpoint đã chốt bằng val
        nt, yt, zt, _ = evaluate(eval_model, test_loader, val_criterion, device, cfg.amp)
        np.save(rd / "test_logits.npy", zt)
        ev.save_predictions(pred_path(cfg, "test"), nt, yt, softmax_np(zt.astype(np.float64)))

    pd.DataFrame(history).to_csv(rd / "history.csv", index=False)
    plot_curves(history, curve_path(cfg), f"{cfg.exp_id} | {cfg.backbone} | seed {cfg.seed}")

    import benchmark as B
    try:
        lat = B.latency_report(eval_model, 1, cfg.img_size, "fp32", device.type, 10, 50)["p50"]
    except Exception as e:  # đo độ trễ sơ bộ không được làm hỏng lần chạy
        print("latency lỗi:", e)
        lat = float("nan")
    summary = {
        "exp_id": cfg.exp_id, "seed": cfg.seed, "backbone": cfg.backbone, "weight_tag": model.weight_tag,
        "params_m": M.count_params(model), "gmacs": M.count_gmacs(eval_model, cfg.img_size),
        "img_size": cfg.img_size, "epochs": cfg.epochs, "best_epoch": best_epoch,
        "val_macro_f1": mv["macro_f1"], "val_top1": mv["top1"], "val_ece": mv["ece"],
        "val_f1_per_class": [float(v) for v in mv["f1"]],
        "sec_per_epoch": t_train / cfg.epochs, "latency_b1_p50_ms": lat,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "torch": torch.__version__, "config": dataclasses.asdict(cfg),
    }
    (rd / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


def load_run(rd: str | Path, device=None, **cfg_over):
    """Dựng lại (cfg, model đã nạp best.pt, eval mode) từ thư mục một lần chạy."""
    rd = Path(rd)
    d = json.loads((rd / "config.json").read_text(encoding="utf-8"))
    d.update(cfg_over)
    cfg = Config(**d)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = M.build_model(cfg.backbone, False, D.NUM_CLASSES, cfg.drop_rate, "finetune")
    model.load_state_dict(torch.load(rd / "best.pt", map_location="cpu"))
    return cfg, model.to(device).eval()


def test_from_checkpoint(rd: str | Path, **cfg_over) -> Path:
    """Chạy TEST đúng một lần cho một run đã huấn luyện (best.pt đã chốt bằng val). Từ chối nếu đã có file."""
    cfg, model = load_run(rd, **cfg_over)
    out = pred_path(cfg, "test")
    if out.exists():
        raise FileExistsError(f"{out} đã tồn tại: test chỉ chạy MỘT lần/seed (quy tắc S4)")
    device = next(model.parameters()).device
    _, _, te = D.load_split(cfg.labels_dir, cfg.fold)
    loader = D.make_loader(te, cfg.images_dir, D.build_transforms(False, cfg.img_size),
                           cfg.batch_size * 2, False, seed=cfg.seed, num_workers=cfg.num_workers)
    names, yt, zt, _ = evaluate(model, loader, nn.CrossEntropyLoss(), device, cfg.amp)
    np.save(Path(rd) / "test_logits.npy", zt)
    ev.save_predictions(out, names, yt, softmax_np(zt.astype(np.float64)))
    return out


def parse_overrides(pairs: list[str]) -> dict:
    """['seed=1','loss=focal','ema_decay=none'] -> dict ép kiểu theo field của Config."""
    types = {f.name: f.type for f in dataclasses.fields(Config)}
    out = {}
    for p in pairs:
        if "=" not in p:
            raise ValueError(f"cần KEY=VALUE, nhận '{p}'")
        k, v = p.split("=", 1)
        if k not in types:
            raise KeyError(f"'{k}' không có trong Config; các key hợp lệ: {sorted(types)}")
        t = str(types[k])
        if "None" in t and v.lower() in ("none", "null"):
            out[k] = None
        elif t.startswith("bool"):
            out[k] = v.lower() in ("1", "true", "yes")
        elif t.startswith("int"):
            out[k] = int(v)
        elif t.startswith("float"):
            out[k] = float(v)
        elif "float" in t:  # float | None
            out[k] = float(v)
        else:
            out[k] = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()
    cfg = Config(**parse_overrides(args.set))
    s = run(cfg)
    print(json.dumps({k: v for k, v in s.items() if k not in ("config", "val_f1_per_class")}, indent=1))


if __name__ == "__main__":
    main()
