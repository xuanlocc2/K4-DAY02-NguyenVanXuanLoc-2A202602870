"""run_all.py - chạy theo giai đoạn, có resume (bỏ qua run đã có summary.json).

    python run_all.py --stage B --epochs 5                      # 5 backbone (Bước 1)
    python run_all.py --stage T --backbone resnet50 --epochs 5  # T00 + các ablation (Bước 2)
    python run_all.py --stage final --backbone resnet50 --epochs 5 --over loss=ls ema_decay=0.999
    python run_all.py --stage test_t00 --epochs 5 ...            # TEST cho T00 seed 0 từ checkpoint
Mọi giai đoạn dùng cùng công thức nền; mỗi ablation CHỈ đổi một trục so với T00.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import train as T

BACKBONES = {  # exp_id -> backbone timm (ResNet, ConvNeXt, transformer, nhẹ, EfficientNet)
    "B01": "resnet50", "B02": "convnext_tiny", "B03": "vit_small_patch16_224",
    "B04": "mobilenetv3_large_100", "B05": "efficientnet_b0",
}
ABLATIONS = {  # exp_id -> (trục, override so với T00)
    "T01": ("augmentation", {"aug": "trivial"}),
    "T02": ("augmentation", {"aug": "color"}),
    "T03": ("loss", {"loss": "ls", "label_smoothing": 0.1}),
    "T04": ("loss", {"loss": "focal", "focal_gamma": 2.0}),
    "T05": ("imbalance", {"sampler": "balanced"}),
    "T06": ("imbalance", {"loss": "ce_weighted", "class_weight_beta": 0.99}),
    "T07": ("ema", {"ema_decay": 0.999}),
    "T08": ("mix", {"mix": "cutmix", "mix_alpha": 1.0}),
}


def go(**kw) -> None:
    cfg = T.Config(**kw)
    if (T.run_dir(cfg) / "summary.json").exists():
        print(f"[skip] {cfg.exp_id} seed{cfg.seed} đã có kết quả")
        return
    T.run(cfg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["B", "T", "final", "test_t00"])
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--backbone", default="resnet50")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--only", nargs="*", help="chỉ chạy các exp_id này")
    ap.add_argument("--over", nargs="*", default=[], help="KEY=VALUE cho cấu hình final")
    ap.add_argument("--base", nargs="*", default=[], help="KEY=VALUE áp cho MỌI run (vd num_workers=4)")
    a = ap.parse_args()
    base = dict(epochs=a.epochs, **T.parse_overrides(a.base))
    want = lambda k: a.only is None or k in a.only  # noqa: E731

    if a.stage == "B":
        for k, bb in BACKBONES.items():
            if want(k):
                go(exp_id=k, backbone=bb, **base)
    elif a.stage == "T":
        if want("T00"):
            go(exp_id="T00", backbone=a.backbone, **base)
        for k, (axis, ov) in ABLATIONS.items():
            if want(k):
                go(exp_id=k, tag=f"{a.backbone}_{axis}", backbone=a.backbone, **ov, **base)
    elif a.stage == "test_t00":
        print("ghi:", T.test_from_checkpoint(Path("runs/T00/seed0"), **T.parse_overrides(a.base)))
    else:  # final: T00 seed 1,2 (+test) và F01 seed 0,1,2 (test do finalize.py)
        ov = T.parse_overrides(a.over)
        for s in a.seeds:
            if s > 0 and want("T00"):
                go(exp_id="T00", backbone=a.backbone, seed=s, save_test_predictions=True, **base)
            # F01: KHÔNG ghi test ở đây; finalize.py chạy test đúng một lần với phương pháp suy luận đã chọn
            if want("F01"):
                go(exp_id="F01", tag=f"{a.backbone}_final", backbone=a.backbone, seed=s, **{**base, **ov})


if __name__ == "__main__":
    main()
