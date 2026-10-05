"""finalize.py - Bước 4: chạy TEST đúng MỘT lần cho mỗi seed của cấu hình chung kết, với phương pháp suy luận
đã chốt trên val. T khớp trên VAL (trên logit trung bình các view), rồi áp lên từng view.

    python finalize.py --exp F01 --views flip --space prob --temperature
Ghi predictions/F01_seed<k>_{val,test}.csv (đã hiệu chuẩn nếu --temperature) và F01_uncal_seed<k>_test.csv (T=1).
Từ chối chạy lại nếu file test của seed đó đã tồn tại.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import dataset as D
import inference as I
import train as T
from evalpath import ev


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="F01")
    ap.add_argument("--views", default="center", choices=["center", "flip", "crop5", "crop5flip", "full"])
    ap.add_argument("--space", default="prob", choices=["prob", "logit"])
    ap.add_argument("--temperature", action="store_true")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--base", nargs="*", default=[], help="KEY=VALUE ghi đè đường dẫn/num_workers")
    a = ap.parse_args()
    over = T.parse_overrides(a.base)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    log = {}
    for rd in sorted(Path(a.runs, a.exp).glob("seed*")):
        cfg, model = T.load_run(rd, **over)
        tp = Path(cfg.pred_dir) / f"{a.exp}_seed{cfg.seed}_test.csv"
        if tp.exists():
            raise FileExistsError(f"{tp} đã tồn tại: test chỉ chạy MỘT lần/seed (quy tắc S4)")
        _, va, te = D.load_split(cfg.labels_dir, cfg.fold)
        mk = lambda df: D.make_loader(df, cfg.images_dir, D.build_transforms(False, 256), 128, False,  # noqa: E731
                                      seed=cfg.seed, num_workers=cfg.num_workers)
        views = I.view_set(a.views)
        nv, yv, zv = I.predict_views(model, mk(va), dev, views, amp=cfg.amp)
        Tv = I.fit_temperature(np.mean(list(zv.values()), 0), yv) if a.temperature else 1.0
        agg = lambda zs, t: (I.aggregate_views([z / t for z in zs.values()], a.space))  # noqa: E731
        nt, yt, zt = I.predict_views(model, mk(te), dev, views, amp=cfg.amp)  # <- lần test duy nhất
        Path(cfg.pred_dir).mkdir(exist_ok=True)
        ev.save_predictions(Path(cfg.pred_dir) / f"{a.exp}_seed{cfg.seed}_val.csv", nv, yv, agg(zv, Tv))
        ev.save_predictions(tp, nt, yt, agg(zt, Tv))
        ev.save_predictions(Path(cfg.pred_dir) / f"{a.exp}_uncal_seed{cfg.seed}_test.csv", nt, yt, agg(zt, 1.0))
        log[cfg.seed] = {"T": Tv, "views": a.views, "space": a.space}
        print(f"seed{cfg.seed}: T = {Tv:.3f} -> {tp}", flush=True)
    Path("results").mkdir(exist_ok=True)
    Path("results", f"{a.exp}_finalize.json").write_text(json.dumps(log, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
