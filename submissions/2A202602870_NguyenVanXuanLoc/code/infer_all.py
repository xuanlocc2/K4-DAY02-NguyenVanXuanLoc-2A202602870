"""infer_all.py - Bước 3: so sánh phương pháp suy luận trên VAL (không đụng test) + độ trễ.

    python infer_all.py --run runs/T00/seed0 --b-glob "runs/B0*/seed0"
Ghi: results/inference_val.csv, results/latency.csv, results/inference_tradeoff.png, results/inference_chosen.json
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import benchmark as B
import dataset as D
import inference as I
import train as T
from evalpath import ev


def metrics(y, p):
    m = ev.compute_metrics(y, p.argmax(1), p)
    return {"val_macro_f1": m["macro_f1"], "val_top1": m["top1"], "val_ece": m["ece"],
            "val_f1_chinee": float(m["f1"][0]), "val_f1_snake": float(m["f1"][7])}


def lat(model, k, img, dtype, device, tag):
    """Độ trễ batch 1 và 32 (p50/p95/p99) cho K lượt forward."""
    out = {}
    for bs in (1, 32):
        r = B.tta_latency(model, k, batch_size=bs, img_size=img, dtype=dtype, device=device) if k > 1 else \
            B.latency_report(model, bs, img, dtype, device)
        out[bs] = r
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/T00/seed0")
    ap.add_argument("--b-glob", default="runs/B0*/seed0")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg, model = T.load_run(a.run)
    _, va, _ = D.load_split(cfg.labels_dir, cfg.fold)
    loader = D.make_loader(va, cfg.images_dir, D.build_transforms(False, 256), 128, False, seed=0,
                           num_workers=cfg.num_workers)

    # --- một lượt duyệt val, mọi view cần thiết (fp32)
    views = {**I.view_set("crop5flip"), "c": I.view_set("center")["c"], "f": I.view_set("flip")["f"],
             "full": I.view_identity,
             "up288": lambda x: I.views_multiscale(I.center_crop(x, 224), [288])[0]}
    _, y, z = I.predict_views(model, loader, dev, views, amp=False)
    sm = I.aggregate_views
    Tc = I.fit_temperature(z["c"], y)
    print(f"T (val) = {Tc:.3f}")
    crops = [f"k{i}" for i in range(5)]
    allv = crops + [f"k{i}f" for i in range(5)]
    rows = [  # id, tên, probs, k view, img, dtype
        ("I00", "center-crop 224, fp32 (mốc)", sm([z["c"]]), 1, 224, "fp32"),
        ("I01", "hflip TTA, TB xác suất", sm([z["c"], z["f"]], "prob"), 2, 224, "fp32"),
        ("I02", "hflip TTA, TB logit", sm([z["c"], z["f"]], "logit"), 2, 224, "fp32"),
        ("I03", "5-crop TTA, TB xác suất", sm([z[k] for k in crops], "prob"), 5, 224, "fp32"),
        ("I04", "5-crop + hflip (10 view), TB xác suất", sm([z[k] for k in allv], "prob"), 10, 224, "fp32"),
        ("I05", f"temperature scaling (T={Tc:.2f}) trên I00", I.apply_temperature(z["c"], Tc), 1, 224, "fp32"),
        ("I06", "toàn ảnh 256 (không crop)", sm([z["full"]]), 1, 256, "fp32"),
        ("I07", "phóng 288", sm([z["up288"]]), 1, 288, "fp32"),
    ]
    # FP16 (autocast): đo chỉ số thật
    _, _, z16 = I.predict_logits(model, loader, dev, I.view_set("center")["c"], amp=True)
    rows.append(("I08", "FP16 (autocast), center 224", sm([z16]), 1, 224, "fp16"))
    # gộp Conv+BN
    fused = I.fuse_conv_bn(model).to(dev)
    _, _, zf = I.predict_logits(fused, loader, dev, I.view_set("center")["c"], amp=False)
    rows.append((f"I09", f"gộp Conv+BN ({fused.n_fused} cặp), fp32", sm([zf]), 1, 224, "fp32"))
    # ensemble các backbone B (cùng thứ tự val, từ val_logits.npy lúc huấn luyện)
    bruns = sorted(glob.glob(a.b_glob))
    best = []
    for r in bruns:
        zb = np.load(Path(r) / "val_logits.npy")
        best.append((metrics(y, I.apply_temperature(zb, 1.0))["val_macro_f1"], r, zb))
    best.sort(key=lambda t: -t[0])
    ens = None
    if len(best) >= 2:
        top = best[:2]
        ens = ("I10", f"ensemble 2 backbone ({', '.join(Path(t[1]).parent.name for t in top)})",
               I.ensemble_probs([I.apply_temperature(t[2], 1.0) for t in top]), 2, 224, "fp32")
        rows.append(ens)

    lat_rows, res = [], []
    for eid, name, p, k, img, dt in rows:
        if eid == "I10":
            ms = [T.load_run(t[1])[1] for t in best[:2]]
            ls = [lat(m, 1, 224, "fp32", dev, eid) for m in ms]
            L = {bs: {q: sum(l[bs][q] for l in ls) for q in ("p50", "p95", "p99")} for bs in (1, 32)}
            note = "tổng độ trễ các thành viên (ước lượng, chạy tuần tự)"
            for bs in (1, 32):
                lat_rows.append({"id": eid, "method": name, "gpu": ls[0][bs]["gpu"], "dtype": dt, "batch": bs,
                                 "img_size": img, **L[bs], "images_per_s": bs / (L[bs]["p50"] / 1000),
                                 "torch": torch.__version__, "note": note})
            lb1 = L[1]
        else:
            m = fused if eid == "I09" else model
            ls = lat(m, k, img, dt, dev, eid)
            for bs in (1, 32):
                r = ls[bs]
                lat_rows.append({"id": eid, "method": name, "gpu": r["gpu"], "dtype": dt, "batch": bs,
                                 "img_size": img, "p50": r["p50"], "p95": r["p95"], "p99": r["p99"],
                                 "images_per_s": r["images_per_s"], "torch": r["torch"], "note": f"{k} lượt forward"})
            lb1 = ls[1]
        res.append({"id": eid, "method": name, "k_views": k, **metrics(y, p),
                    "b1_p50_ms": lb1["p50"], "b1_p95_ms": lb1["p95"], "b1_p99_ms": lb1["p99"]})
        print(f"{eid} {name}: F1 {res[-1]['val_macro_f1']:.4f} ECE {res[-1]['val_ece']:.4f} "
              f"p95@b1 {lb1['p95']:.1f} ms", flush=True)

    df = pd.DataFrame(res)
    df.to_csv(out / "inference_val.csv", index=False)
    pd.DataFrame(lat_rows).to_csv(out / "latency.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(df.b1_p95_ms, df.val_macro_f1)
    for _, r in df.iterrows():
        ax.annotate(r["id"], (r.b1_p95_ms, r.val_macro_f1), fontsize=8, xytext=(3, 3), textcoords="offset points")
    ax.axvline(100, ls="--", c="r", lw=0.8, label="ngân sách 100 ms")
    ax.set(xlabel="p95 độ trễ, batch 1 (ms)", ylabel="macro-F1 (val)", title="Độ chính xác - độ trễ")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "inference_tradeoff.png", dpi=130)

    ok = df[df.b1_p95_ms <= 100]
    cheapest_best = df.sort_values(["val_macro_f1", "b1_p95_ms"], ascending=[False, True]).iloc[0]
    rt = ok.sort_values(["val_macro_f1", "b1_p95_ms"], ascending=[False, True]).iloc[0] if len(ok) else None
    chosen = {"final_by_val": cheapest_best.to_dict(), "realtime_p95_le_100ms": None if rt is None else rt.to_dict(),
              "temperature": Tc, "base_run": a.run, "gpu": lat_rows[0]["gpu"]}
    (out / "inference_chosen.json").write_text(json.dumps(chosen, indent=1, default=float), encoding="utf-8")
    print(df[["id", "method", "val_macro_f1", "val_ece", "b1_p95_ms"]].to_string(index=False))
    print("Chọn theo val:", cheapest_best["id"], "|", cheapest_best["method"])


if __name__ == "__main__":
    main()
