"""make_results.py - gom kết quả THẬT (runs/, results/, predictions/) thành results.xlsx (7 sheet).
Chạy từ thư mục chứa runs/ results/ predictions/:  python make_results.py
Ô nào chưa có dữ liệu thì để trống, không điền số giả."""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evalpath import ev

CLASSES = ["Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia", "Rubber Vine",
           "Siam Weed", "Snake Weed", "Negatives"]


def summaries():
    rows = []
    for p in sorted(glob.glob("runs/*/seed*/summary.json")):
        s = json.load(open(p, encoding="utf-8"))
        c = s["config"]
        rows.append({"exp_id": s["exp_id"], "seed": s["seed"], "backbone": s["backbone"], "weight_tag": s["weight_tag"],
                     "params_M": round(s["params_m"], 2), "GMACs": round(s["gmacs"], 2), "img_size": s["img_size"],
                     "epochs": s["epochs"], "best_epoch": s["best_epoch"], "val_macro_f1": s["val_macro_f1"],
                     "val_top1": s["val_top1"], "val_ece": s["val_ece"], "sec_per_epoch": s["sec_per_epoch"],
                     "latency_b1_p50_ms": s["latency_b1_p50_ms"], "gpu": s["gpu"], "torch": s["torch"],
                     "aug": c["aug"], "loss": c["loss"], "sampler": c["sampler"], "ema": c["ema_decay"], "mix": c["mix"]})
    return pd.DataFrame(rows)


def test_group(pattern):
    ms = []
    for f in sorted(glob.glob(pattern)):
        r = ev.read_pred(f)
        ms.append((r.seed, ev.compute_metrics(r.y_true, r.y_pred, r.probs)))
    return ms


def agg(ms, keys=("top1", "macro_f1", "balanced_acc", "ece", "nll")):
    out = {"n_seeds": len(ms), "seeds": str([s for s, _ in ms])}
    for k in keys:
        v = np.array([m[k] for _, m in ms])
        out[f"{k}_mean"] = v.mean() if len(v) else np.nan
        out[f"{k}_std"] = v.std(ddof=1) if len(v) > 1 else np.nan
    return out


def main():
    Path("results").mkdir(exist_ok=True)
    S = summaries()
    sheets = {}
    sheets["Backbones"] = S[S.exp_id.str.startswith("B")].drop(columns=["aug", "loss", "sampler", "ema", "mix"]) if len(S) else S
    sheets["Training"] = S[S.exp_id.str.startswith("T")] if len(S) else S
    sheets["Inference"] = pd.read_csv("results/inference_val.csv") if Path("results/inference_val.csv").exists() else pd.DataFrame()
    sheets["Latency"] = pd.read_csv("results/latency.csv") if Path("results/latency.csv").exists() else pd.DataFrame()

    groups = {"T00 (mốc, I00)": "predictions/T00_seed*_test.csv", "F01 (chung kết)": "predictions/F01_seed*_test.csv",
              "F01 chưa temperature": "predictions/F01_uncal_seed*_test.csv"}
    fin, pc = [], []
    for name, pat in groups.items():
        ms = test_group(pat)
        if not ms:
            continue
        fin.append({"config": name, **agg(ms)})
        rec = np.array([m["recall"] for _, m in ms])
        f1 = np.array([m["f1"] for _, m in ms])
        for i, c in enumerate(CLASSES):
            pc.append({"config": name, "class": c, "recall_mean": rec[:, i].mean(), "recall_std": rec[:, i].std(ddof=1) if len(ms) > 1 else np.nan,
                       "f1_mean": f1[:, i].mean(), "support": int(ms[0][1]["support"][i])})
    vm = test_group("predictions/F01_seed*_val.csv")
    if vm and fin:
        fin.append({"config": "F01 (val, đã hiệu chuẩn)", **agg(vm)})
    sheets["Final"] = pd.DataFrame(fin)
    sheets["PerClass"] = pd.DataFrame(pc)

    summ = []
    d = {r["config"]: r for r in fin}
    if "F01 (chung kết)" in d:
        f = d["F01 (chung kết)"]
        summ += [("Top-1 test F01 (mean)", f["top1_mean"]), ("Macro-F1 test F01 (mean)", f["macro_f1_mean"]),
                 ("ECE test F01 (sau T)", f["ece_mean"])]
    if "F01 chưa temperature" in d:
        summ.append(("ECE test F01 (trước T)", d["F01 chưa temperature"]["ece_mean"]))
    if "T00 (mốc, I00)" in d:
        summ.append(("Macro-F1 test T00 (mean)", d["T00 (mốc, I00)"]["macro_f1_mean"]))
    sheets["Summary"] = pd.DataFrame(summ, columns=["metric", "value"])

    with pd.ExcelWriter("results.xlsx") as w:
        for k in ["Backbones", "Training", "Inference", "Final", "PerClass", "Latency", "Summary"]:
            sheets[k].to_excel(w, sheet_name=k, index=False)
    print({k: len(v) for k, v in sheets.items()})


if __name__ == "__main__":
    main()
