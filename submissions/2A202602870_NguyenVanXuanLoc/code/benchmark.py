"""benchmark.py - đo độ trễ suy luận đúng cách (warmup, synchronize, >= 50 lần, p50/p95/p99).

Điều kiện đo: chỉ đo forward của model trên tensor ngẫu nhiên đã nằm sẵn trên thiết bị
(KHÔNG gồm giải mã ảnh/tiền xử lý), eval + inference_mode.
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch


def bench(fn, warmup: int = 10, iters: int = 100, sync=None) -> dict:
    """Thời gian một lần gọi fn() tính bằng ms: p50/p95/p99/mean."""
    sync = sync or (lambda: None)
    for _ in range(warmup):
        fn()
    sync()
    ts = []
    for _ in range(iters):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        ts.append((time.perf_counter() - t0) * 1000)
    a = np.asarray(ts)
    return {"p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)), "mean": float(a.mean()), "n": iters}


def _prep(model, dtype: str, device: str):
    m = copy.deepcopy(model).to(device).eval()
    if dtype == "fp16":
        m = m.half()
    return m


def latency_report(model, batch_size: int, img_size: int, dtype: str = "fp32", device: str = "cuda",
                   warmup: int = 10, iters: int = 100) -> dict:
    """dtype: fp32 | amp (autocast fp16) | fp16 (model.half()). Trả dict ghi thẳng vào sheet Latency."""
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    m = _prep(model, dtype, device)
    x = torch.randn(batch_size, 3, img_size, img_size, device=device,
                    dtype=torch.float16 if dtype == "fp16" else torch.float32)
    sync = torch.cuda.synchronize if device == "cuda" else None

    @torch.inference_mode()
    def fn():
        with torch.autocast(device, dtype=torch.float16, enabled=(dtype == "amp" and device == "cuda")):
            m(x)

    r = bench(fn, warmup, iters, sync)
    return {"gpu": torch.cuda.get_device_name(0) if device == "cuda" else "CPU", "dtype": dtype,
            "batch": batch_size, "img_size": img_size, **r,
            "images_per_s": batch_size / (r["p50"] / 1000), "torch": torch.__version__}


def tta_latency(model, k_views: int, **kw) -> dict:
    """Độ trễ TTA K view (K lượt forward batch 1 nối tiếp) và tỉ lệ so với K * p50 một view."""
    single = latency_report(model, **kw)
    device = "cuda" if torch.cuda.is_available() and kw.get("device", "cuda") == "cuda" else "cpu"
    dtype = kw.get("dtype", "fp32")
    m = _prep(model, dtype, device)
    x = torch.randn(kw["batch_size"], 3, kw["img_size"], kw["img_size"], device=device,
                    dtype=torch.float16 if dtype == "fp16" else torch.float32)
    sync = torch.cuda.synchronize if device == "cuda" else None

    @torch.inference_mode()
    def fn():
        with torch.autocast(device, dtype=torch.float16, enabled=(dtype == "amp" and device == "cuda")):
            for _ in range(k_views):
                m(x)

    r = bench(fn, kw.get("warmup", 10), kw.get("iters", 100), sync)
    return {**single, **{k: r[k] for k in ("p50", "p95", "p99", "mean")}, "k_views": k_views,
            "ratio_vs_k_times_single": r["p50"] / (k_views * single["p50"])}
