"""step0_checks.py - Bước 0: EDA + kiểm tra split + kiểm tra pipeline. Ghi results/step0_*.{txt,png}.
    python step0_checks.py [--images data/images --labels data/labels]"""
import argparse
import io
import math
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

import dataset as D
import model as M
import train as T

ap = argparse.ArgumentParser()
ap.add_argument("--images", default="data/images")
ap.add_argument("--labels", default="data/labels")
ap.add_argument("--out", default="results")
a = ap.parse_args()
out = Path(a.out)
out.mkdir(exist_ok=True)
T.set_seed(0)
log = []

# 1) split + kiểm tra bắt buộc
tr, va, te = D.load_split(a.labels, 0)
buf = io.StringIO()
with redirect_stdout(buf):
    chk = D.check_split(tr, va, te, a.images)
log.append(buf.getvalue())

# 2) phân bố lớp
cnt = pd.DataFrame({k: v["Label"].value_counts().sort_index() for k, v in
                    (("train", tr), ("val", va), ("test", te))}).rename(index=dict(enumerate(D.CLASS_NAMES)))
cnt["total"] = cnt.sum(1)
cnt["ty_le_tong"] = (cnt.total / cnt.total.sum()).round(4)
log.append("Phân bố lớp:\n" + cnt.to_string())
log.append(f"Negatives / tổng = {cnt.loc['Negatives','total'] / cnt.total.sum():.3f}; "
           f"Negatives / một loài trung bình = {cnt.loc['Negatives','total'] / cnt.total.iloc[:8].mean():.2f} (bài báo: 9.106)")
fig, ax = plt.subplots(figsize=(8, 4))
cnt[["train", "val", "test"]].plot.bar(stacked=True, ax=ax)
ax.set_title("Số ảnh mỗi lớp theo tập (fold 0)")
fig.tight_layout()
fig.savefig(out / "step0_class_distribution.png", dpi=130)
plt.close(fig)

# 3) >= 3 ảnh mỗi lớp
fig, axs = plt.subplots(9, 3, figsize=(7, 20))
for c in range(9):
    fns = tr[tr.Label == c].sample(3, random_state=0).Filename
    for j, f in enumerate(fns):
        axs[c, j].imshow(Image.open(Path(a.images) / f))
        axs[c, j].axis("off")
    axs[c, 0].set_title(D.CLASS_NAMES[c], fontsize=8, loc="left")
fig.tight_layout()
fig.savefig(out / "step0_samples.png", dpi=90)
plt.close(fig)

# 4) kiểm tra pipeline: loss đầu ~ ln 9, overfit một batch, augmentation
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
m = M.build_model("resnet18", True, 9, 0.0, "finetune").to(dev)
tf = D.build_transforms(True, 224, "color")
ld = D.make_loader(tr, a.images, tf, 32, True, seed=0, num_workers=2)
x, y, fn = next(iter(ld))
x, y = x.to(dev), y.to(dev)
M.set_train_mode(m)
with torch.no_grad():
    l0 = F.cross_entropy(m(x), y).item()
log.append(f"Loss ban đầu (head mới): {l0:.3f} | ln 9 = {math.log(9):.3f}")
opt = torch.optim.AdamW(m.parameters(), 1e-3)
for i in range(40):
    opt.zero_grad()
    loss = F.cross_entropy(m(x), y)
    loss.backward()
    opt.step()
log.append(f"Overfit một batch 32 ảnh: loss sau 40 bước = {loss.item():.4f}")
m.train()
a1 = m.training
m.eval()
log.append(f"model.train() -> training={a1}; model.eval() -> training={m.training}")
mean, std = torch.tensor(D.IMAGENET_MEAN)[:, None, None], torch.tensor(D.IMAGENET_STD)[:, None, None]
fig, axs = plt.subplots(2, 6, figsize=(14, 5))
for i, ax in enumerate(axs.ravel()):
    ax.imshow((x[i].cpu() * std + mean).clamp(0, 1).permute(1, 2, 0))
    ax.set_title(D.CLASS_NAMES[int(y[i])], fontsize=8)
    ax.axis("off")
fig.suptitle("Sau augmentation (đã giải chuẩn hoá) + nhãn")
fig.tight_layout()
fig.savefig(out / "step0_augmentation.png", dpi=100)
(out / "step0_log.txt").write_text("\n\n".join(log), encoding="utf-8")
print("\n\n".join(log))
