"""dataset.py - đọc DeepWeeds, kiểm tra chia dữ liệu, transform, DataLoader.

Giao diện giữ đúng như khung:
    load_split(labels_dir, fold=0)            -> (train_df, val_df, test_df)
    check_split(train_df, val_df, test_df, images_dir) -> dict
    build_transforms(train, img_size, aug)    -> torchvision transform
    DeepWeedsDataset[i]                       -> (image_tensor, label:int, filename:str)
    make_loader(df, images_dir, transform, batch_size, train, sampler, num_workers)
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T

NUM_CLASSES = 9
# Thứ tự lớp theo cột `Label` của labels.csv (0 = Chinee Apple ... 7 = Snake Weed, 8 = Negatives).
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
TOTAL_IMAGES = 17509


def load_split(labels_dir: str | Path, fold: int = 0):
    """Đọc train/val/test_subset{fold}.csv nguyên bản (S1): không sửa, không lọc, không chia lại."""
    d = Path(labels_dir)
    return tuple(pd.read_csv(d / f"{s}_subset{fold}.csv") for s in ("train", "val", "test"))


def check_split(train_df, val_df, test_df, images_dir: str | Path) -> dict:
    """Kiểm tra bắt buộc (README mục 2.1). In ra, trả về dict, `assert` khi vi phạm."""
    sets = {"train": train_df, "val": val_df, "test": test_df}
    n = {k: len(v) for k, v in sets.items()}
    total = sum(n.values())
    per_class = {k: v["Label"].value_counts().sort_index().to_dict() for k, v in sets.items()}
    names = {k: set(v["Filename"]) for k, v in sets.items()}
    overlap = {
        "train&val": len(names["train"] & names["val"]),
        "train&test": len(names["train"] & names["test"]),
        "val&test": len(names["val"] & names["test"]),
    }
    union = len(names["train"] | names["val"] | names["test"])
    images_dir = Path(images_dir)
    missing = sum(1 for f in set().union(*names.values()) if not (images_dir / f).exists())
    ratio = {k: round(v / total, 4) for k, v in n.items()}

    print("so anh:", n, "| tong", total, "| ty le", ratio)
    print("so anh moi lop:")
    print(pd.DataFrame(per_class).fillna(0).astype(int).rename(index=dict(enumerate(CLASS_NAMES))))
    print("giao:", overlap, "| hop:", union, "| thieu file:", missing)

    assert all(v == 0 for v in overlap.values()), f"Rò rỉ dữ liệu giữa các tập: {overlap}"
    assert union == total == TOTAL_IMAGES, f"hợp {union}, tổng {total}, kỳ vọng {TOTAL_IMAGES}"
    assert missing == 0, f"{missing} file trong CSV không có trong {images_dir}"
    for k, target in (("train", 0.6), ("val", 0.2), ("test", 0.2)):
        assert abs(ratio[k] - target) < 0.01, f"tỉ lệ {k} = {ratio[k]} lệch khỏi {target}"
    return {"n": n, "ratio": ratio, "per_class": per_class, "overlap": overlap,
            "union": union, "missing": missing}


def build_transforms(train: bool, img_size: int = 224, aug: str = "basic"):
    """Transform cho train/val/test.

    Val/test: Resize(256) -> CenterCrop(img_size) -> ToTensor -> Normalize (không ngẫu nhiên).
    Với img_size=256 phép này giữ nguyên ảnh (ảnh gốc 256x256) - dùng cho TTA/dò độ phân giải.
    Train `aug`: basic (RandomResizedCrop + lật ngang) | color (+ColorJitter) | trivial
    (+TrivialAugmentWide) | randaug (+RandAugment) | flips (lật ngang + dọc + xoay 90 độ).
    Ảnh cỏ nhìn từ trên xuống nên lật dọc/xoay hợp lệ về mặt vật lý; trục B kiểm chứng điều này.
    """
    norm = [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if not train:
        return T.Compose([T.Resize(256), T.CenterCrop(img_size)] + norm)
    ops = [T.RandomResizedCrop(img_size, scale=(0.35, 1.0)), T.RandomHorizontalFlip()]
    if aug == "color":
        ops.append(T.ColorJitter(0.3, 0.3, 0.3, 0.05))
    elif aug == "trivial":
        ops.append(T.TrivialAugmentWide())
    elif aug == "randaug":
        ops.append(T.RandAugment())
    elif aug == "flips":
        ops += [T.RandomVerticalFlip(), T.RandomApply([T.RandomRotation((90, 90))], p=0.5)]
    elif aug != "basic":
        raise ValueError(f"aug không hợp lệ: {aug}")
    return T.Compose(ops + norm)


class DeepWeedsDataset(Dataset):
    """(ảnh đã transform, nhãn int, tên file str). Thứ tự theo df."""

    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None):
        self.files = df["Filename"].tolist()
        self.labels = df["Label"].astype(int).tolist()
        self.dir = Path(images_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int):
        img = Image.open(self.dir / self.files[i]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.labels[i]), self.files[i]


def _worker_init(worker_id: int) -> None:
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


def make_loader(df: pd.DataFrame, images_dir: str | Path, transform, batch_size: int,
                train: bool, sampler: str | None = None, num_workers: int = 2, seed: int = 0):
    """DataLoader. Eval: không shuffle, giữ thứ tự df. Train: shuffle hoặc sampler cân bằng lớp."""
    ds = DeepWeedsDataset(df, images_dir, transform)
    g = torch.Generator()
    g.manual_seed(seed)
    kw = dict(num_workers=num_workers, pin_memory=torch.cuda.is_available(),
              worker_init_fn=_worker_init, generator=g, persistent_workers=num_workers > 0)
    if not train:
        return DataLoader(ds, batch_size=batch_size, shuffle=False, **kw)
    if sampler == "balanced":
        counts = df["Label"].value_counts()
        w = df["Label"].map(lambda c: 1.0 / counts[c]).to_numpy()
        sp = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=len(df),
                                   replacement=True, generator=g)
        return DataLoader(ds, batch_size=batch_size, sampler=sp, drop_last=True, **kw)
    if sampler is not None:
        raise ValueError(f"sampler không hợp lệ: {sampler}")
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True, **kw)
