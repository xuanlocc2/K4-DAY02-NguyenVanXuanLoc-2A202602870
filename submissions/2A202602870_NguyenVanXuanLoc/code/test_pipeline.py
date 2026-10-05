"""Kiểm tra nhanh trên CPU với dữ liệu giả (không tải pretrained): python -m unittest test_pipeline -v
Bắt lỗi pipeline TRƯỚC khi tốn GPU Kaggle."""
import math
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

import dataset as D
import inference as I
import losses as L
import model as M
import train as T
from evalpath import ev

torch.set_num_threads(2)


def make_fake_data(root: Path, n=100, size=64):
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    rng = np.random.RandomState(0)
    rows = []
    for i in range(n):
        lab = i % 9
        arr = (rng.rand(size, size, 3) * 60 + lab * 20).clip(0, 255).astype("uint8")  # lớp phân biệt được theo độ sáng
        Image.fromarray(arr).save(root / "images" / f"{i}.jpg")
        rows.append({"Filename": f"{i}.jpg", "Label": lab, "Species": str(lab)})
    df = pd.DataFrame(rows)
    for name, sl in (("train", slice(0, 60)), ("val", slice(60, 80)), ("test", slice(80, 100))):
        df.iloc[sl].to_csv(root / "labels" / f"{name}_subset0.csv", index=False)


class TestLosses(unittest.TestCase):
    def test_focal_gamma0_equals_ce(self):
        z, y = torch.randn(16, 9), torch.randint(0, 9, (16,))
        self.assertAlmostEqual(L.FocalLoss(0.0)(z, y).item(), F.cross_entropy(z, y).item(), places=5)

    def test_ls0_equals_ce(self):
        z, y = torch.randn(16, 9), torch.randint(0, 9, (16,))
        self.assertAlmostEqual(L.LabelSmoothingCE(0.0)(z, y).item(), F.cross_entropy(z, y).item(), places=5)

    def test_initial_loss_near_ln9(self):
        z, y = torch.zeros(32, 9), torch.randint(0, 9, (32,))
        self.assertAlmostEqual(F.cross_entropy(z, y).item(), math.log(9), places=5)

    def test_cutmix_lambda_is_area(self):
        x = torch.arange(4 * 3 * 32 * 32, dtype=torch.float32).reshape(4, 3, 32, 32)
        y = torch.arange(4)
        np.random.seed(1)
        xm, (_, _, lam) = L.mix_batch(x, y, 1.0, "cutmix")
        changed = (xm != x).any(1).float().mean(0)  # chỉ tính các vị trí khác
        self.assertTrue(0.0 <= lam <= 1.0)
        self.assertLessEqual(abs((1 - lam) - ((xm != x).any(1).float().mean().item())), 0.3)  # thô

    def test_class_weights_mean_one(self):
        w = L.class_weights([100, 10, 5, 50, 1, 20, 30, 40, 200], 0.0)
        self.assertAlmostEqual(w.mean().item(), 1.0, places=5)


class TestModel(unittest.TestCase):
    def test_param_groups_cover_all_trainable(self):
        m = M.build_model("resnet18", False, 9, 0.0, "scratch")
        g = M.param_groups(m, 1e-4, 1e-3, 0.05)
        self.assertEqual(sum(len(x["params"]) for x in g), sum(p.requires_grad for p in m.parameters()))
        self.assertEqual(g[-1]["lr"], 1e-3)

    def test_frozen_bn_stays_eval(self):
        m = M.build_model("resnet18", False, 9, 0.0, "frozen")
        M.set_train_mode(m)
        self.assertFalse(m.bn1.training)
        self.assertTrue(m.fc.training)
        self.assertEqual(sum(p.requires_grad for p in m.parameters()), 2)


class TestInference(unittest.TestCase):
    def test_fuse_conv_bn_matches(self):
        m = M.build_model("resnet18", False, 9, 0.0, "scratch").eval()
        for mod in m.modules():  # BN có thống kê khác mặc định
            if isinstance(mod, torch.nn.BatchNorm2d):
                mod.running_mean.normal_(0, 0.1)
                mod.running_var.uniform_(0.5, 1.5)
        f = I.fuse_conv_bn(m)
        self.assertGreater(f.n_fused, 0)
        self.assertLess(f.fuse_max_err, 1e-3)

    def test_temperature_reduces_nll_on_overconfident(self):
        torch.manual_seed(0)
        y = torch.randint(0, 9, (500,))
        z = F.one_hot(y, 9).float() * 3 + torch.randn(500, 9) * 2
        z = z * 4  # quá tự tin
        T_ = I.fit_temperature(z.numpy(), y.numpy())
        self.assertGreater(T_, 1.0)
        nll = lambda t: F.cross_entropy(z / t, y).item()  # noqa: E731
        self.assertLess(nll(T_), nll(1.0))

    def test_views_shapes(self):
        x = torch.randn(2, 3, 256, 256)
        self.assertEqual(len(I.views_multicrop(x, 224)), 5)
        self.assertEqual(I.center_crop(x, 224).shape[-1], 224)
        self.assertEqual(I.views_multiscale(x, [224, 288])[1].shape[-1], 288)

    def test_aggregate_rows_sum_to_one(self):
        zs = [np.random.randn(5, 9), np.random.randn(5, 9)]
        for sp in ("prob", "logit"):
            np.testing.assert_allclose(I.aggregate_views(zs, sp).sum(1), 1.0, atol=1e-6)


class TestEndToEnd(unittest.TestCase):
    def test_run_two_epochs_and_test_once(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            make_fake_data(td)
            D.TOTAL_IMAGES = 100  # dữ liệu giả
            kw = dict(backbone="resnet18", init="scratch", epochs=2, batch_size=16, img_size=64, num_workers=0,
                      images_dir=str(td / "images"), labels_dir=str(td / "labels"), out_dir=str(td / "runs"),
                      pred_dir=str(td / "pred"), curves_dir=str(td / "curves"), amp=False, warmup_epochs=0.5)
            cfg = T.Config(exp_id="X00", save_test_predictions=True, ema_decay=0.9, mix="cutmix", **kw)
            s = T.run(cfg)
            self.assertTrue(0 <= s["val_macro_f1"] <= 1)
            for p in ("X00_seed0_val.csv", "X00_seed0_test.csv"):
                self.assertTrue((td / "pred" / p).exists(), p)
            self.assertTrue(any((td / "curves").glob("X00_*.png")))
            # file dự đoán phải đọc lại được bằng eval.py của repo
            r = ev.read_pred(str(td / "pred" / "X00_seed0_test.csv"))
            self.assertEqual(r.probs.shape, (20, 9))
            # test-from-checkpoint: chạy lần 2 phải bị chặn
            cfg2 = T.Config(exp_id="X01", **kw)
            T.run(cfg2)
            T.test_from_checkpoint(td / "runs" / "X01" / "seed0", **{k: kw[k] for k in
                                   ("images_dir", "labels_dir", "pred_dir", "num_workers")})
            with self.assertRaises(FileExistsError):
                T.test_from_checkpoint(td / "runs" / "X01" / "seed0", **{k: kw[k] for k in
                                       ("images_dir", "labels_dir", "pred_dir", "num_workers")})

    def test_overfit_one_batch(self):
        torch.manual_seed(0)
        m = M.build_model("resnet18", False, 9, 0.0, "scratch")
        x, y = torch.randn(16, 3, 64, 64), torch.randint(0, 9, (16,))
        opt = torch.optim.AdamW(m.parameters(), 1e-3)
        m.train()
        for _ in range(60):
            opt.zero_grad()
            loss = F.cross_entropy(m(x), y)
            loss.backward()
            opt.step()
        self.assertLess(loss.item(), 0.2)


if __name__ == "__main__":
    unittest.main()
