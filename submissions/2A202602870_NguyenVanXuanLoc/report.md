# Báo cáo Lab Day 2 - DeepWeeds (fold 0)

**Tình trạng: chạy rút gọn, chưa đủ yêu cầu.** Mọi số dưới đây lấy từ các lần chạy thật trên Kaggle T4 (2 epoch mỗi run, batch 64, AdamW, lr 1e-4 backbone / 1e-3 head, wd 0.05, AMP, warmup 1 epoch + cosine). 2 epoch là quá ngắn nên các mô hình CNN chưa hội tụ; kết quả không phản ánh khả năng thật của chúng.

## Bước 1 - Backbone (val, seed 0) - `results.xlsx` sheet Backbones
| Backbone | Params (M) | GMACs | Val macro-F1 | Val top-1 | s/epoch | p50 b1 (ms) |
|---|---|---|---|---|---|---|
| resnet50 | 23.5 | 4.09 | 0.456 | 0.659 | 46 | 6.0 |
| convnext_tiny | 27.8 | 4.45 | **0.936** | 0.951 | 60 | 5.5 |
| vit_small_patch16_224 | 21.7 | 4.60 | 0.914 | 0.937 | 31 | 4.9 |
| mobilenetv3_large_100 | 4.2 | 0.22 | 0.481 | 0.625 | 42 | 6.1 |
| efficientnet_b0 | 4.0 | 0.38 | 0.569 | 0.676 | 50 | 7.8 |

Nhận xét: sau 2 epoch, ConvNeXt-T và ViT-S (pretrained mạnh hơn, LayerNorm) vượt xa ResNet-50/MobileNet/EfficientNet; khoảng cách này nhiều khả năng do chưa hội tụ (lr thấp, BN), không nên kết luận về kiến trúc. Độ trễ p50 batch 1 ở đây là đo sơ bộ (warmup 10, 50 lần, có synchronize) lúc huấn luyện.

## Bước 2 - Công thức huấn luyện: **KHÔNG hoàn thành**
Các ablation (T01 aug, T03 label smoothing, T05 sampler cân bằng) nằm trong `run_all.py` nhưng bị bỏ qua do hết ngân sách thời gian của kernel. Không có kết luận về trục huấn luyện nào.

## Bước 3 - Suy luận: **KHÔNG hoàn thành**
`infer_all.py` (TTA, ensemble, FP16, gộp BN, độ trễ p50/p95/p99) chưa chạy được trên Kaggle, nên không có sheet Inference/Latency và không có cấu hình p95 <= 100 ms được đo đúng cách. Chỉ có temperature scaling (T khớp trên val) trong `finalize.py`.

## Bước 4 - Chung kết (test, mỗi seed một lần)
Vì không có ablation, "chung kết" F01 chỉ là cấu hình mốc (ResNet-50, CE, aug basic) cộng temperature scaling, **không phải một cải tiến được chọn trên val**. Backbone tốt nhất theo val (ConvNeXt-T) không được dùng cho chung kết vì không còn thời gian huấn luyện lại.

| Cấu hình | Seed | Top-1 test | Macro-F1 test | ECE test |
|---|---|---|---|---|
| T00 mốc (resnet50) | 1, 2 (thiếu seed 0) | 0.668 +- 0.006 | 0.470 +- 0.012 | 0.065 |
| F01 (mốc + T) | 0 (chỉ 1 seed) | 0.651 | 0.442 | 0.033 |
| F01 chưa temperature | 0 | 0.651 | 0.442 | 0.074 |

- Temperature scaling giảm ECE test (0.074 -> 0.033) trên cùng dự đoán: đây là ý duy nhất có bằng chứng.
- Chênh lệch macro-F1 val (0.4545) và test (0.4418) của F01 là 0.013 (<= 0.02), chỉ 1 seed.
- Hai lớp khó Chinee Apple và Snake Weed: recall test 0.133 / 0.225 (F01), rất thấp, do mô hình chưa hội tụ. Parthenium gần như không nhận được (recall 0.017-0.044).
- F01 (seed 0) và T00 (seed 1, 2) khác seed nên so sánh F01 với T00 không có ý nghĩa thống kê; không khẳng định cải thiện.

## Hạn chế / phần thiếu (trung thực)
1. Chỉ 2 epoch/run; 3 seed không đủ: T00 có seed 1, 2; F01 có seed 0.
2. Không có T00 seed 0, không có ablation, không có sheet Inference/Latency đo đúng cách, các PNG curves chỉ có cho các run đã chạy.
3. Notebook `code/lab_day2.ipynb` vẫn là bản khung (các bước EDA/kiểm tra pipeline chưa điền).
4. Checkpoint và dữ liệu không được commit.
