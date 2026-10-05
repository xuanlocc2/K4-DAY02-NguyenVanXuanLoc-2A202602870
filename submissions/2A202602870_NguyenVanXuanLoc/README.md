# Lab Day 2 (Track 4) - Nguyen Van Xuan Loc - 2A202602870

- Kaggle (chạy chính): https://www.kaggle.com/code/lcnguynvnxun/k4-day2-auto  
  Kaggle (T00 seed 1, 2): https://www.kaggle.com/code/lcnguynvnxun/k4-day2-auto-seeds
- Phần cứng: Kaggle Tesla T4. Thư viện: torch/timm của image Kaggle (xem `runs/*/seed*/summary.json`, trường `torch`).
- **Bài chưa hoàn chỉnh.** Chạy rút gọn (2 epoch mỗi run) vì hết thời gian trước hạn nộp. Phần thiếu được liệt kê trong `report.md`; không có số nào được điền giả.

## Cách chạy lại (trong `code/`, sau khi có `data/images`, `data/labels`)
```
python test_pipeline.py                      # kiểm tra nhanh trên CPU
python run_all.py --stage B --epochs 12      # 5 backbone
python run_all.py --stage T --backbone convnext_tiny --epochs 12
python run_all.py --stage final --backbone convnext_tiny --epochs 12 --over loss=ls
python infer_all.py --run runs/T00/seed0 && python finalize.py --exp F01 --temperature
python make_results.py
```
Seed: 0, 1, 2. Fold 0. `eval.py` không bị sửa.
