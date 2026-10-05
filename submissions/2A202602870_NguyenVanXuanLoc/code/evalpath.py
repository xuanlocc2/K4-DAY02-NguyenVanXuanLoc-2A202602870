"""Import eval.py (repo gốc, KHÔNG sửa) từ bất kỳ thư mục con nào của repo."""
import sys
from pathlib import Path

for _p in Path(__file__).resolve().parents:
    if (_p / "eval.py").exists():
        sys.path.insert(0, str(_p))
        break
import eval as ev  # noqa: E402,F401
