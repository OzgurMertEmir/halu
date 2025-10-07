# tests/conftest.py
import os, sys, random, pathlib
import numpy as np

# --- Make "src" importable so `import halu` works in tests
def _ensure_src_on_path():
    root = pathlib.Path(__file__).resolve().parents[1]  # repo root
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
_ensure_src_on_path()

# --- Determinism knobs
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

def pytest_sessionstart(session):
    random.seed(1337)
    np.random.seed(1337)
    try:
        import torch
        torch.manual_seed(1337)
        # Leave deterministic_algorithms at default; we don't need it for pure-math tests.
    except Exception:
        pass