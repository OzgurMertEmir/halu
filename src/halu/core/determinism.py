from __future__ import annotations
import os, random, hashlib
import numpy as np

try:
    import torch
except Exception:
    torch = None

# --- public API --------------------------------------------------------------

def set_seed_and_register(seed: int = 1337, deterministic: bool = True) -> None:
    """
    Set seeds for Python, NumPy, (optionally) Torch, and register seed globally.
    Call this AS EARLY AS POSSIBLE in your entrypoint / orchestration.
    """
    # Python & NumPy
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Torch (if available)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if deterministic:
            # Prefer strict determinism; warn_only avoids hard errors on rare kernels.
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass
            try:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            except Exception:
                pass

            # Deterministic cuBLAS (CUDA ≥ 10.2); use one of the two approved settings.
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    # Tell our utils about the seed (centralizes any module-level RNG usage)
    try:
        from halu.core import utils as _utils
        _utils.SEED = seed  # override the default
    except Exception:
        pass

def stable_hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def stable_hash_arr(a: np.ndarray) -> str:
    # Hash shape + dtype + raw bytes
    h = hashlib.sha256()
    h.update(str(a.shape).encode("utf-8"))
    h.update(str(a.dtype).encode("utf-8"))
    h.update(np.ascontiguousarray(a).view(np.uint8))
    return h.hexdigest()

def stable_hash_df(df) -> str:
    """
    Hash a DataFrame deterministically: dtypes, columns, index, and values.
    """
    import pandas as pd
    assert isinstance(df, pd.DataFrame)
    h = hashlib.sha256()
    h.update(",".join(map(str, df.columns)).encode())
    h.update(",".join(map(str, df.dtypes)).encode())
    # Sort by index/columns to avoid order noise if caller desires stability
    vals = df.sort_index(axis=0).sort_index(axis=1).to_numpy(copy=True)
    h.update(np.ascontiguousarray(vals).view(np.uint8))
    return h.hexdigest()