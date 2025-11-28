import numpy as np
from utils.logger import Logger
from config import FLOAT_TYPE_NAME

try:
    from config import USE_GPU
except ImportError:
    USE_GPU = False

logger = Logger(name="Backend")


def get_backend():
    """Return xp, GPU_ENABLED"""
    if not USE_GPU:
        logger._emit("INFO", "NumPy backend active (CPU)")
        return np, False

    try:
        import cupy as cp

        cp.cuda.Device(0).use()
        cp.array([0.0])
        logger._emit("INFO", "CuPy backend active (GPU)")
        return cp, True
    except Exception as e:
        logger._emit("WARN", f"CUDA unavailable, fallback to NumPy: {e}")
        return np, False


# Import xp and GPU_ENABLED for easy access
xp, GPU_ENABLED = get_backend()

# Uniform data type handling
DTYPE_FLOAT = getattr(xp, FLOAT_TYPE_NAME)
logger._emit("INFO", f"Using data type: {FLOAT_TYPE_NAME}")

def as_xp_array(a):
    return xp.asarray(a, dtype=DTYPE_FLOAT)
