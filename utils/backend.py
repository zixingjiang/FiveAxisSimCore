import numpy as np
import taichi as ti
from utils.logger import Logger
from config import FLOAT_TYPE_NAME, TAICHI_BACKEND

logger = Logger(name="Backend")


def init_taichi():
    """Initialize Taichi with the configured backend."""
    backend = TAICHI_BACKEND.lower()
    default_fp = ti.f32 if FLOAT_TYPE_NAME == "float32" else ti.f64

    if backend == "auto":
        # Let Taichi automatically select the best backend
        ti.init(default_fp=default_fp)
        # Query which backend Taichi actually selected
        actual_backend = _get_actual_backend()
        logger._emit(
            "INFO",
            f"Taichi initialized with {actual_backend.upper()} backend (auto-selected)",
        )
        return actual_backend

    backend_map = {
        "cuda": ti.cuda,
        "vulkan": ti.vulkan,
        "metal": ti.metal,
        "opengl": ti.opengl,
        "cpu": ti.cpu,
    }

    arch = backend_map.get(backend, ti.cpu)
    ti.init(arch=arch, default_fp=default_fp)

    # Verify which backend was actually used
    actual_backend = _get_actual_backend()
    if actual_backend != backend:
        logger._emit(
            "WARN",
            f"{backend.upper()} unavailable, using {actual_backend.upper()} instead",
        )
    else:
        logger._emit(
            "INFO", f"Taichi initialized with {actual_backend.upper()} backend"
        )

    return actual_backend


def _get_actual_backend():
    """Query which backend Taichi is actually using."""
    cfg = ti.lang.impl.current_cfg()
    arch = cfg.arch
    arch_name_map = {
        ti.cuda: "cuda",
        ti.vulkan: "vulkan",
        ti.metal: "metal",
        ti.opengl: "opengl",
        ti.cpu: "cpu",
        ti.arm64: "cpu",  # ARM64 is a CPU backend
        ti.x64: "cpu",  # x64 is a CPU backend
    }
    return arch_name_map.get(arch, "cpu")


# Initialize Taichi
ACTIVE_BACKEND = init_taichi()
GPU_ENABLED = ACTIVE_BACKEND in ("cuda", "vulkan", "metal", "opengl")

# Use numpy as the array library (Taichi kernels operate on numpy arrays)
xp = np

# Uniform data type handling
DTYPE_FLOAT = getattr(np, FLOAT_TYPE_NAME)
TI_DTYPE_FLOAT = ti.f32 if FLOAT_TYPE_NAME == "float32" else ti.f64
logger._emit("INFO", f"Using data type: {FLOAT_TYPE_NAME}")


def as_xp_array(a):
    """Convert input to numpy array with the configured dtype."""
    return np.asarray(a, dtype=DTYPE_FLOAT)


def to_numpy(a):
    """Ensure array is a numpy array (compatibility function)."""
    if isinstance(a, np.ndarray):
        return a
    return np.asarray(a)


# ============================================================
# Taichi Kernels for common operations
# ============================================================


@ti.kernel
def ti_clip_kernel(
    arr: ti.types.ndarray(), lo: ti.f32, hi: ti.f32, out: ti.types.ndarray()
):
    """Clip array values between lo and hi."""
    for i in arr:
        out[i] = ti.min(ti.max(arr[i], lo), hi)


@ti.kernel
def ti_norm_axis1_kernel(arr: ti.types.ndarray(), out: ti.types.ndarray()):
    """Compute L2 norm along axis 1 for a 2D array."""
    for i in range(arr.shape[0]):
        s = 0.0
        for j in range(arr.shape[1]):
            s += arr[i, j] ** 2
        out[i] = ti.sqrt(s)


@ti.kernel
def ti_sum_axis1_kernel(arr: ti.types.ndarray(), out: ti.types.ndarray()):
    """Compute sum along axis 1 for a 2D array."""
    for i in range(arr.shape[0]):
        s = 0.0
        for j in range(arr.shape[1]):
            s += arr[i, j]
        out[i] = s


@ti.kernel
def ti_dot_2d_1d_kernel(
    arr2d: ti.types.ndarray(), arr1d: ti.types.ndarray(), out: ti.types.ndarray()
):
    """Compute dot product of each row of 2D array with 1D array."""
    for i in range(arr2d.shape[0]):
        s = 0.0
        for j in range(arr2d.shape[1]):
            s += arr2d[i, j] * arr1d[j]
        out[i] = s
