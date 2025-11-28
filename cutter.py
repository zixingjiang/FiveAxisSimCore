import numpy as np
import taichi as ti
import open3d as o3d
from sdfer import Sdfer, plot_multiple_sdfer
from utils.logger import Logger
from utils.backend import xp, GPU_ENABLED, DTYPE_FLOAT


# ============================================================
# Taichi Kernels for SDF computation
# ============================================================


@ti.kernel
def ti_sdf_cylinder(
    pts: ti.types.ndarray(),
    origin: ti.types.ndarray(),
    direction: ti.types.ndarray(),
    radius: ti.f32,
    length: ti.f32,
    sdf_out: ti.types.ndarray(),
):
    """Compute SDF for a cylinder cutter."""
    for i in range(pts.shape[0]):
        # Relative position
        rx = pts[i, 0] - origin[0]
        ry = pts[i, 1] - origin[1]
        rz = pts[i, 2] - origin[2]

        # Height along direction
        h = rx * direction[0] + ry * direction[1] + rz * direction[2]

        # Radial distance
        r2 = rx * rx + ry * ry + rz * rz - h * h
        r = ti.sqrt(ti.max(r2, 0.0))

        R = radius

        # Outside region distance
        dh_top = ti.max(h - length, 0.0)
        dh_bottom = ti.max(-h, 0.0)
        dr = ti.max(r - R, 0.0)
        outside = ti.sqrt(dr * dr + ti.max(dh_top, dh_bottom) ** 2)

        # Inside region distance
        dist_side = R - r
        dist_top = length - h
        dist_bottom = h
        inside = -ti.min(ti.min(dist_side, dist_top), dist_bottom)

        # Combine
        if r <= R and h >= 0 and h <= length:
            sdf_out[i] = inside
        else:
            sdf_out[i] = outside


@ti.kernel
def ti_sdf_ball(
    pts: ti.types.ndarray(),
    origin: ti.types.ndarray(),
    direction: ti.types.ndarray(),
    radius: ti.f32,
    length: ti.f32,
    sdf_out: ti.types.ndarray(),
):
    """Compute SDF for a ball-end cutter."""
    for i in range(pts.shape[0]):
        rx = pts[i, 0] - origin[0]
        ry = pts[i, 1] - origin[1]
        rz = pts[i, 2] - origin[2]

        h = rx * direction[0] + ry * direction[1] + rz * direction[2]
        r2 = rx * rx + ry * ry + rz * rz - h * h
        r = ti.sqrt(ti.max(r2, 0.0))

        R = radius
        h_clamped = ti.max(ti.min(h, length), 0.0)

        # Radius at height (ball section: h < R, cylinder section: h >= R)
        # Initialize before conditional to ensure variable is defined
        R_at_h = R
        if h_clamped < R:
            R_at_h = ti.sqrt(ti.max(R * R - (R - h_clamped) ** 2, 0.0))

        # Outside distance
        dh_top = ti.max(h - length, 0.0)
        dh_bottom = ti.max(-h, 0.0)
        dr = ti.max(r - R_at_h, 0.0)
        outside = ti.sqrt(dr * dr + ti.max(dh_top, dh_bottom) ** 2)

        # Inside distance
        dist_side = R_at_h - r
        dist_top = length - h
        dist_bottom = h
        inside = -ti.min(ti.min(dist_side, dist_top), dist_bottom)

        if r <= R_at_h and h >= 0 and h <= length:
            sdf_out[i] = inside
        else:
            sdf_out[i] = outside


@ti.kernel
def ti_sdf_taper(
    pts: ti.types.ndarray(),
    origin: ti.types.ndarray(),
    direction: ti.types.ndarray(),
    radius_base: ti.f32,
    radius_tip: ti.f32,
    length: ti.f32,
    sdf_out: ti.types.ndarray(),
):
    """Compute SDF for a tapered cutter."""
    for i in range(pts.shape[0]):
        rx = pts[i, 0] - origin[0]
        ry = pts[i, 1] - origin[1]
        rz = pts[i, 2] - origin[2]

        h = rx * direction[0] + ry * direction[1] + rz * direction[2]
        r2 = rx * rx + ry * ry + rz * rz - h * h
        r = ti.sqrt(ti.max(r2, 0.0))

        h_clamped = ti.max(ti.min(h, length), 0.0)
        R = radius_tip + (radius_base - radius_tip) * (h_clamped / length)

        dh_top = ti.max(h - length, 0.0)
        dh_bottom = ti.max(-h, 0.0)
        dr = ti.max(r - R, 0.0)
        outside = ti.sqrt(dr * dr + ti.max(dh_top, dh_bottom) ** 2)

        dist_side = R - r
        dist_top = length - h
        dist_bottom = h
        inside = -ti.min(ti.min(dist_side, dist_top), dist_bottom)

        if r <= R and h >= 0 and h <= length:
            sdf_out[i] = inside
        else:
            sdf_out[i] = outside


@ti.kernel
def ti_cut_update_sdf(
    old_sdf: ti.types.ndarray(),
    tool_sdf: ti.types.ndarray(),
    new_sdf: ti.types.ndarray(),
    removed_count: ti.types.ndarray(),
):
    """Update SDF values after cutting and count removed voxels."""
    for i in range(old_sdf.shape[0]):
        new_val = ti.max(old_sdf[i], -tool_sdf[i])
        new_sdf[i] = new_val
        if old_sdf[i] <= 0 and new_val > 0:
            ti.atomic_add(removed_count[0], 1)


# ============================================================
# Base Cutter Class
# ============================================================
class BaseCutter:
    """Base cutter class defining the common interface. All cutters have the bottom center as origin and the Z-axis as the axis."""

    def __init__(self, length: DTYPE_FLOAT):
        self.length = DTYPE_FLOAT(length)

    def radius_at(self, h: DTYPE_FLOAT) -> DTYPE_FLOAT:
        """返回高度 h 处刀具的半径"""
        raise NotImplementedError

    # ======== 生成刀具 SDF ========
    def get_sdfer(self, points, origin, direction, vis=False, shape=None):
        """生成刀具的符号距离场 (SDF)"""
        sdf_flat = self._sdf_points(points, origin, direction)
        sdfer = Sdfer(points, sdf_flat, shape)
        if vis:
            sdfer.plot_level(0.0)
        return sdfer

    # ======== 刀具几何核心计算 ========
    def _sdf_points(
        self, pts_flat: np.ndarray, origin: np.ndarray, direction: np.ndarray
    ) -> np.ndarray:
        """Compute the SDF values of the cutter at the specified points."""

        d = direction / np.linalg.norm(direction)

        # Basic geometric relations
        rel = pts_flat - origin
        h = rel @ d
        r2 = (
            np.sum(rel**2, axis=1) - h**2
        )  # Equivalent to np.linalg.norm(rel - np.outer(h, d), axis=1)**2, but faster
        r = np.sqrt(np.maximum(r2, 0))

        # Radius distribution
        R = np.asarray(self.radius_at(np.clip(h, 0, self.length)))

        # Outside region distance
        dh_top = np.maximum(h - self.length, 0)
        dh_bottom = np.maximum(-h, 0)
        dr = np.maximum(r - R, 0)
        outside = np.sqrt(dr**2 + np.maximum(dh_top, dh_bottom) ** 2)

        # Inside region distance
        dist_side = R - r
        dist_top = self.length - h
        dist_bottom = h
        inside = -np.minimum(np.minimum(dist_side, dist_top), dist_bottom)

        # Combine overall SDF
        sdf = np.where((r <= R) & (h >= 0) & (h <= self.length), inside, outside)

        return sdf

    # ======== Cutting operation ========
    def cut_inplace(
        self,
        sdfer_workpiece: Sdfer,
        origin: np.ndarray,
        direction: np.ndarray,
        margin: DTYPE_FLOAT = 1.0,
    ) -> int:
        """Perform cutting operation in the local region of the cutter (in-place modification of sdfer_workpiece)"""
        d = direction / np.linalg.norm(direction)
        L = self.length
        Rmax = max(
            getattr(self, "radius_base", 0.0),
            getattr(self, "radius", 0.0),
            getattr(self, "radius_tip", 0.0),
        )
        margin = DTYPE_FLOAT(margin)

        # --- Block coarse filtering ---
        rel = sdfer_workpiece.block_centers - origin
        h = rel @ d
        r2 = np.sum(rel**2, axis=1) - h**2
        r = np.sqrt(np.maximum(r2, 0))

        # Effective expanded radius = block half diagonal
        block_r = sdfer_workpiece.block_size * np.sqrt(3) / 2
        mask_block = (
            (h >= -margin - block_r)
            & (h <= L + margin + block_r)
            & (r <= Rmax + margin + block_r)
        )
        hit_blocks = np.nonzero(mask_block)[0]
        if hit_blocks.size == 0:
            return 0

        starts = sdfer_workpiece.block_starts[hit_blocks]
        counts = sdfer_workpiece.block_counts[hit_blocks]

        cand_idx = np.concatenate(
            [sdfer_workpiece.block_order[s : s + c] for s, c in zip(starts, counts)]
        )
        if cand_idx.size == 0:
            return 0

        # --- Filter inactive voxels ---
        active_mask = sdfer_workpiece.active_mask[cand_idx]
        if not np.any(active_mask):
            return 0
        cand_idx = cand_idx[active_mask]

        # --- Geometric local filtering ---
        pts = sdfer_workpiece.pts[cand_idx]
        rel = pts - origin
        h = rel @ d
        r2 = np.sum(rel**2, axis=1) - h**2
        r = np.sqrt(np.maximum(r2, 0))

        local_mask = (h >= -margin) & (h <= L + margin) & (r <= Rmax + margin)
        if not np.any(local_mask):
            return 0

        idx_final = cand_idx[local_mask]
        pts_sub = sdfer_workpiece.pts[idx_final]
        sdf_tool = self._sdf_points(pts_sub, origin, direction)

        # --- Update in-place ---
        old = sdfer_workpiece.sdf[idx_final]
        new = np.maximum(old, -sdf_tool)
        removed = (old <= 0) & (new > 0)
        sdfer_workpiece.sdf[idx_final] = new
        sdfer_workpiece.deactivate_outer_volume()

        return int(np.count_nonzero(removed))


# ============================================================
# Ball cutter
# ============================================================
class BallCutter(BaseCutter):
    """Ball-end cylindrical cutter with fixed radius."""

    def __init__(self, radius: DTYPE_FLOAT, length: DTYPE_FLOAT):
        super().__init__(length)
        self.radius = radius

    def radius_at(self, h):
        """
        Return the cross-sectional radius at height h.
        Conventions:
            - origin is the bottom of the ball (h = 0)
            - 0 ≤ h ≤ radius     → hemisphere
            - radius ≤ h ≤ length → cylindrical section
        Hemisphere formula:
            r(h) = sqrt(R^2 - (R - h)^2)
        """
        R = self.radius
        h = np.asarray(h, dtype=DTYPE_FLOAT)

        # 半球段
        mask_ball = h < R
        r = np.empty_like(h, dtype=DTYPE_FLOAT)

        # 半球公式
        r_ball = np.sqrt(np.maximum(R * R - (R - h) ** 2, 0))

        # 柱段：半径恒定 R
        r_cyl = np.full_like(h, R, dtype=DTYPE_FLOAT)

        r[mask_ball] = r_ball[mask_ball]
        r[~mask_ball] = r_cyl[~mask_ball]

        return r

    def _sdf_points(
        self, pts_flat: np.ndarray, origin: np.ndarray, direction: np.ndarray
    ) -> np.ndarray:
        """Compute SDF using Taichi kernel for ball cutter."""
        d = direction / np.linalg.norm(direction)
        pts_f32 = pts_flat.astype(np.float32)
        origin_f32 = origin.astype(np.float32)
        d_f32 = d.astype(np.float32)
        sdf_out = np.zeros(pts_flat.shape[0], dtype=np.float32)

        ti_sdf_ball(
            pts_f32, origin_f32, d_f32, float(self.radius), float(self.length), sdf_out
        )
        return sdf_out.astype(DTYPE_FLOAT)


# ============================================================
# Tapered and flat-bottom cutters
# ============================================================
class TapperCutter(BaseCutter):
    """Tapered flat-bottom cutter with different base and tip radii."""

    def __init__(
        self, radius_base: DTYPE_FLOAT, radius_tip: DTYPE_FLOAT, length: DTYPE_FLOAT
    ):
        super().__init__(length)
        assert radius_base >= radius_tip
        self.radius_base = radius_base
        self.radius_tip = radius_tip

    def radius_at(self, h):
        """Linearly interpolate the radius at height h"""
        return self.radius_tip + (self.radius_base - self.radius_tip) * (
            h / self.length
        )

    def _sdf_points(
        self, pts_flat: np.ndarray, origin: np.ndarray, direction: np.ndarray
    ) -> np.ndarray:
        """Compute SDF using Taichi kernel for taper cutter."""
        d = direction / np.linalg.norm(direction)
        pts_f32 = pts_flat.astype(np.float32)
        origin_f32 = origin.astype(np.float32)
        d_f32 = d.astype(np.float32)
        sdf_out = np.zeros(pts_flat.shape[0], dtype=np.float32)

        ti_sdf_taper(
            pts_f32,
            origin_f32,
            d_f32,
            float(self.radius_base),
            float(self.radius_tip),
            float(self.length),
            sdf_out,
        )
        return sdf_out.astype(DTYPE_FLOAT)


class CylinderCutter(BaseCutter):
    """Cylindrical flat-bottom cutter with fixed radius."""

    def __init__(self, radius: DTYPE_FLOAT, length: DTYPE_FLOAT):
        super().__init__(length)
        self.radius = radius

    def radius_at(self, h):
        """Fixed radius"""
        return np.full_like(h, self.radius, dtype=DTYPE_FLOAT)

    def _sdf_points(
        self, pts_flat: np.ndarray, origin: np.ndarray, direction: np.ndarray
    ) -> np.ndarray:
        """Compute SDF using Taichi kernel for cylinder cutter."""
        d = direction / np.linalg.norm(direction)
        pts_f32 = pts_flat.astype(np.float32)
        origin_f32 = origin.astype(np.float32)
        d_f32 = d.astype(np.float32)
        sdf_out = np.zeros(pts_flat.shape[0], dtype=np.float32)

        ti_sdf_cylinder(
            pts_f32, origin_f32, d_f32, float(self.radius), float(self.length), sdf_out
        )
        return sdf_out.astype(DTYPE_FLOAT)


# ============================================================
# Self-test
# ============================================================
if __name__ == "__main__":
    cutter = TapperCutter(radius_base=0.5, radius_tip=0.2, length=8.0)
    ball_cutter = BallCutter(radius=0.5, length=8.0)
    pitch = 0.1
    mesh = o3d.geometry.TriangleMesh.create_box(width=8, height=8, depth=8)
    mesh.translate([-4, -4, 0])
    sdfer_workpiece = Sdfer.from_mesh(mesh, pitch, block_size=4 * pitch)

    sdfer_cutter = cutter.get_sdfer(
        sdfer_workpiece.pts,
        np.array([0, 0, 0]),
        np.array([0, 1, 1]),
        vis=False,
        shape=sdfer_workpiece.shape,
    )
    sdfer_ball_cutter = ball_cutter.get_sdfer(
        sdfer_workpiece.pts,
        np.array([0, 0, 1]),
        np.array([0, -1, 1]),
        vis=False,
        shape=sdfer_workpiece.shape,
    )

    plot_multiple_sdfer(
        [sdfer_workpiece, sdfer_cutter, sdfer_ball_cutter], opacities=[0.3, 0.7, 0.7]
    )

    chip = ball_cutter.cut_inplace(
        sdfer_workpiece, np.array([0, 0, 0]), np.array([0, 0, 1]), margin=2 * pitch
    )

    plot_multiple_sdfer([sdfer_workpiece], opacities=[0.9])
