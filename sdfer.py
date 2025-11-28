import numpy as np
import taichi as ti
import open3d as o3d
from utils.logger import Logger
from utils.mesh_convert import as_mesh_type
import pyvista as pv
from utils.backend import xp, GPU_ENABLED, DTYPE_FLOAT, to_numpy

logger = Logger("Sdfer")


# ============================================================
# Taichi Kernels for SDF operations
# ============================================================


@ti.kernel
def ti_build_block_key(
    pts: ti.types.ndarray(),
    bmin: ti.types.ndarray(),
    pitch: ti.f32,
    block_size: ti.f32,
    keys: ti.types.ndarray(),
):
    """Compute block keys for spatial hashing."""
    for i in range(pts.shape[0]):
        bx = ti.floor((pts[i, 0] - bmin[0] + pitch / 2) / block_size)
        by = ti.floor((pts[i, 1] - bmin[1] + pitch / 2) / block_size)
        bz = ti.floor((pts[i, 2] - bmin[2] + pitch / 2) / block_size)
        keys[i] = (
            (ti.cast(bx, ti.int64) << 40)
            + (ti.cast(by, ti.int64) << 20)
            + ti.cast(bz, ti.int64)
        )


@ti.kernel
def ti_compute_block_centers(
    uniq_keys: ti.types.ndarray(),
    bmin: ti.types.ndarray(),
    block_size: ti.f32,
    centers: ti.types.ndarray(),
):
    """Compute block centers from unique keys."""
    for i in range(uniq_keys.shape[0]):
        k = uniq_keys[i]
        ix = (k >> 40) & ((1 << 20) - 1)
        iy = (k >> 20) & ((1 << 20) - 1)
        iz = k & ((1 << 20) - 1)
        centers[i, 0] = bmin[0] + ti.cast(ix, ti.f32) * block_size + block_size / 2
        centers[i, 1] = bmin[1] + ti.cast(iy, ti.f32) * block_size + block_size / 2
        centers[i, 2] = bmin[2] + ti.cast(iz, ti.f32) * block_size + block_size / 2


@ti.kernel
def ti_block_sphere_filter(
    block_centers: ti.types.ndarray(),
    origin: ti.types.ndarray(),
    direction: ti.types.ndarray(),
    block_half: ti.f32,
    radius: ti.f32,
    length: ti.f32,
    margin: ti.f32,
    mask: ti.types.ndarray(),
):
    """Filter blocks that may intersect with a cutter cylinder/sphere."""
    for i in range(block_centers.shape[0]):
        # Compute relative position
        rx = block_centers[i, 0] - origin[0]
        ry = block_centers[i, 1] - origin[1]
        rz = block_centers[i, 2] - origin[2]

        # Height along direction
        h = rx * direction[0] + ry * direction[1] + rz * direction[2]

        # Squared radial distance
        r2 = rx * rx + ry * ry + rz * rz - h * h
        r = ti.sqrt(ti.max(r2, 0.0))

        # Block diagonal radius for conservative test
        block_r = block_half * ti.sqrt(3.0)

        # Check if block may intersect
        in_h = (h >= -margin - block_r) and (h <= length + margin + block_r)
        in_r = r <= radius + margin + block_r

        mask[i] = 1 if (in_h and in_r) else 0


@ti.kernel
def ti_point_cylinder_filter(
    pts: ti.types.ndarray(),
    origin: ti.types.ndarray(),
    direction: ti.types.ndarray(),
    radius: ti.f32,
    length: ti.f32,
    margin: ti.f32,
    mask: ti.types.ndarray(),
):
    """Filter points that may be inside a cutter cylinder."""
    for i in range(pts.shape[0]):
        rx = pts[i, 0] - origin[0]
        ry = pts[i, 1] - origin[1]
        rz = pts[i, 2] - origin[2]

        h = rx * direction[0] + ry * direction[1] + rz * direction[2]
        r2 = rx * rx + ry * ry + rz * rz - h * h
        r = ti.sqrt(ti.max(r2, 0.0))

        in_h = (h >= -margin) and (h <= length + margin)
        in_r = r <= radius + margin

        mask[i] = 1 if (in_h and in_r) else 0


class Sdfer:
    """Open3D-based Signed Distance Field (SDF) calculator (flattened version)."""

    # ======== Main constructor: only responsible for storing data ========
    def __init__(
        self,
        pts_flat: np.ndarray,
        sdf_flat: np.ndarray,
        shape: tuple[int] | None = None,
        pitch: DTYPE_FLOAT | None = None,
        block_size: DTYPE_FLOAT | None = None,
    ):
        """
        Parameters:
        ----
        pts_flat : (N,3) ndarray
            Flattened array of all voxel center coordinates.
        sdf_flat : (N,) ndarray
            Signed distance corresponding to each voxel.
        shape : tuple[int], optional
            If provided, used for visualization reshape, order='F', (nx,ny,nz)
        pitch : DTYPE_FLOAT, optional
            Voxel edge length (used for subsequent calculations).
        block_size : DTYPE_FLOAT, optional
            Block size for block indexing (used to accelerate local clipping).
        """
        assert (
            pts_flat.ndim == 2 and pts_flat.shape[1] == 3
        ), "pts_flat shape must be (N, 3)"
        assert (
            sdf_flat.ndim == 1 and sdf_flat.shape[0] == pts_flat.shape[0]
        ), "sdf_flat length must match pts_flat rows"

        # Store as numpy arrays (Taichi operates on numpy arrays)
        self.pts = np.asarray(pts_flat, dtype=DTYPE_FLOAT)
        self.sdf = np.asarray(sdf_flat, dtype=DTYPE_FLOAT)
        self.shape = tuple(shape) if shape is not None else None
        self.bmin = np.min(self.pts, axis=0)
        self.bmax = np.max(self.pts, axis=0)
        self.active_mask = np.ones(len(self.sdf), dtype=bool)

        # === Generate voxel block index table for local clipping acceleration ===
        if pitch is not None:
            self.pitch = DTYPE_FLOAT(pitch)

        if block_size is not None:
            if not (block_size / self.pitch).is_integer():
                logger.error("block_size must be an integer multiple of pitch")
                raise SystemExit
            self.block_size = DTYPE_FLOAT(block_size)
            self._build_block_index()

    # ======== Factory method 1: Generate from mesh ========
    @classmethod
    def from_mesh(
        cls,
        mesh: o3d.geometry.TriangleMesh,
        pitch: DTYPE_FLOAT = 0.5,
        block_size: DTYPE_FLOAT | None = None,
    ):
        """
        从 Open3D mesh 生成符号距离场（自动检测 GPU）
        """
        # Open3D RaycastingScene 必须在 CPU 上运行
        o3c = o3d.core
        o3d_mesh = as_mesh_type(mesh, "open3d")
        mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(o3d_mesh)
        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(mesh_t)

        # Prepare voxel points
        aabb = mesh.get_axis_aligned_bounding_box()
        bmin, bmax = np.asarray(aabb.min_bound), np.asarray(aabb.max_bound)
        span = bmax - bmin
        n_cells = np.ceil(span / pitch).astype(int)
        # Expand by 1 layer
        bmin = bmin - pitch / 2
        bmax_aligned = bmin + n_cells * pitch + pitch / 2
        dims = n_cells + 2

        xs = np.linspace(bmin[0], bmax_aligned[0], dims[0])
        ys = np.linspace(bmin[1], bmax_aligned[1], dims[1])
        zs = np.linspace(bmin[2], bmax_aligned[2], dims[2])
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
        pts_flat = np.column_stack(
            (X.ravel(order="F"), Y.ravel(order="F"), Z.ravel(order="F"))
        )

        # Compute signed distance
        pts_t = o3c.Tensor(
            pts_flat, dtype=o3c.float32, device=o3c.Device("CPU:0")
        )  # Open3D RaycastingScene only supports CPU
        sdf_flat = scene.compute_signed_distance(pts_t).numpy().ravel(order="F")

        obj = cls(pts_flat, sdf_flat, shape=dims, pitch=pitch, block_size=block_size)

        return obj

    # ======== Factory method 2: Generate from data ========
    @classmethod
    def from_data(
        cls,
        pts_flat: xp.ndarray,
        sdf_flat: xp.ndarray,
        shape: tuple[int] | None = None,
        pitch: DTYPE_FLOAT | None = None,
        block_size: DTYPE_FLOAT | None = None,
    ):
        return cls(pts_flat, sdf_flat, shape, pitch=pitch, block_size=block_size)

    def deactivate_outer_volume(self):
        """Mark SDF outer voxels as inactive"""
        margin = self.pitch if hasattr(self, "pitch") else DTYPE_FLOAT(1.0)
        self.active_mask &= self.sdf <= margin

    def _build_block_index(self):
        """Build AABB block index table based on block_size"""
        bmin = self.bmin
        n_pts = self.pts.shape[0]

        # Use Taichi kernel to compute block keys
        key = np.zeros(n_pts, dtype=np.int64)
        ti_build_block_key(
            self.pts.astype(np.float32),
            bmin.astype(np.float32),
            float(self.pitch),
            float(self.block_size),
            key,
        )

        # Sort by key and record start and end indices
        order = np.argsort(key)
        self.block_key_sorted = key[order]
        self.block_order = order
        uniq_keys, idx_start, counts = np.unique(
            self.block_key_sorted, return_index=True, return_counts=True
        )
        self.block_keys = uniq_keys
        self.block_starts = idx_start
        self.block_counts = counts

        # Compute block centers using Taichi
        n_blocks = len(uniq_keys)
        centers = np.zeros((n_blocks, 3), dtype=DTYPE_FLOAT)
        ti_compute_block_centers(
            uniq_keys, bmin.astype(np.float32), float(self.block_size), centers
        )
        self.block_centers = centers
        self.block_half = np.full(n_blocks, self.block_size * 0.5, dtype=DTYPE_FLOAT)

    def get_block_indices(
        self, origin, radius, *, refine=False, debug=False
    ) -> np.ndarray:
        """
        Return candidate voxel indices covered by the radius.
        - Coarse filtering by block (match self.block_keys using searchsorted)
        - Optional refine: radius refinement using Euclidean distance, if enabled, only return voxel indices truly within the radius
        """
        origin = np.asarray(origin, dtype=DTYPE_FLOAT)
        r = DTYPE_FLOAT(radius)

        # ------- Calculate integer coordinate range of hit blocks -------
        bmin = self.bmin
        block_min = np.floor((origin - r - bmin) / self.block_size).astype(np.int64)
        block_max = np.floor((origin + r - bmin) / self.block_size).astype(np.int64)

        # Note: if r is very small, the three arange may have only 1 element
        bx = np.arange(int(block_min[0]), int(block_max[0]) + 1, dtype=np.int64)
        by = np.arange(int(block_min[1]), int(block_max[1]) + 1, dtype=np.int64)
        bz = np.arange(int(block_min[2]), int(block_max[2]) + 1, dtype=np.int64)

        # ------- Construct candidate block keys (ensure dtype matches self.block_keys) -------
        cand_keys = (
            (bx[:, None, None] << 40) + (by[None, :, None] << 20) + bz[None, None, :]
        ).ravel()
        cand_keys = cand_keys.astype(self.block_keys.dtype, copy=False)

        # ------- Binary search cand_keys in self.block_keys (which is sorted) -------
        # pos points to the insertion position; then filter out non-matching keys
        pos = np.searchsorted(self.block_keys, cand_keys)
        valid_mask = pos < self.block_keys.size
        pos = pos[valid_mask]
        key_match = self.block_keys[pos] == cand_keys[valid_mask]
        hit_blocks = pos[
            key_match
        ]  # These are the indices of "hit blocks" in self.block_keys

        if hit_blocks.size == 0:
            if debug:
                print("[get_block_indices] no block matched")
            return np.empty(0, dtype=np.int64)

        # ------- Map block indices to point index sets -------
        starts = self.block_starts[hit_blocks].astype(np.int64, copy=False)
        counts = self.block_counts[hit_blocks].astype(np.int64, copy=False)

        # Collect slices of each hit block, then concatenate
        segs = []
        for i in range(int(starts.size)):
            s = int(starts[i])
            c = int(counts[i])
            if c > 0:
                segs.append(self.block_order[s : s + c])
        if len(segs) == 0:
            if debug:
                print("[get_block_indices] blocks matched but empty ranges")
            return np.empty(0, dtype=np.int64)

        cand_idx = np.concatenate(segs)
        # Different blocks may have overlapping voxel indices at the boundaries, so deduplicate once
        cand_idx = np.unique(cand_idx)

        if debug:
            # Decode the (ix,iy,iz) of the first few blocks to see if they surround the center
            sample = hit_blocks[: min(5, hit_blocks.size)]
            k = self.block_keys[sample]
            ix = (k >> 40) & ((1 << 20) - 1)
            iy = (k >> 20) & ((1 << 20) - 1)
            iz = k & ((1 << 20) - 1)
            print(
                f"[get_block_indices] blocks_hit={int(hit_blocks.size)}, "
                f"coarse_pts={int(cand_idx.size)}, "
                f"sample_block_ids[0..]: ix={ix}, iy={iy}, iz={iz}"
            )

        if not refine:
            return cand_idx, hit_blocks

        # ------- Radius refinement -------
        pts_sub = self.pts[cand_idx]
        d2 = np.sum((pts_sub - origin) ** 2, axis=1)
        keep = d2 <= r * r
        idx_final = cand_idx[keep]

        if debug:
            n_fine = int(idx_final.size)
            if n_fine > 0:
                d = np.sqrt(d2[keep])
                print(
                    f"[get_block_indices] fine_pts={n_fine}, "
                    f"min/max_dist={float(d.min()):.4f}/{float(d.max()):.4f}, "
                    f"radius={float(r):.4f}"
                )
            else:
                print(f"[get_block_indices] fine_pts=0, radius={float(r):.4f}")

        return idx_final, hit_blocks

    # ======== Visualization functions ========
    def plot_level(self, level: DTYPE_FLOAT = 0.0, tol: DTYPE_FLOAT = 1e-5):
        """
        Plot the contour of the specified isosurface level.
        """
        if self.shape is None:
            raise ValueError("plot_level() requires shape information.")

        pts_cpu = self.pts
        sdf_cpu = self.sdf

        grid = pv.ImageData()
        grid.dimensions = np.array(self.shape)
        grid.origin = pts_cpu.min(axis=0)
        grid.spacing = (pts_cpu.max(axis=0) - pts_cpu.min(axis=0)) / (
            np.array(self.shape) - 2
        )
        grid.point_data["sdf"] = sdf_cpu.reshape(self.shape, order="F").ravel(order="F")

        contour = grid.contour([level - tol, level + tol], scalars="sdf")
        pl = pv.Plotter()
        pl.add_mesh(contour, color="lightgrey", opacity=0.5)
        pl.show()

    def plot_sdf_volume(self):
        """
        Plot the volume rendering of the SDF voxel field.
        """
        if self.shape is None:
            raise ValueError("plot_sdf_volume() requires shape information.")

        pts_cpu = self.pts
        sdf_cpu = self.sdf

        grid = pv.ImageData()
        grid.dimensions = np.array(self.shape)
        grid.origin = pts_cpu.min(axis=0)
        grid.spacing = (pts_cpu.max(axis=0) - pts_cpu.min(axis=0)) / (
            np.array(self.shape) - 2
        )
        grid.point_data["sdf"] = sdf_cpu.reshape(self.shape, order="F").ravel(order="F")

        pl = pv.Plotter()
        pl.add_volume(
            grid, scalars="sdf", cmap="coolwarm", opacity="sigmoid_5", shade=True
        )
        pl.show()


def plot_multiple_sdfer(
    sdfers: list[Sdfer],
    legends: list[str] = None,
    level: DTYPE_FLOAT = 0.0,
    tol: DTYPE_FLOAT = 1e-5,
    colors=None,
    opacities=None,
    interactive_update: bool = False,
):
    """
    Simultaneously display multiple SDF isosurfaces.

    Parameters:
        sdfers: list[Sdfer]
            List of Sdfer instances to be plotted.
        legends: list[str] | None
            Legend labels for each object.
        level: DTYPE_FLOAT
            Isosurface level (default 0.0).
        colors: list[str] | None
            Colors for each object.
        opacities: list[DTYPE_FLOAT] | None
            Opacities for each object.
    """
    assert len(sdfers) > 0, "sdfers list cannot be empty."
    assert legends is None or len(legends) == len(
        sdfers
    ), "Length of legends must match sdfers."
    pl = pv.Plotter()

    if colors is None:
        colors = ["lightgrey", "lightcoral", "skyblue", "yellowgreen", "orange"]
    if opacities is None:
        opacities = [0.5] * len(sdfers)

    for i, sdf_obj in enumerate(sdfers):
        if sdf_obj.shape is None:
            raise ValueError(
                "plot_multiple_sdfer() requires shape information for each object."
            )

        sdf_cpu = sdf_obj.sdf

        grid = pv.ImageData()
        grid.dimensions = np.array(sdf_obj.shape)
        grid.origin = np.asarray(sdf_obj.bmin)
        grid.spacing = (np.asarray(sdf_obj.bmax) - grid.origin) / (
            np.array(sdf_obj.shape) - 2
        )
        grid.point_data["sdf"] = sdf_cpu.reshape(sdf_obj.shape, order="F").ravel(
            order="F"
        )

        contour = grid.contour([level - tol, level + tol], scalars="sdf")
        color = colors[i % len(colors)]
        opacity = opacities[i % len(opacities)]
        pl.add_mesh(
            contour,
            color=color,
            opacity=opacity,
            label=legends[i] if legends else f"SDF {i + 1}",
        )

    pl.add_legend()
    pl.show(interactive_update=interactive_update)


# ------------------------------------------------------------
# Self-test
# ------------------------------------------------------------
if __name__ == "__main__":
    sphere_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=5.0)
    box_mesh = o3d.geometry.TriangleMesh.create_box(width=6, height=3, depth=3)
    sdf1 = Sdfer.from_mesh(sphere_mesh, pitch=0.25)
    sdf2 = Sdfer.from_mesh(box_mesh, pitch=0.25)

    plot_multiple_sdfer(
        [sdf1, sdf2], level=0.0, colors=["red", "blue"], opacities=[0.4, 0.3]
    )

    # Block index self-test
    test_box = o3d.geometry.TriangleMesh.create_box(width=4, height=4, depth=4)
    sdf_test = Sdfer.from_mesh(test_box, pitch=0.5, block_size=0.5 * 2)

    center = np.mean(sdf_test.pts, axis=0)
    radius = 1
    idx_test, hit_blocks = sdf_test.get_block_indices(
        center, radius, refine=False, debug=True
    )
    hit_keys = sdf_test.block_keys[hit_blocks]

    print(
        f"[BlockIndex Test] total={len(sdf_test.pts)}, blocks={len(sdf_test.block_keys)}, candidates={len(idx_test)}"
    )

    # ---- Prepare visualization data ----
    pts_all = sdf_test.pts
    pts_sub = sdf_test.pts[idx_test]
    center_np = center
    bmin_np = sdf_test.bmin
    bsize = float(sdf_test.block_size)
    k = sdf_test.block_keys
    ix = (k >> 40) & ((1 << 20) - 1)
    iy = (k >> 20) & ((1 << 20) - 1)
    iz = k & ((1 << 20) - 1)

    # ---- Construct block cubes ----
    cubes_hit = []
    cubes_rest = []

    k = sdf_test.block_keys
    ix = (k >> 40) & ((1 << 20) - 1)
    iy = (k >> 20) & ((1 << 20) - 1)
    iz = k & ((1 << 20) - 1)

    bmin_np = sdf_test.bmin
    bsize = float(sdf_test.block_size)

    for i in range(len(ix)):
        c = bmin_np + np.array([ix[i], iy[i], iz[i]], dtype=float) * bsize + bsize / 2
        cube = pv.Cube(center=c, x_length=bsize, y_length=bsize, z_length=bsize)
        # If the block is hit, add to cubes_hit; otherwise, add to cubes_rest
        if k[i] in hit_keys:
            cubes_hit.append(cube)
        else:
            cubes_rest.append(cube)

    # ---- Visualization ----
    pl = pv.Plotter()
    pl.add_mesh(
        pv.MultiBlock(cubes_rest), color="white", style="wireframe", opacity=0.3
    )
    pl.add_mesh(pv.MultiBlock(cubes_hit), color="red", style="wireframe", line_width=3)
    pl.add_points(pts_all, color="gray", point_size=4)
    pl.add_points(pts_sub, color="red", point_size=10)
    pl.add_mesh(pv.Sphere(radius=0.02, center=center_np), color="yellow")
    pl.add_mesh(pv.Sphere(radius=radius, center=center_np), color="cyan", opacity=0.2)
    pl.show()
