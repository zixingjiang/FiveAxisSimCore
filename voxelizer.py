import numpy as np
import pyvista as pv
import open3d as o3d
import trimesh
import time
from utils.mesh_convert import as_mesh_type
from utils.logger import Logger


class Voxelizer:
    """
    Voxelizer(voxel_size: float)
    ----------------------------
    Voxelize any 3D mesh (Trimesh/PyVista/Open3D) into a 3D occupancy matrix.
    """

    def __init__(self, voxel_size: float):
        self.voxel_size = float(voxel_size)
        self.tmesh = None
        self.__logger = Logger(name="Voxelizer", logfile=None, use_color=True)
        self.__logger.info(f"Initialized voxel_size={self.voxel_size:.4f}")

    def voxelize(self, mesh: trimesh.Trimesh | pv.PolyData | o3d.geometry.TriangleMesh):
        self.__logger.info("Starting voxelization...")
        start = time.perf_counter()

        tmesh = as_mesh_type(mesh, "trimesh")
        voxel = tmesh.voxelized(self.voxel_size)

        occ = voxel.matrix
        bbox_min = tmesh.bounds[0]
        bbox_max = bbox_min + np.array(occ.shape) * self.voxel_size

        self.tmesh = tmesh
        elapsed = time.perf_counter() - start
        self.__logger.info(
            f"Voxelized: shape={occ.shape}, occupied={occ.sum()}, time={elapsed:.3f}s"
        )

        return occ, bbox_min, bbox_max

    def __to_uniform_grid(self, occ, bbox_min, bbox_max):
        dims = np.array(occ.shape)
        grid = pv.ImageData()
        grid.dimensions = dims + 1
        grid.origin = bbox_min
        grid.spacing = (bbox_max - bbox_min) / dims
        grid.cell_data["occ"] = occ.astype(float).flatten(order="F")
        return grid

    def visualize(self, occ, bbox_min, bbox_max, color="#88CCFF", edge=True):
        grid = self.__to_uniform_grid(occ, bbox_min, bbox_max)
        pl = pv.Plotter(title="Voxel Visualization")
        pl.add_mesh(as_mesh_type(self.tmesh, "pyvista"), color="lightgray", opacity=0.5)

        occupied = grid.threshold(value=0.5, scalars="occ")
        pl.add_mesh(occupied, color=color, opacity=1.0, show_edges=True)

        if edge:
            edges = grid.extract_all_edges()
            pl.add_mesh(edges, color="black", line_width=0.5)

        pl.show_axes()
        pl.show()


# ------------------------------------------------------------
# Self-test
# ------------------------------------------------------------
if __name__ == "__main__":
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
    vox = Voxelizer(0.1)
    occ, bmin, bmax = vox.voxelize(sphere)
    vox.visualize(occ, bmin, bmax, edge=False)
