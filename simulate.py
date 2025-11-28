import numpy as np
import pyvista as pv
import open3d as o3d
from cutter import BaseCutter, TapperCutter, CylinderCutter, BallCutter
from utils.logger import Logger
from utils.gcode_reader import GCodeReader
from sdfer import Sdfer, plot_multiple_sdfer
from tqdm import tqdm
from utils.backend import xp, GPU_ENABLED, as_xp_array, DTYPE_FLOAT


PITCH = 0.1


class Simulator:
    """Five-axis cutting simulator."""

    def __init__(
        self,
        cutter: BaseCutter,
        workpiece_mesh: o3d.geometry.TriangleMesh,
        pitch: DTYPE_FLOAT = 0.2,
        block_size: DTYPE_FLOAT = 1.0,
    ):
        self.cutter = cutter
        self.sdfer_workpiece = Sdfer.from_mesh(
            workpiece_mesh, pitch=pitch, block_size=block_size
        )
        self.pitch = pitch
        self.logger = Logger("Simulator")

    # ------------------------------------------------------------
    # Trajectory-aware AABB global clipping, only keep voxels that may be cut
    # ------------------------------------------------------------
    def _prefilter_workpiece(self, trajectory: np.ndarray):
        """Pre-filter workpiece voxels based on trajectory direction-aware bounding box"""
        cutter = self.cutter
        pts = as_xp_array(self.sdfer_workpiece.pts)

        # 1. Trajectory positions and directions
        pos = as_xp_array(trajectory[:, :3])
        dirs = as_xp_array(trajectory[:, 3:6])
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

        # 2. Direction-aware bounding box
        Rmax = max(
            getattr(cutter, "radius", 0.0),
            getattr(cutter, "radius_base", 0.0),
            getattr(cutter, "radius_tip", 0.0),
        )
        p_min = pos - dirs * cutter.length - Rmax
        p_max = pos + dirs * cutter.length + Rmax
        traj_min = np.min(p_min, axis=0)
        traj_max = np.max(p_max, axis=0)
        margin = 2 * self.pitch

        # 3. Global mask filtering
        mask_global = (
            (pts[:, 0] >= traj_min[0] - margin)
            & (pts[:, 0] <= traj_max[0] + margin)
            & (pts[:, 1] >= traj_min[1] - margin)
            & (pts[:, 1] <= traj_max[1] + margin)
            & (pts[:, 2] >= traj_min[2] - margin)
            & (pts[:, 2] <= traj_max[2] + margin)
        )

        # Directly set active mask inside Sdfer
        self.sdfer_workpiece.active_mask = mask_global  # AABB Prefiltering
        self.sdfer_workpiece.deactivate_outer_volume()  # Deactivate outer voxels

    # ------------------------------------------------------------
    # Main simulation loop
    # ------------------------------------------------------------
    def run(self, trajectory: np.ndarray, vis_interval: int = 500):
        """
        Execute cutting step-by-step according to the trajectory.
        Parameters
        ----
        trajectory : ndarray [N,6]
            [X,Y,Z,I,J,K]
        vis_interval : int
            How many steps between each visualization update
        """
        # ---- Step 0: Global prefilter ----
        self._prefilter_workpiece(trajectory)
        wp_cut = (
            self.sdfer_workpiece
        )  # alias, changes will be in-place to self.sdfer_workpiece
        cutter = self.cutter

        # ---- Step 1: Main cutting loop ----
        chip_voxel_list = []
        with tqdm(total=len(trajectory), ncols=100, desc="Simulating") as pbar:
            for i, row in enumerate(trajectory):
                x, y, z, i_, j_, k_ = row
                origin = np.array([x, y, z])
                direction = np.array([i_, j_, k_])

                num_chip_voxels = cutter.cut_inplace(
                    wp_cut, origin, direction, margin=self.pitch * 4
                )
                chip_voxel_list.append(num_chip_voxels)

                if (i + 1) % vis_interval == 0:
                    try:
                        cutter_sdf = cutter.get_sdfer(
                            as_xp_array(wp_cut.pts),
                            origin,
                            direction,
                            shape=wp_cut.shape,
                        )
                        plot_multiple_sdfer([wp_cut, cutter_sdf], opacities=[0.9])
                    except Exception as e:
                        print(f"[warn] Visualization skipped: {e}")

                pbar.update(1)

        return wp_cut, chip_voxel_list

    def visualize_chip_volume_3d(
        self, trajectory: np.ndarray, chip_voxel_list: list[int], pitch: float = 0.1
    ):
        """
        Visualize the chip volume at each trajectory point using PyVista.
        trajectory: (N, 3)
        chip_voxel_list: (N,) int
        """
        trajectory = np.asarray(trajectory)
        chip_voxel_list = np.asarray(chip_voxel_list)
        chip_volume_list = chip_voxel_list * (pitch**3)

        # Construct point cloud
        cloud = pv.PolyData(trajectory)
        cloud["chip_volume_list [mm^3]"] = chip_volume_list
        pl = pv.Plotter()
        pl.add_mesh(
            cloud,
            scalars="chip_volume_list [mm^3]",
            point_size=8,
            render_points_as_spheres=True,
            cmap="jet",
            opacity=0.9,
        )
        pl.show()

    def run_with_animation(
        self,
        trajectory: np.ndarray,
        output_path: str = "simulation.gif",
        frame_interval: int = 1,
        fps: int = 10,
        show_cutter: bool = True,
        show_trajectory: bool = True,
        workpiece_color: str = "lightblue",
        cutter_color: str = "red",
        window_size: tuple = (1024, 768),
    ):
        """
        Execute cutting simulation with animation output.

        Parameters
        ----------
        trajectory : ndarray [N,6]
            [X,Y,Z,I,J,K] - position and direction for each step
        output_path : str
            Path to save the animation (supports .gif, .mp4)
        frame_interval : int
            Record a frame every N steps
        fps : int
            Frames per second for the output animation
        show_cutter : bool
            Whether to show the cutter in the animation
        show_trajectory : bool
            Whether to show the trajectory path
        workpiece_color : str
            Color of the workpiece mesh
        cutter_color : str
            Color of the cutter
        window_size : tuple
            Size of the rendering window (width, height)

        Returns
        -------
        wp_cut : Sdfer
            The cut workpiece SDF
        chip_voxel_list : list[int]
            Number of voxels removed at each step
        """
        # ---- Step 0: Global prefilter ----
        self._prefilter_workpiece(trajectory)
        wp_cut = self.sdfer_workpiece
        cutter = self.cutter

        # Get cutter geometry parameters
        Rmax = max(
            getattr(cutter, "radius", 0.0),
            getattr(cutter, "radius_base", 0.0),
            getattr(cutter, "radius_tip", 0.0),
        )

        # Setup plotter for off-screen rendering
        pl = pv.Plotter(off_screen=True, window_size=window_size)
        pl.set_background("white")

        # Add trajectory line if requested
        if show_trajectory:
            traj_points = trajectory[:, :3]
            traj_line = pv.Spline(traj_points, len(traj_points) * 2)
            pl.add_mesh(
                traj_line, color="gray", line_width=2, opacity=0.5, name="trajectory"
            )

        # Initial workpiece mesh
        workpiece_mesh = self._sdfer_to_mesh(wp_cut)
        if workpiece_mesh is not None:
            pl.add_mesh(
                workpiece_mesh, color=workpiece_color, opacity=0.9, name="workpiece"
            )

        # Set up camera
        pl.camera_position = "iso"
        pl.camera.zoom(0.8)

        # Open GIF writer
        pl.open_gif(output_path, fps=fps)

        # ---- Main cutting loop with animation ----
        chip_voxel_list = []

        with tqdm(
            total=len(trajectory), ncols=100, desc="Simulating with animation"
        ) as pbar:
            for i, row in enumerate(trajectory):
                x, y, z, i_, j_, k_ = row
                origin = np.array([x, y, z])
                direction = np.array([i_, j_, k_])
                direction_norm = direction / np.linalg.norm(direction)

                # Perform cutting
                num_chip_voxels = cutter.cut_inplace(
                    wp_cut, origin, direction, margin=self.pitch * 4
                )
                chip_voxel_list.append(num_chip_voxels)

                # Record frame at specified intervals
                if i % frame_interval == 0 or i == len(trajectory) - 1:
                    # Update workpiece mesh
                    pl.remove_actor("workpiece")
                    workpiece_mesh = self._sdfer_to_mesh(wp_cut)
                    if workpiece_mesh is not None:
                        pl.add_mesh(
                            workpiece_mesh,
                            color=workpiece_color,
                            opacity=0.9,
                            name="workpiece",
                        )

                    # Update cutter visualization
                    if show_cutter:
                        pl.remove_actor("cutter")
                        cutter_mesh = self._create_cutter_mesh(
                            origin, direction_norm, Rmax, cutter.length
                        )
                        pl.add_mesh(
                            cutter_mesh, color=cutter_color, opacity=0.7, name="cutter"
                        )

                        # Add cutter position marker
                        pl.remove_actor("cutter_tip")
                        tip_sphere = pv.Sphere(radius=Rmax * 0.3, center=origin)
                        pl.add_mesh(tip_sphere, color="yellow", name="cutter_tip")

                    # Add progress indicator
                    pl.remove_actor("progress_text")
                    progress_text = f"Step: {i+1}/{len(trajectory)}"
                    pl.add_text(
                        progress_text,
                        position="upper_left",
                        font_size=12,
                        name="progress_text",
                    )

                    # Write frame
                    pl.write_frame()

                pbar.update(1)

        # Close and save
        pl.close()
        self.logger.info(f"Animation saved to {output_path}")

        return wp_cut, chip_voxel_list

    def _sdfer_to_mesh(self, sdfer: Sdfer, level: float = 0.0) -> pv.PolyData | None:
        """Convert SDF to PyVista mesh using marching cubes."""
        if sdfer.shape is None:
            return None

        try:
            # Create structured grid from SDF
            grid = pv.ImageData()
            grid.dimensions = np.array(sdfer.shape)
            grid.origin = sdfer.pts.min(axis=0)
            grid.spacing = (sdfer.pts.max(axis=0) - sdfer.pts.min(axis=0)) / (
                np.array(sdfer.shape) - 2
            )
            grid.point_data["sdf"] = sdfer.sdf.reshape(sdfer.shape, order="F").ravel(
                order="F"
            )

            # Extract isosurface
            contour = grid.contour([level], scalars="sdf")
            if contour.n_points > 0:
                return contour
        except Exception as e:
            self.logger.warn(f"Failed to create mesh: {e}")

        return None

    def _create_cutter_mesh(
        self, origin: np.ndarray, direction: np.ndarray, radius: float, length: float
    ) -> pv.PolyData:
        """Create a mesh representation of the cutter."""
        # Create a cylinder aligned with Z-axis
        cylinder = pv.Cylinder(
            center=(0, 0, length / 2),
            direction=(0, 0, 1),
            radius=radius,
            height=length,
            resolution=32,
            capping=True,
        )

        # Compute rotation to align with direction
        z_axis = np.array([0, 0, 1])
        direction = direction / np.linalg.norm(direction)

        if np.allclose(direction, z_axis):
            rotation_matrix = np.eye(3)
        elif np.allclose(direction, -z_axis):
            rotation_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        else:
            # Rodrigues' rotation formula
            v = np.cross(z_axis, direction)
            s = np.linalg.norm(v)
            c = np.dot(z_axis, direction)
            vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            rotation_matrix = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s + 1e-10))

        # Apply rotation and translation
        transform = np.eye(4)
        transform[:3, :3] = rotation_matrix
        transform[:3, 3] = origin

        cylinder = cylinder.transform(transform, inplace=False)
        return cylinder

    def create_comparison_animation(
        self,
        original_sdfer: Sdfer,
        cut_sdfer: Sdfer,
        trajectory: np.ndarray,
        output_path: str = "comparison.gif",
        fps: int = 5,
        n_frames: int = 36,
        window_size: tuple = (1200, 600),
    ):
        """
        Create a side-by-side comparison animation showing original vs cut workpiece.

        Parameters
        ----------
        original_sdfer : Sdfer
            Original workpiece SDF
        cut_sdfer : Sdfer
            Cut workpiece SDF
        trajectory : np.ndarray
            Trajectory array for showing cut path
        output_path : str
            Path to save the animation
        fps : int
            Frames per second
        n_frames : int
            Number of frames for full rotation
        window_size : tuple
            Window size (width, height)
        """
        pl = pv.Plotter(shape=(1, 2), off_screen=True, window_size=window_size)

        # Left: Original workpiece
        pl.subplot(0, 0)
        pl.add_text("Original", position="upper_edge", font_size=14)
        original_mesh = self._sdfer_to_mesh(original_sdfer)
        if original_mesh is not None:
            pl.add_mesh(original_mesh, color="lightgray", opacity=0.9)
        pl.set_background("white")

        # Right: Cut workpiece with trajectory
        pl.subplot(0, 1)
        pl.add_text("After Cutting", position="upper_edge", font_size=14)
        cut_mesh = self._sdfer_to_mesh(cut_sdfer)
        if cut_mesh is not None:
            pl.add_mesh(cut_mesh, color="lightblue", opacity=0.9)

        # Add trajectory
        traj_points = trajectory[:, :3]
        traj_line = pv.Spline(traj_points, len(traj_points) * 2)
        pl.add_mesh(traj_line, color="red", line_width=3, opacity=0.7)
        pl.set_background("white")

        # Link cameras
        pl.link_views()
        pl.camera_position = "iso"
        pl.camera.zoom(0.8)

        # Create rotation animation
        pl.open_gif(output_path, fps=fps)

        for i in range(n_frames):
            pl.camera.azimuth = i * (360 / n_frames)
            pl.write_frame()

        pl.close()
        self.logger.info(f"Comparison animation saved to {output_path}")


# ------------------------------------------------------------
# Self-test + line-by-line performance analysis
# ------------------------------------------------------------
if __name__ == "__main__":
    cutter = BallCutter(radius=0.5, length=8.0)
    mesh = o3d.geometry.TriangleMesh.create_box(width=10, height=10, depth=5)
    mesh.translate((-5, -5, -2.5))
    sdf_original = Sdfer.from_mesh(mesh, pitch=PITCH, block_size=PITCH * 10)

    traj_file = "assets/gcode/test.ncc"
    with open(traj_file, "r", encoding="utf-8") as fp:
        traj = GCodeReader.load(fp)
        traj = as_xp_array(traj)

    sim = Simulator(cutter, mesh, pitch=PITCH, block_size=PITCH * 10)

    # Run simulation with animation
    sdfer_cut, chip_voxel_list = sim.run_with_animation(
        traj,
        output_path="simulation.gif",
        frame_interval=1,  # Record every step
        fps=5,
        show_cutter=True,
        show_trajectory=True,
        workpiece_color="lightblue",
        cutter_color="red",
    )

    # Create comparison animation (rotating view of before/after)
    sim.create_comparison_animation(
        sdf_original,
        sdfer_cut,
        traj,
        output_path="comparison.gif",
        fps=10,
        n_frames=72,  # 2 full rotations at 10fps = ~7 seconds
    )

    # Also show interactive visualization
    plot_multiple_sdfer(
        [sdf_original, sdfer_cut], opacities=[0.1, 1], colors=["lightgrey", "lightblue"]
    )

    # Show chip volume distribution
    sim.visualize_chip_volume_3d(traj[:, :3], chip_voxel_list, pitch=PITCH)
