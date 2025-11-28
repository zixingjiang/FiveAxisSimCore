import numpy as np
import pyvista as pv
import trimesh
import open3d as o3d


# ------------------------------------------------------------
#  Basic conversion functions
# ------------------------------------------------------------


def pv_to_trimesh(mesh: pv.PolyData) -> trimesh.Trimesh:
    faces_arr = np.asarray(mesh.faces)
    verts = np.asarray(mesh.points)
    faces = []
    i = 0
    while i < len(faces_arr):
        nv = faces_arr[i]
        faces.append(faces_arr[i + 1 : i + 1 + nv])
        i += nv + 1
    tri_faces = [f for f in faces if len(f) == 3]
    if not tri_faces:
        return pv_to_trimesh(mesh.triangulate())
    return trimesh.Trimesh(vertices=verts, faces=np.array(tri_faces), process=False)


def trimesh_to_pv(mesh: trimesh.Trimesh) -> pv.PolyData:
    faces = np.hstack(np.column_stack((np.full(len(mesh.faces), 3), mesh.faces)))
    return pv.PolyData(mesh.vertices, faces)


def trimesh_to_o3d(mesh: trimesh.Trimesh) -> o3d.geometry.TriangleMesh:
    o3 = o3d.geometry.TriangleMesh()
    o3.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    o3.triangles = o3d.utility.Vector3iVector(mesh.faces)
    return o3


def o3d_to_trimesh(mesh: o3d.geometry.TriangleMesh) -> trimesh.Trimesh:
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def pv_to_o3d(mesh: pv.PolyData) -> o3d.geometry.TriangleMesh:
    faces_arr = np.asarray(mesh.faces)
    verts = np.asarray(mesh.points)
    faces = []
    i = 0
    while i < len(faces_arr):
        nv = faces_arr[i]
        faces.append(faces_arr[i + 1 : i + 1 + nv])
        i += nv + 1
    tri_faces = [f for f in faces if len(f) == 3]
    if not tri_faces:
        return pv_to_o3d(mesh.triangulate())
    o3 = o3d.geometry.TriangleMesh()
    o3.vertices = o3d.utility.Vector3dVector(verts)
    o3.triangles = o3d.utility.Vector3iVector(np.array(tri_faces))
    return o3


def o3d_to_pv(mesh: o3d.geometry.TriangleMesh) -> pv.PolyData:
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles, dtype=np.int64)
    n_faces = faces.shape[0]
    face_sizes = np.full((n_faces, 1), 3, dtype=np.int64)
    faces_with_size = np.hstack((face_sizes, faces))
    faces_flat = faces_with_size.flatten()
    return pv.PolyData(vertices, faces_flat)


# ------------------------------------------------------------
#  Auto-detect conversion interface
# ------------------------------------------------------------


def ensure_trimesh(obj):
    if isinstance(obj, trimesh.Trimesh):
        return obj
    if isinstance(obj, pv.PolyData):
        return pv_to_trimesh(obj)
    if isinstance(obj, o3d.geometry.TriangleMesh):
        return o3d_to_trimesh(obj)
    raise TypeError(f"Unsupported mesh type: {type(obj)}")


def ensure_pv(obj):
    if isinstance(obj, pv.PolyData):
        return obj
    if isinstance(obj, trimesh.Trimesh):
        return trimesh_to_pv(obj)
    if isinstance(obj, o3d.geometry.TriangleMesh):
        return o3d_to_pv(obj)
    raise TypeError(f"Unsupported mesh type: {type(obj)}")


def ensure_o3d(obj):
    if isinstance(obj, o3d.geometry.TriangleMesh):
        return obj
    if isinstance(obj, trimesh.Trimesh):
        return trimesh_to_o3d(obj)
    if isinstance(obj, pv.PolyData):
        return pv_to_o3d(obj)
    raise TypeError(f"Unsupported mesh type: {type(obj)}")


# -----------------------------------------------------
#  General entry point
# -----------------------------------------------------

_TYPE_MAP = {
    "pv": ensure_pv,
    "pyvista": ensure_pv,
    "trimesh": ensure_trimesh,
    "tm": ensure_trimesh,
    "o3d": ensure_o3d,
    "open3d": ensure_o3d,
}


def as_mesh_type(obj, target_type: str):
    """
    通用网格类型转换。
    target_type: 'pv' | 'trimesh' | 'o3d'
    """
    key = target_type.lower()
    if key not in _TYPE_MAP:
        raise ValueError(f"Unknown target type: {target_type}")
    return _TYPE_MAP[key](obj)


# -----------------------------------------------------
#  Self-test
# -----------------------------------------------------
if __name__ == "__main__":
    shapes = {
        "Cube": pv.Cube(center=(0, 0, 0), x_length=10, y_length=10, z_length=10),
        "Sphere": pv.Sphere(radius=5, theta_resolution=32, phi_resolution=32),
        "Cylinder": pv.Cylinder(
            center=(0, 0, 0), direction=(0, 0, 1), radius=3, height=8
        ),
        "Cone": pv.Cone(center=(0, 0, 0), direction=(0, 0, 1), height=10, radius=4),
        "Ellipsoid": pv.ParametricEllipsoid(5, 3, 2),
        "Torus": pv.ParametricTorus(ringradius=6, crosssectionradius=2),
    }

    for name, mesh in shapes.items():
        print(f"Testing {name}")
        tmesh = pv_to_trimesh(mesh)
        o3 = trimesh_to_o3d(tmesh)
        mesh_back = o3d_to_pv(o3)

        pl = pv.Plotter(title=f"{name} conversion test")
        pl.add_mesh(mesh_back, color="lightgray")
        pl.show()

    print("PyVista -> Trimesh -> Open3D -> PyVista OK")

    for name, mesh in shapes.items():
        print(f"Testing {name}")
        o3 = pv_to_o3d(mesh)
        tmesh = o3d_to_trimesh(o3)
        mesh_back = trimesh_to_pv(tmesh)

        pl = pv.Plotter(title=f"{name} conversion test")
        pl.add_mesh(mesh_back, color="lightgray")
        pl.show()

    print("PyVista -> Open3D -> Trimesh -> PyVista OK")
