import trimesh
import trimesh.voxel.creation
import numpy as np
import subprocess
import shlex
import path
import open3d


mesh = trimesh.load('./svh_hand/righthand_base.stl')

vg = trimesh.voxel.creation.local_voxelize(mesh, point=[0, 0, 0], radius=32, pitch=0.003125)
vg.show()
exit()

# scene = trimesh.Scene([vg, mesh])
# mesh.show()

# mesh.show()

# binvoxer_kwargs = {'binvox_path': './binvox', 'dimension': 128}
#
# vg = trimesh.exchange.binvox.voxelize_mesh(mesh, binvoxer=None, export_type='off', **binvoxer_kwargs)
# print(dir(vg))
# vg.show()


def get_binvox_file(cad_file, solid=True):
    cad_file = path.Path(cad_file)
    vox_file = cad_file.with_suffix(".binvox")
    if vox_file.exists():
        raise IOError(f"Binvox file exists: {vox_file}")

    out_file = cad_file.with_suffix(".solid.binvox")
    if not out_file.exists():
        cmd = f"binvox -d 64 -aw -dc -pb {cad_file}"
        subprocess.check_output(shlex.split(cmd))
        vox_file.rename(out_file)
    return out_file


with open('D_1_full_smooth.binvox', 'rb') as f:
    vg = trimesh.exchange.binvox.load_binvox(f)
    print(vg.pitch)

    point = vg.points
    pc = trimesh.PointCloud(point, colors=[255, 0, 0])

    extents = mesh.bounding_box.extents
    bbox_diagonal = np.sqrt((extents ** 2).sum())
    print(bbox_diagonal)

    pitch = 1.0 * bbox_diagonal / 32

    pcd = open3d.PointCloud()
    pcd.points = open3d.Vector3dVector(point)
    pcd = open3d.voxel_down_sample(pcd, voxel_size=pitch)
    points = np.asarray(pcd.points)
    pc_1 = trimesh.PointCloud(points, colors=[0, 255, 0])
    # print(points.shape)
    # exit()
    sdf = mesh.nearest.signed_distance(points)
    pc.show()
    pc_1.show()

    # vg.show()
    # scene = trimesh.Scene()
    # scene.add_geometry(vg)
    # scene.show()


