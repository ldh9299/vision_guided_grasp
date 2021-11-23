import trimesh


mesh = trimesh.load('./svh_hand_new/d12.stl')
# mesh.show()
# exit()
plane_normal = (-1, 0, 0)
plane_origin = (0.06, 0, 0.0)
new_mesh = trimesh.intersections.slice_mesh_plane(mesh, plane_normal, plane_origin)
new_mesh.show()
# new_mesh = trimesh.creation.box(extents=(0.0005, 0.0005, 0.0005))
# new_mesh.show()
# new_mesh.export('./svh_hand_new/d12.stl')
