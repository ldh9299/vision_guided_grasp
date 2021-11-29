import shutil
import os
import trimesh

# src_dir = 'dataset/contactpose_data'
# dst_dir = 'dataset/object_models'
# for root, dirs, files in os.walk(src_dir):
#     for filename in files:
#         if filename.endswith('.ply'):
#             filepath = os.path.join(root, filename)
#             shutil.copy(filepath, dst_dir)

dst_dir = 'dataset/object_models'
for root, dirs, files in os.walk(dst_dir):
    for filename in files:
        if filename.endswith('.ply'):
            filepath = os.path.join(root, filename)
            mesh = trimesh.load_mesh(filepath)
            mesh.show()
            exit()

