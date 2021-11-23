import torch
from kaolin.metrics import trianglemesh
import numpy as np
device = 'cuda:0'
dtype = torch.float32
import trimesh

obj_vertices = np.array([[0, 0, 0],
                          [0, 1, 0],
                          [0, 0, 1]])
obj_faces = np.array([[0, 1, 2]])
mesh = trimesh.Trimesh(obj_vertices, obj_faces)
mesh.show()
vertices = torch.tensor([[[0, 0, 0],
                          [0, 1, 0],
                          [0, 0, -1]],

                         [[0, 0, 0],
                          [0, 1, 1],
                          [1, 0, 1]]], device=device, dtype=dtype)

faces = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)


point1 = torch.from_numpy(np.load('../dataset/object_vertices/D_1.npy').astype(np.float32)).to(device).repeat(2, 1, 1)
# point1 = torch.tensor([[[2, 0.5, 0]],
#                        [[0.5, 0.5, 0.5]]], device=device, dtype=dtype)

dist, idx, dist_type = trianglemesh.point_to_mesh_distance(point1, vertices, faces)
print(vertices.shape)
print(faces.shape)
print(point1.shape)
print(dist[0])

pc = point1.detach().cpu().numpy()[0]
pc = trimesh.PointCloud(pc, colors=[0, 255, 0])
scene = trimesh.Scene([pc, mesh])
scene.show()