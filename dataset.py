import numpy as np
import open3d.open3d.io
import trimesh
import torch
from tqdm import tqdm
import pytorch3d.transforms


class Dataset:
    def __init__(self):
        self.data = np.load('object_in_use_all.npy', allow_pickle=True)[13:14]
        # print(self.data[0].keys())

    def __getitem__(self, index):
        # index = 6
        # print(self.data[index]['human_id'], self.data[index]['object_name'])
        contactmap_filename = '/home/ldh/ContactPose/data/contactpose_data/full{}_use/{}/{}.ply'.format(
            self.data[index]['human_id'], self.data[index]['object_name'], self.data[index]['object_name'])
        mesh = open3d.open3d.io.read_triangle_mesh(contactmap_filename)
        mesh.compute_vertex_normals()
        idx = np.load('dataset/idx_3000/{}.npy'.format(self.data[index]['object_name']))

        points = torch.tensor(np.asarray(mesh.vertices)[idx])
        normals = torch.tensor(np.asarray(mesh.vertex_normals)[idx])
        contactmap = torch.tensor(np.asarray(mesh.vertex_colors)[:, 0][idx])
        J = torch.tensor(self.data[index]['J'])
        root_mat = torch.tensor(self.data[index]['root_mat'][:3, :3])
        line_idx = self.data[index]['line_idx'][idx]
        finger_idx = line_idx // 4
        part_idx = line_idx % 4
        data_ = dict(points=points, normals=normals, contactmap=contactmap, J=J, root_mat=root_mat, line_idx=line_idx,
                     finger_idx=finger_idx, part_idx=part_idx, index=index)
        return data_

    def __len__(self):
        return len(self.data)
        # return 1


if __name__ == '__main__':
    # data_id = 0
    # dataset = Dataset()
    # print(dataset.data[data_id].keys())
    # print(len(dataset))
    # print(dataset.data[data_id]['object_name'])
    #
    # data = np.load('object_in_use_all.npy', allow_pickle=True)
    # index = 1
    # contactmap_filename = '/home/ldh/ContactPose/data/contactpose_data/full{}_use/{}/{}.ply'.format(
    #     data[index]['human_id'], data[index]['object_name'], data[index]['object_name'])
    # mesh = open3d.open3d.io.read_triangle_mesh(contactmap_filename)
    # mesh_ = trimesh.load_mesh(contactmap_filename)
    # geoms = []
    # geoms.append(mesh)
    # # open3d.open3d.visualization.draw_geometries(geoms)
    # # exit()
    # print(np.max(data[index]['line_idx']), np.min(data[index]['line_idx']))
    #
    # id_0 = (data[index]['line_idx'] >= 16) & (data[index]['line_idx'] <= 19) & (np.asarray(mesh.vertex_colors)[:, 0] > 0.4)
    # pc = trimesh.PointCloud(np.asarray(mesh.vertices)[id_0], colors=[255, 0, 0])
    # pc_ = trimesh.PointCloud(np.asarray(mesh_.vertices), colors=[0, 255, 0])
    # scene = trimesh.Scene()
    # scene.add_geometry([pc_, pc])
    # scene.show()
    # # print(mesh.vertices.shape)
    # exit()
    # # print(mesh.vertex_normals.shape)
    # # print(mesh.visual.vertex_colors.shape)
    # # exit()
    # idx = np.load('dataset/idx_3000/{}.npy'.format(dataset.data[data_id]['object_name']))
    # points = mesh.vertices[idx]
    # normals = mesh.vertex_normals[idx]
    # contactmap = mesh.visual.vertex_colors[idx]
    # pc_sub = trimesh.PointCloud(points, colors=contactmap)
    # pc_sub.show()

    dataset = Dataset()
    print(len(dataset))
    exit()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,
                                             num_workers=1)
    for data in dataloader:
        id_0 = (data['line_idx'] >= 0) & (data['line_idx'] <= 19) & (
                    data['contactmap'] > 0.4)
        pc = open3d.geometry.PointCloud()
        pc.points = open3d.utility.Vector3dVector(data['points'].squeeze())
        pc.normals = open3d.utility.Vector3dVector(data['normals'].squeeze())
        open3d.visualization.draw_geometries([pc])
        pass
