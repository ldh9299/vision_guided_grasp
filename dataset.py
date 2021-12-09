import numpy as np
import open3d.open3d.io
import trimesh
import torch
import torch.utils.data as data
from tqdm import tqdm
import os


def read_object_models(dst_dir):
    object_models_dict = {}
    points_idx_dict = {}
    # idx = np.load('dataset/idx_3000/{}.npy'.format(self.data[index]['object_name']))
    for root, dirs, files in os.walk(dst_dir):
        for filename in files:
            if filename.endswith('.ply'):
                object_filepath = os.path.join(root, filename)
                mesh = open3d.open3d.io.read_triangle_mesh(object_filepath)
                mesh.compute_vertex_normals()
                obj_name = filename[:-4]
                object_models_dict[obj_name] = mesh
                points_idx_dict[obj_name] = np.load('dataset/idx_3000/{}.npy'.format(obj_name))
    return object_models_dict, points_idx_dict


class Contact_dataset(data.Dataset):
    def __init__(self, debug_vis=False):
        super(Contact_dataset).__init__()
        self.debug_vis = debug_vis
        self.data = np.load('dataset/object_in_use_all.npy', allow_pickle=True)[0:64]
        dst_dir = 'dataset/object_models'
        self.object_models_dict, self.points_idx_dict = read_object_models(dst_dir)

    def __getitem__(self, index):
        # index = 6

        # object_filename = 'dataset/object_models/{}.ply'.format(self.data[index]['object_name'])
        # mesh = open3d.open3d.io.read_triangle_mesh(object_filename)
        # mesh.compute_vertex_normals()
        object_name = self.data[index]['object_name']
        mesh = self.object_models_dict[object_name]

        points_idx = self.points_idx_dict[object_name]

        points = torch.tensor(np.asarray(mesh.vertices)[points_idx])

        normals = torch.tensor(np.asarray(mesh.vertex_normals)[points_idx])

        contactmap = torch.tensor(self.data[index]['contact_map'][points_idx])
        J = torch.tensor(self.data[index]['J'])
        root_mat = torch.tensor(self.data[index]['root_mat'][:3, :3])
        line_idx = self.data[index]['line_idx'][points_idx]
        finger_idx = line_idx // 4
        part_idx = line_idx % 4
        vis = self.debug_vis
        if vis:
            pc = trimesh.PointCloud(points, colors=[255, 255, 0])
            mask = contactmap < 0.4
            pc_mask = trimesh.PointCloud(points[mask], colors=[0, 255, 255])
            scene = trimesh.Scene([pc, pc_mask])
            scene.show()

        data_dict = dict(points=points, normals=normals, contactmap=contactmap, J=J, root_mat=root_mat, line_idx=line_idx,
                         finger_idx=finger_idx, part_idx=part_idx, index=index)
        return data_dict

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    dataset = Contact_dataset(debug_vis=True)
    print(len(dataset))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,
                                             num_workers=1)
    for data in dataloader:
        pass
