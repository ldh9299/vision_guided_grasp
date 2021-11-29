# hithand layer for torch
import torch
from pytorch3d.transforms import rotation_conversions
import math
import trimesh
import glob
import os
import numpy as np
import copy
import trimesh.voxel.creation
import time
import chamfer_distance as chd


def point2point_signed(
        x,
        y,
        x_normals=None,
        y_normals=None,
):
    """
    signed distance between two pointclouds
    Args:
        x: FloatTensor of shape (N, P1, D) representing a batch of point clouds
            with P1 points in each batch element, batch size N and feature
            dimension D.
        y: FloatTensor of shape (N, P2, D) representing a batch of point clouds
            with P2 points in each batch element, batch size N and feature
            dimension D.
        x_normals: Optional FloatTensor of shape (N, P1, D).
        y_normals: Optional FloatTensor of shape (N, P2, D).
    Returns:
        - y2x_signed: Torch.Tensor
            the sign distance from y to x
        - y2x_signed: Torch.Tensor
            the sign distance from y to x
        - yidx_near: Torch.tensor
            the indices of x vertices closest to y
    """

    N, P1, D = x.shape
    P2 = y.shape[1]

    if y.shape[0] != N or y.shape[2] != D:
        raise ValueError("y does not have the correct shape.")

    ch_dist = chd.ChamferDistance()

    x_near, y_near, xidx_near, yidx_near = ch_dist(x, y)

    xidx_near_expanded = xidx_near.view(N, P1, 1).expand(N, P1, D).to(torch.long)
    x_near = y.gather(1, xidx_near_expanded)

    yidx_near_expanded = yidx_near.view(N, P2, 1).expand(N, P2, D).to(torch.long)
    y_near = x.gather(1, yidx_near_expanded)

    x2y = x - x_near
    y2x = y - y_near

    if x_normals is not None:
        y_nn = x_normals.gather(1, yidx_near_expanded)
        in_out = torch.bmm(y_nn.view(-1, 1, 3), y2x.view(-1, 3, 1)).view(N, -1).sign()
        # in_out_expand = in_out.expand(N, P2, 3)
        y2x_signed = y2x.norm(dim=2) * in_out
        # print(y2x_signed.shape)

    else:
        y2x_signed = y2x.norm(dim=2)

    if y_normals is not None:
        x_nn = y_normals.gather(1, xidx_near_expanded)
        in_out_x = torch.bmm(x_nn.view(-1, 1, 3), x2y.view(-1, 3, 1)).view(N, -1).sign()
        x2y_signed = x2y.norm(dim=2) * in_out_x
    else:
        x2y_signed = x2y.norm(dim=2)

    return y2x_signed, x2y_signed, yidx_near, xidx_near


# All lengths are in mm and rotations in radians

CUR_DIR = os.path.dirname(os.path.abspath(__file__))


def timeCalc(func):
    def run(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print("函数{0}的运行时间为： {1}".format(func.__name__, time.time() - start))
        return result

    return run


def fps(points, npoint):
    """
    Input:
        mesh: input mesh
        graph: graph for mesh
        npoint: target point number to sample
    Return:
        centroids: sampled pointcloud index, [npoint]
    """

    N, C = points.shape
    centroids = np.zeros(npoint, dtype=int)
    distance = np.ones(N) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest

        centroid = points[farthest, :].reshape(1, 3)

        dist = np.sum((points - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]

        farthest = np.argmax(distance, -1)
    return centroids


def save_to_mesh(vertices, faces, output_mesh_path=None):
    assert output_mesh_path is not None
    with open(output_mesh_path, 'w') as fp:
        for vert in vertices:
            fp.write('v %f %f %f\n' % (vert[0], vert[1], vert[2]))
        for face in faces + 1:
            fp.write('f %d %d %d\n' % (face[0], face[1], face[2]))
    print('Output mesh save to: ', os.path.abspath(output_mesh_path))


class SvhHandLayer(torch.nn.Module):
    # @timeCalc
    def __init__(self, device='cuda', show_mesh=False):
        # The forward kinematics equations implemented here are from
        super().__init__()
        self.device = device
        self.show_mesh = show_mesh
        zero_tensor = torch.tensor(0.0, dtype=torch.float32, device=device)
        one_tensor = torch.tensor(1.0, dtype=torch.float32, device=device)
        self.J1_a = zero_tensor
        self.J1_alpha = zero_tensor
        self.J1_d = zero_tensor
        self.J1_theta = zero_tensor

        self.J2_a = one_tensor * -0.04596
        self.J2_alpha = zero_tensor
        self.J2_d = zero_tensor
        self.J2_theta = np.pi * one_tensor

        self.J3_a = one_tensor * 0.0485
        self.J3_alpha = zero_tensor
        self.J3_d = zero_tensor
        self.J3_theta = zero_tensor

        self.J4_a = one_tensor * 0.03
        self.J4_alpha = zero_tensor
        self.J4_d = zero_tensor
        self.J4_theta = zero_tensor

        self.J5_a = zero_tensor
        self.J5_alpha = zero_tensor
        self.J5_d = zero_tensor
        self.J5_theta = zero_tensor

        self.J6a_a = zero_tensor
        self.J6a_alpha = zero_tensor
        self.J6a_d = zero_tensor
        self.J6a_theta = zero_tensor

        self.J6b_a = zero_tensor
        self.J6b_alpha = one_tensor * -1.571
        self.J6b_d = zero_tensor
        self.J6b_theta = zero_tensor

        self.J7a_a = zero_tensor
        self.J7a_alpha = zero_tensor
        self.J7a_d = zero_tensor
        self.J7a_theta = zero_tensor

        self.J7b_a = zero_tensor
        self.J7b_alpha = one_tensor * -1.571
        self.J7b_d = zero_tensor
        self.J7b_theta = zero_tensor

        self.J8a_a = zero_tensor
        self.J8a_alpha = zero_tensor
        self.J8a_d = zero_tensor
        self.J8a_theta = zero_tensor

        self.J8b_a = zero_tensor
        self.J8b_alpha = one_tensor * 1.571
        self.J8b_d = zero_tensor
        self.J8b_theta = zero_tensor

        self.J9a_a = zero_tensor
        self.J9a_alpha = zero_tensor
        self.J9a_d = zero_tensor
        self.J9a_theta = zero_tensor

        self.J9b_a = zero_tensor
        self.J9b_alpha = one_tensor * 1.571
        self.J9b_d = zero_tensor
        self.J9b_theta = zero_tensor

        self.J10_a = 0.04804 * one_tensor
        self.J10_alpha = zero_tensor
        self.J10_d = zero_tensor
        self.J10_theta = zero_tensor

        self.J11_a = 0.05004 * one_tensor
        self.J11_alpha = zero_tensor
        self.J11_d = zero_tensor
        self.J11_theta = zero_tensor

        self.J12_a = 0.05004 * one_tensor
        self.J12_alpha = zero_tensor
        self.J12_d = zero_tensor
        self.J12_theta = zero_tensor

        self.J13_a = 0.04454 * one_tensor
        self.J13_alpha = zero_tensor
        self.J13_d = zero_tensor
        self.J13_theta = zero_tensor

        self.J14_a = 0.026 * one_tensor
        self.J14_alpha = zero_tensor
        self.J14_d = zero_tensor
        self.J14_theta = zero_tensor

        self.J15_a = 0.032 * one_tensor
        self.J15_alpha = zero_tensor
        self.J15_d = zero_tensor
        self.J15_theta = zero_tensor

        self.J16_a = 0.032 * one_tensor
        self.J16_alpha = zero_tensor
        self.J16_d = zero_tensor
        self.J16_theta = zero_tensor

        self.J17_a = 0.022 * one_tensor
        self.J17_alpha = zero_tensor
        self.J17_d = zero_tensor
        self.J17_theta = zero_tensor

        # self.A1 = torch.tensor(0.0, dtype=torch.float32, device=device)
        # self.A2 = torch.tensor(0.001 * 55, dtype=torch.float32, device=device)
        # self.A3 = torch.tensor(0.001 * 25, dtype=torch.float32, device=device)

        # self.D0 = torch.tensor(0.001 * 9.5, dtype=torch.float32, device=device)
        # self.D1 = torch.tensor(0.0, dtype=torch.float32, device=device)
        # self.D2 = torch.tensor(0.0, dtype=torch.float32, device=device)
        # self.D3 = torch.tensor(0.0, dtype=torch.float32, device=device)

        # self.phi0 = torch.tensor(0.0, dtype=torch.float32, device=device)
        # self.phi1 = torch.tensor(0.0, dtype=torch.float32, device=device)
        # self.phi2 = torch.tensor(0.0, dtype=torch.float32, device=device)
        # self.phi3 = torch.tensor(0.0, dtype=torch.float32, device=device)

        dir_path = os.path.split(os.path.abspath(__file__))[0]

        # self.T = torch.from_numpy(np.load(os.path.join(dir_path, './T.npy')).astype(np.float32)).to(device).reshape(-1, 4, 4)
        # transformation of base link
        # self.base_2_world = torch.tensor([[1.0, 0.0, 0.0, 0.0],
        #                                   [0.0, 1.0, 0.0, 0.0],
        #                                   [0.0, 0.0, 1.0, -0.032],
        #                                   [0.0, 0.0, 0.0, 1.0]], dtype=torch.float32, device=device)

        self.base_2_world = torch.tensor([[0.0, 0.0, -1.0, 0.13],
                                          [0.0, -1.0, 0.0, 0.0],
                                          [-1.0, 0.0, 0.0, 0.0],
                                          [0.0, 0.0, 0.0, 1.0]], dtype=torch.float32, device=device)

        self.base_2_world_normal = torch.tensor([[0.0, 0.0, -1.0, 0.0],
                                                 [0.0, -1.0, 0.0, 0.0],
                                                 [-1.0, 0.0, 0.0, 0.0],
                                                 [0.0, 0.0, 0.0, 1.0]], dtype=torch.float32, device=device)

        # transformation of palm_1 to base link
        self.p1_2_base = torch.tensor([[1.0, 0.0, 0.0, 0.0],
                                       [0.0, 1.0, 0.0, -0.01313],
                                       [0.0, 0.0, 1.0, 0.032],
                                       [0.0, 0.0, 0.0, 1.0]], dtype=torch.float32, device=device)

        self.p1_2_base_normal = torch.tensor([[1.0, 0.0, 0.0, 0.0],
                                              [0.0, 1.0, 0.0, 0.0],
                                              [0.0, 0.0, 1.0, 0.0],
                                              [0.0, 0.0, 0.0, 1.0]], dtype=torch.float32, device=device)

        # transformation of palm_2 to palm_1
        self.p2_2_p1 = torch.tensor([[1.0, 0.0, 0, 0.0184],
                                     [0.0, 1, 0, 0.006],
                                     [0.0, 0.0, 1.0, 0.0375],
                                     [0, 0, 0, 1]], dtype=torch.float32, device=device)

        self.p2_2_p1_normal = torch.tensor([[1.0, 0.0, 0, 0],
                                            [0.0, 1, 0, 0.0],
                                            [0.0, 0.0, 1.0, 0],
                                            [0, 0, 0, 1]], dtype=torch.float32, device=device)

        # transformation of thumb to palm_1
        self.thumb_2_p1 = torch.tensor([[0, -1, 0, -0.0169],
                                        [0.966, 0, 0.2588, 0.02626],
                                        [-0.2588, 0.0, 0.9659, 0],
                                        [0, 0, 0, 1]], dtype=torch.float32, device=device)

        self.thumb_2_p1_normal = torch.tensor([[0, -1, 0, 0],
                                               [0.966, 0, 0.2588, 0],
                                               [-0.2588, 0.0, 0.9659, 0],
                                               [0, 0, 0, 1]], dtype=torch.float32, device=device)
        # transformation of fore to palm_1
        self.fore_2_p1 = torch.tensor([[0, -1, 0, -0.025],
                                       [0.0, 0, -1, 0.0],
                                       [1, 0.0, 0.0, 0.110],
                                       [0, 0, 0, 1]], dtype=torch.float32, device=device)

        self.fore_2_p1_normal = torch.tensor([[0, -1, 0, 0],
                                              [0.0, 0, -1, 0.0],
                                              [1, 0.0, 0.0, 0.0],
                                              [0, 0, 0, 1]], dtype=torch.float32, device=device)

        # middle
        self.middle_2_p1 = torch.tensor([[0, -1, 0, 0.0],
                                         [0.0, 0, -1, 0.0],
                                         [1, 0.0, 0.0, 0.110],
                                         [0, 0, 0, 1]], dtype=torch.float32, device=device)

        self.middle_2_p1_normal = torch.tensor([[0, -1, 0, 0],
                                                [0.0, 0, -1, 0.0],
                                                [1, 0.0, 0.0, 0.0],
                                                [0, 0, 0, 1]], dtype=torch.float32, device=device)

        # ring
        self.ring_2_p2 = torch.tensor([[0, 1, 0, 0.003855],
                                       [0.0, 0, 1, -0.006],
                                       [1, 0.0, 0.0, 0.0655],
                                       [0, 0, 0, 1]], dtype=torch.float32, device=device)

        self.ring_2_p2_normal = torch.tensor([[0, 1, 0, 0],
                                              [0.0, 0, 1, 0.0],
                                              [1, 0.0, 0.0, 0.0],
                                              [0, 0, 0, 1]], dtype=torch.float32, device=device)

        # little
        self.little_2_p2 = torch.tensor([[0, 1, 0, 0.025355],
                                         [0.0, 0, 1, -0.006],
                                         [1, 0.0, 0.0, 0.056],
                                         [0, 0, 0, 1]], dtype=torch.float32, device=device)

        self.little_2_p2_normal = torch.tensor([[0, 1, 0, 0],
                                                [0.0, 0, 1, 0.0],
                                                [1, 0.0, 0.0, 0.0],
                                                [0, 0, 0, 1]], dtype=torch.float32, device=device)

        # self.fore_2_base = torch.tensor([[1.0,  0.0, 0.0, -0.001429881],
        #                                  [0.0,  1.0, 0.0, -0.016800135],
        #                                  [0.0,  0.0, 1.0,       0.122043545],
        #                                  [0,         0,         0,        1]], dtype=torch.float32, device=device)
        #
        # self.fore_2_base_normal = torch.tensor([[1.0,  0.0, 0.0, 0],
        #                                         [0.0,  1.0, 0.0, 0],
        #                                         [0.0,  0.0, 1.0, 0],
        #                                         [0,    0,   0,   1]], dtype=torch.float32, device=device)
        #
        # self.middle_2_base = torch.tensor([[0.0,  0.1736479, 0.9848078,  0.002071571],
        #                           [0.0, -0.9848078, 0.1736479, -0.043396306],
        #                           [1.0,  0.0,      0.0,       0.103043545],
        #                           [0,         0,         0,        1]], dtype=torch.float32, device=device)
        #
        # self.middle_2_base_normal = torch.tensor([[0.0,  0.1736479, 0.9848078,  0],
        #                                  [0.0, -0.9848078, 0.1736479, 0],
        #                                  [1.0,  0.0,      0.0,       0],
        #                                  [0,         0,         0,   1]], dtype=torch.float32, device=device)
        #
        # self.ring_2_p2 = torch.tensor([[0.0,  0.1736479, 0.9848078,  0.002071571],
        #                           [0.0, -0.9848078, 0.1736479, -0.043396306],
        #                           [1.0,  0.0,      0.0,       0.103043545],
        #                           [0,         0,         0,        1]], dtype=torch.float32, device=device)
        #
        # self.ring_2_p2_normal = torch.tensor([[0.0,  0.1736479, 0.9848078,  0],
        #                                  [0.0, -0.9848078, 0.1736479, 0],
        #                                  [1.0,  0.0,      0.0,       0],
        #                                  [0,         0,         0,   1]], dtype=torch.float32, device=device)
        #
        # self.little_2_p2 = torch.tensor([[0.0,  0.1736479, 0.9848078,  0.002071571],
        #                           [0.0, -0.9848078, 0.1736479, -0.043396306],
        #                           [1.0,  0.0,      0.0,       0.103043545],
        #                           [0,         0,         0,        1]], dtype=torch.float32, device=device)
        #
        # self.little_2_p2_normal = torch.tensor([[0.0,  0.1736479, 0.9848078,  0],
        #                                  [0.0, -0.9848078, 0.1736479, 0],
        #                                  [1.0,  0.0,      0.0,       0],
        #                                  [0,         0,         0,   1]], dtype=torch.float32, device=device)

        self.device = device
        self.meshes = self.load_meshes()

        # righthand_base
        self.righthand = self.meshes["righthand_base"][0]
        self.righthand_normal = self.meshes["righthand_base"][2]

        # palm_1 (fore and middle)
        self.h10 = self.meshes['h10'][0]
        self.h10_normal = self.meshes['h10'][2]

        # palm_2 (ring and little)
        self.h11 = self.meshes['h11'][0]
        self.h11_normal = self.meshes['h11'][2]

        # thumb
        self.d10 = self.meshes['d10'][0]
        self.d10_normal = self.meshes['d10'][2]
        self.d11 = self.meshes['d11'][0]
        self.d11_normal = self.meshes['d11'][2]
        self.d12 = self.meshes['d12'][0]
        self.d12_normal = self.meshes['d12'][2]
        self.d13 = self.meshes['d13'][0]
        self.d13_normal = self.meshes['d13'][2]

        # fore
        self.f10 = self.meshes['f10'][0]
        self.f10_normal = self.meshes['f10'][2]
        self.f11 = self.meshes['f11'][0]
        self.f11_normal = self.meshes['f11'][2]
        self.f12 = self.meshes['f12'][0]
        self.f12_normal = self.meshes['f12'][2]
        self.f13 = self.meshes['f13'][0]
        self.f13_normal = self.meshes['f13'][2]

        # middle
        self.f20 = self.meshes['f20'][0]
        self.f20_normal = self.meshes['f20'][2]
        self.f21 = self.meshes['f21'][0]
        self.f21_normal = self.meshes['f21'][2]
        self.f22 = self.meshes['f22'][0]
        self.f22_normal = self.meshes['f22'][2]
        self.f23 = self.meshes['f23'][0]
        self.f23_normal = self.meshes['f23'][2]

        # ring
        self.f30 = self.meshes['f30'][0]
        self.f30_normal = self.meshes['f30'][2]
        self.f31 = self.meshes['f31'][0]
        self.f31_normal = self.meshes['f31'][2]
        self.f32 = self.meshes['f32'][0]
        self.f32_normal = self.meshes['f32'][2]
        self.f33 = self.meshes['f33'][0]
        self.f33_normal = self.meshes['f33'][2]

        # little
        self.f40 = self.meshes['f40'][0]
        self.f40_normal = self.meshes['f40'][2]
        self.f41 = self.meshes['f41'][0]
        self.f41_normal = self.meshes['f41'][2]
        self.f42 = self.meshes['f42'][0]
        self.f42_normal = self.meshes['f42'][2]
        self.f43 = self.meshes['f43'][0]
        self.f43_normal = self.meshes['f43'][2]

        self.gripper_faces = [
            self.meshes["righthand_base"][1],
            self.meshes['h10'][1], self.meshes['h11'][1],
            self.meshes['d10'][1], self.meshes['d11'][1], self.meshes['d12'][1], self.meshes['d13'][1],
            self.meshes['f10'][1], self.meshes['f11'][1], self.meshes['f12'][1], self.meshes['f13'][1],
            self.meshes['f20'][1], self.meshes['f21'][1], self.meshes['f22'][1], self.meshes['f23'][1],
            self.meshes['f30'][1], self.meshes['f31'][1], self.meshes['f32'][1], self.meshes['f33'][1],
            self.meshes['f40'][1], self.meshes['f41'][1], self.meshes['f42'][1], self.meshes['f43'][1],
        ]

        # self.vertex_idxs = np.load('vertex_idxs.npy')

        # self.vertice_face_areas = [
        #     self.meshes["righthand_base"][2],  # self.meshes["palm_2"][2],
        #     self.meshes['base'][2], self.meshes['proximal'][2],
        #     self.meshes['medial'][2], self.meshes['distal'][2]
        # ]

        # self.num_vertices_per_part = [
        #     self.meshes["righthand_base"][0].shape[0],  # self.meshes["palm_2"][0].shape[0],
        #     self.meshes['base'][0].shape[0], self.meshes['proximal'][0].shape[0],
        #     self.meshes['medial'][0].shape[0], self.meshes['distal'][0].shape[0]
        # ]

    # @timeCalc
    def load_meshes(self):
        mesh_path = os.path.dirname(os.path.realpath(__file__)) + "/../meshes/svh_hand_new/*"
        mesh_files = glob.glob(mesh_path)
        mesh_files = [f for f in mesh_files if os.path.isfile(f)]
        meshes = {}
        name_dict = {'righthand_base': 0, 'h10': 1, 'h11': 2,
                     'd10': 3, 'd11': 4, 'd12': 5, 'd13': 6,
                     'f10': 7, 'f11': 8, 'f12': 9, 'f13': 10,
                     'f20': 11, 'f21': 12, 'f22': 13, 'f23': 14,
                     'f30': 15, 'f31': 16, 'f32': 17, 'f33': 18,
                     'f40': 19, 'f41': 20, 'f42': 21, 'f43': 22}
        for mesh_file in mesh_files:
            name = os.path.basename(mesh_file)[:-4]
            obj_id = name_dict[name]
            idxs = np.load(CUR_DIR + '/vert_idxs/vert_ids_{}.npy'.format(obj_id))
            mesh = trimesh.load(mesh_file)
            if self.show_mesh:
                temp = torch.ones(mesh.vertices.shape[0], 1).float()
                vertex_normals = copy.deepcopy(mesh.vertex_normals)
                meshes[name] = [
                    torch.cat((torch.FloatTensor(np.array(mesh.vertices)), temp), dim=-1).to(self.device),
                    mesh.faces,
                    torch.cat((torch.FloatTensor(vertex_normals), temp), dim=-1).to(self.device).to(torch.float)
                ]
            else:
                temp = torch.ones(idxs.shape[0], 1).float()
                vertex_normals = copy.deepcopy(mesh.vertex_normals)
                meshes[name] = [
                    torch.cat((torch.FloatTensor(np.array(mesh.vertices[idxs])), temp), dim=-1).to(self.device),
                    # torch.LongTensor(np.asarray(mesh.faces)).to(self.device),
                    mesh.faces,
                    # torch.FloatTensor(np.asarray(vert_area_weight)).to(self.device),
                    # vert_area_weight,
                    torch.cat((torch.FloatTensor(vertex_normals[idxs]), temp), dim=-1).to(self.device).to(torch.float)
                    # mesh.vertex_normals,
                ]
        return meshes

    # @timeCalc
    def forward(self, pose, theta):
        """[summary]
        Args:
            pose (Tensor (batch_size x 4 x 4)): The pose of the base link of the hand as a translation matrix.
            theta (Tensor (batch_size x 15)): The seven degrees of freedome of the Barrett hand. The first column specifies the angle between
            fingers F1 and F2,  the second to fourth column specifies the joint angle around the proximal link of each finger while the fifth
            to the last column specifies the joint angle around the distal link for each finger

       """
        batch_size = pose.shape[0]
        pose_normal = pose.clone()
        rot_270 = torch.from_numpy(np.identity(4)).to(self.device).float()
        rot_270[:3, :3] = rotation_conversions.axis_angle_to_matrix(torch.tensor([0, np.pi / 2, 0]))
        pose_normal[:, :3, 3] = torch.zeros(3, device=pose.device)

        # right_hand_base
        right_hand_vertices = self.righthand.repeat(batch_size, 1, 1)
        T_base_2_w = torch.matmul(pose, self.base_2_world)
        right_hand_vertices = torch.matmul(T_base_2_w,
                                           right_hand_vertices.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        righthand_vertices_normal = self.righthand_normal.repeat(batch_size, 1, 1)
        T_base_2_w_normal = torch.matmul(pose_normal, self.base_2_world_normal)
        righthand_vertices_normal = torch.matmul(T_base_2_w_normal,
                                                 righthand_vertices_normal.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        # palm_1
        h10_vertices = self.h10.repeat(batch_size, 1, 1)
        T_p1_2_w = torch.matmul(T_base_2_w, self.p1_2_base)
        h10_vertices = torch.matmul(T_p1_2_w,
                                    h10_vertices.transpose(2, 1)).transpose(1, 2)[:, :, :3]
        h10_vertices_normal = self.h10_normal.repeat(batch_size, 1, 1)
        T_p1_2_w_normal = torch.matmul(T_base_2_w_normal, self.p1_2_base_normal)
        h10_vertices_normal = torch.matmul(T_p1_2_w_normal,
                                           h10_vertices_normal.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        # palm_2
        h11_vertices = self.h11.repeat(batch_size, 1, 1)

        T_J5 = self.forward_kinematics(self.J5_a, self.J5_alpha,
                                       self.J5_d, theta[:, 0]/3 + self.J5_theta, batch_size)
        T_p2_2_w = torch.matmul(torch.matmul(T_p1_2_w, self.p2_2_p1), T_J5)
        h11_vertices = torch.matmul(T_p2_2_w,
                                    h11_vertices.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        h11_vertices_normal = self.h11_normal.repeat(batch_size, 1, 1)
        T_p2_2_w_normal = torch.matmul(torch.matmul(T_p1_2_w_normal, self.p2_2_p1_normal), T_J5)
        h11_vertices_normal = torch.matmul(T_p2_2_w_normal,
                                           h11_vertices_normal.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        # thumb
        d10_vertices = self.d10.repeat(batch_size, 1, 1)

        T_J1 = self.forward_kinematics(self.J1_a, self.J1_alpha,
                                       self.J1_d, -(theta[:, 0] - 0.26) + self.J1_theta, batch_size)
        T_thumb0_2_w = torch.matmul(torch.matmul(T_p1_2_w, self.thumb_2_p1), T_J1)
        d10_vertices = torch.matmul(T_thumb0_2_w,
                                    d10_vertices.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        d10_vertices_normal = self.d10_normal.repeat(batch_size, 1, 1)
        T_thumb0_2_w_normal = torch.matmul(torch.matmul(T_p1_2_w_normal, self.thumb_2_p1_normal), T_J1)
        d10_vertices_normal = torch.matmul(T_thumb0_2_w_normal,
                                           d10_vertices_normal.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        d11_vertices = self.d11.repeat(batch_size, 1, 1)
        T_J2 = self.forward_kinematics(self.J2_a, self.J2_alpha,
                                       self.J2_d, (theta[:, 1] - 0.17) + self.J2_theta - 0.9704, batch_size)

        T_thumb1_2_w = torch.matmul(torch.matmul(T_thumb0_2_w, rot_270), T_J2)
        # T_thumb1_2_w = torch.matmul(T_thumb0_2_w, T_J2)
        d11_vertices = torch.matmul(T_thumb1_2_w,  # T_thumb1_2_w
                                    d11_vertices.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        d11_vertices_normal = self.d11_normal.repeat(batch_size, 1, 1)
        T_thumb1_2_w_normal = torch.matmul(torch.matmul(T_thumb0_2_w_normal, rot_270), T_J2)
        d11_vertices_normal = torch.matmul(T_thumb1_2_w_normal,
                                           d11_vertices_normal.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        d12_vertices = self.d12.repeat(batch_size, 1, 1)
        T_J3 = self.forward_kinematics(self.J3_a, self.J3_alpha,
                                       self.J3_d, (1.01511 * theta[:, 1] + 0.35) + self.J3_theta, batch_size)
        T_thumb2_2_w = torch.matmul(T_thumb1_2_w, T_J3)
        d12_vertices = torch.matmul(T_thumb2_2_w,
                                    d12_vertices.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        d12_vertices_normal = self.d12_normal.repeat(batch_size, 1, 1)
        T_thumb2_2_w_normal = torch.matmul(T_thumb1_2_w_normal, T_J3)
        d12_vertices_normal = torch.matmul(T_thumb2_2_w_normal,
                                           d12_vertices_normal.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        d13_vertices = self.d13.repeat(batch_size, 1, 1)
        T_J4 = self.forward_kinematics(self.J4_a, self.J4_alpha,
                                       self.J4_d, (1.44889 * theta[:, 1] + 0.087) + self.J4_theta, batch_size)
        T_thumb3_2_w = torch.matmul(T_thumb2_2_w, T_J4)
        d13_vertices = torch.matmul(T_thumb3_2_w,
                                    d13_vertices.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        d13_vertices_normal = self.d13_normal.repeat(batch_size, 1, 1)
        T_thumb3_2_w_normal = torch.matmul(T_thumb2_2_w_normal, T_J4)
        d13_vertices_normal = torch.matmul(T_thumb3_2_w_normal,
                                           d13_vertices_normal.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        # fore
        f10_vertices = self.f10.repeat(batch_size, 1, 1)
        T_J6a = self.forward_kinematics(self.J6a_a, self.J6a_alpha,
                                        self.J6a_d, (0.8 - theta[:, 2]) / 0.8 * (theta[:, 8] - 0.18) + self.J6a_theta,
                                        batch_size)  # 1x Finger_spread
        T_fore_2_w = torch.matmul(torch.matmul(T_p1_2_w, self.fore_2_p1), T_J6a)
        f10_vertices = torch.matmul(T_fore_2_w,
                                    f10_vertices.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        f10_vertices_normal = self.f10_normal.repeat(batch_size, 1, 1)
        T_fore_2_w_normal = torch.matmul(torch.matmul(T_p1_2_w_normal, self.fore_2_p1_normal), T_J6a)
        f10_vertices_normal = torch.matmul(T_fore_2_w_normal,
                                           f10_vertices_normal.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        f11_vertices = self.f11.repeat(batch_size, 1, 1)
        T_J6b = self.forward_kinematics(self.J6b_a, self.J6b_alpha,
                                        self.J6b_d, (theta[:, 2] + 0.10) + self.J6b_theta, batch_size)
        T_fore1_2_w = torch.matmul(T_fore_2_w, T_J6b)
        f11_vertices = torch.matmul(T_fore1_2_w,
                                    f11_vertices.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        f11_vertices_normal = self.f11_normal.repeat(batch_size, 1, 1)
        T_fore1_2_w_normal = torch.matmul(T_fore_2_w_normal, T_J6b)
        f11_vertices_normal = torch.matmul(T_fore1_2_w_normal,
                                           f11_vertices_normal.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        f12_vertices = self.f12.repeat(batch_size, 1, 1)
        T_J10 = self.forward_kinematics(self.J10_a, self.J10_alpha,
                                        self.J10_d, (theta[:, 3] + 0.26) + self.J10_theta, batch_size)
        T_fore2_2_w = torch.matmul(T_fore1_2_w, T_J10)
        f12_vertices = torch.matmul(T_fore2_2_w,
                                    f12_vertices.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        f12_vertices_normal = self.f12_normal.repeat(batch_size, 1, 1)
        T_fore2_2_w_normal = torch.matmul(T_fore1_2_w_normal, T_J10)
        f12_vertices_normal = torch.matmul(T_fore2_2_w_normal,
                                           f12_vertices_normal.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        f13_vertices = self.f13.repeat(batch_size, 1, 1)
        T_J14 = self.forward_kinematics(self.J14_a, self.J14_alpha,
                                        self.J14_d, 1.045 * theta[:, 3] + self.J14_theta, batch_size)
        T_fore3_2_w = torch.matmul(T_fore2_2_w, T_J14)
        f13_vertices = torch.matmul(T_fore3_2_w,
                                    f13_vertices.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        f13_vertices_normal = self.f13_normal.repeat(batch_size, 1, 1)
        T_fore3_2_w_normal = torch.matmul(T_fore2_2_w_normal, T_J14)
        f13_vertices_normal = torch.matmul(T_fore3_2_w_normal,
                                           f13_vertices_normal.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        # middle
        f20_vertices = self.f20.repeat(batch_size, 1, 1)
        T_J7a = self.forward_kinematics(self.J7a_a, self.J7a_alpha,
                                        self.J7a_d, self.J7a_theta, batch_size)  # No Finger_spread
        T_middle_2_w = torch.matmul(torch.matmul(T_p1_2_w, self.middle_2_p1), T_J7a)
        f20_vertices = torch.matmul(T_middle_2_w,
                                    f20_vertices.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        f20_vertices_normal = self.f20_normal.repeat(batch_size, 1, 1)
        T_middle_2_w_normal = torch.matmul(torch.matmul(T_p1_2_w_normal, self.middle_2_p1_normal), T_J7a)
        f20_vertices_normal = torch.matmul(T_middle_2_w_normal,
                                           f20_vertices_normal.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        f21_vertices = self.f21.repeat(batch_size, 1, 1)
        T_J7b = self.forward_kinematics(self.J7b_a, self.J7b_alpha,
                                        self.J7b_d, (theta[:, 4] + 0.10) + self.J7b_theta, batch_size)
        T_middle1_2_w = torch.matmul(T_middle_2_w, T_J7b)
        f21_vertices = torch.matmul(T_middle1_2_w,
                                    f21_vertices.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        f21_vertices_normal = self.f21_normal.repeat(batch_size, 1, 1)
        T_middle1_2_w_normal = torch.matmul(T_middle_2_w_normal, T_J7b)
        f21_vertices_normal = torch.matmul(T_middle1_2_w_normal,
                                           f21_vertices_normal.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        f22_vertices = self.f22.repeat(batch_size, 1, 1)
        T_J11 = self.forward_kinematics(self.J11_a, self.J11_alpha,
                                        self.J11_d, (theta[:, 5] + 0.26) + self.J11_theta, batch_size)
        T_middle2_2_w = torch.matmul(T_middle1_2_w, T_J11)
        f22_vertices = torch.matmul(T_middle2_2_w,
                                    f22_vertices.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        f22_vertices_normal = self.f22_normal.repeat(batch_size, 1, 1)
        T_middle2_2_w_normal = torch.matmul(T_middle1_2_w_normal, T_J11)
        f22_vertices_normal = torch.matmul(T_middle2_2_w_normal,
                                           f22_vertices_normal.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        f23_vertices = self.f23.repeat(batch_size, 1, 1)
        T_J15 = self.forward_kinematics(self.J15_a, self.J15_alpha,
                                        self.J15_d, 1.0434 * theta[:, 5] + self.J15_theta, batch_size)
        T_middle3_2_w = torch.matmul(T_middle2_2_w, T_J15)
        f23_vertices = torch.matmul(T_middle3_2_w,
                                    f23_vertices.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        f23_vertices_normal = self.f23_normal.repeat(batch_size, 1, 1)
        T_middle3_2_w_normal = torch.matmul(T_middle2_2_w_normal, T_J15)
        f23_vertices_normal = torch.matmul(T_middle3_2_w_normal,
                                           f23_vertices_normal.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        # ring
        f30_vertices = self.f30.repeat(batch_size, 1, 1)
        T_J8a = self.forward_kinematics(self.J8a_a, self.J8a_alpha,
                                        self.J8a_d, (theta[:, 8] - 0.18) + self.J8a_theta,
                                        batch_size)  # 1x Finger_spread
        T_ring_2_w = torch.matmul(torch.matmul(T_p2_2_w, self.ring_2_p2), T_J8a)
        f30_vertices = torch.matmul(T_ring_2_w,
                                    f30_vertices.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        f30_vertices_normal = self.f30_normal.repeat(batch_size, 1, 1)
        T_ring_2_w_normal = torch.matmul(torch.matmul(T_p2_2_w_normal, self.ring_2_p2_normal), T_J8a)
        f30_vertices_normal = torch.matmul(T_ring_2_w_normal,
                                           f30_vertices_normal.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        f31_vertices = self.f31.repeat(batch_size, 1, 1)
        T_J8b = self.forward_kinematics(self.J8b_a, self.J8b_alpha,
                                        self.J8b_d, (theta[:, 6] + 0.23) + self.J8b_theta, batch_size)
        T_ring1_2_w = torch.matmul(T_ring_2_w, T_J8b)
        f31_vertices = torch.matmul(T_ring1_2_w,
                                    f31_vertices.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        f31_vertices_normal = self.f31_normal.repeat(batch_size, 1, 1)
        T_ring1_2_w_normal = torch.matmul(T_ring_2_w_normal, T_J8b)
        f31_vertices_normal = torch.matmul(T_ring1_2_w_normal,
                                           f31_vertices_normal.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        f32_vertices = self.f32.repeat(batch_size, 1, 1)
        T_J12 = self.forward_kinematics(self.J12_a, self.J12_alpha,
                                        self.J12_d, (1.3588 * theta[:, 6] + 0.42) + self.J12_theta, batch_size)
        T_ring2_2_w = torch.matmul(T_ring1_2_w, T_J12)
        f32_vertices = torch.matmul(T_ring2_2_w,
                                    f32_vertices.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        f32_vertices_normal = self.f32_normal.repeat(batch_size, 1, 1)
        T_ring2_2_w_normal = torch.matmul(T_ring1_2_w_normal, T_J12)
        f32_vertices_normal = torch.matmul(T_ring2_2_w_normal,
                                           f32_vertices_normal.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        f33_vertices = self.f33.repeat(batch_size, 1, 1)
        T_J16 = self.forward_kinematics(self.J16_a, self.J16_alpha,
                                        self.J16_d, (1.42093 * theta[:, 6] + 0.17) + self.J16_theta, batch_size)
        T_ring3_2_w = torch.matmul(T_ring2_2_w, T_J16)
        f33_vertices = torch.matmul(T_ring3_2_w,
                                    f33_vertices.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        f33_vertices_normal = self.f33_normal.repeat(batch_size, 1, 1)
        T_ring3_2_w_normal = torch.matmul(T_ring2_2_w_normal, T_J16)
        f33_vertices_normal = torch.matmul(T_ring3_2_w_normal,
                                           f33_vertices_normal.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        # little
        f40_vertices = self.f40.repeat(batch_size, 1, 1)
        T_J9a = self.forward_kinematics(self.J9a_a, self.J9a_alpha,
                                        self.J9a_d, 2 * (theta[:, 8] - 0.18) + self.J9a_theta,
                                        batch_size)  # 2x Finger_spread
        T_little_2_w = torch.matmul(torch.matmul(T_p2_2_w, self.little_2_p2), T_J9a)
        f40_vertices = torch.matmul(T_little_2_w,
                                    f40_vertices.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        f40_vertices_normal = self.f40_normal.repeat(batch_size, 1, 1)
        T_little_2_w_normal = torch.matmul(torch.matmul(T_p2_2_w_normal, self.little_2_p2_normal), T_J9a)
        f40_vertices_normal = torch.matmul(T_little_2_w_normal,
                                           f40_vertices_normal.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        f41_vertices = self.f41.repeat(batch_size, 1, 1)
        T_J9b = self.forward_kinematics(self.J9b_a, self.J9b_alpha,
                                        self.J9b_d, (theta[:, 7] + 0.23) + self.J9b_theta, batch_size)
        T_little1_2_w = torch.matmul(T_little_2_w, T_J9b)
        f41_vertices = torch.matmul(T_little1_2_w,
                                    f41_vertices.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        f41_vertices_normal = self.f41_normal.repeat(batch_size, 1, 1)
        T_little1_2_w_normal = torch.matmul(T_little_2_w_normal, T_J9b)
        f41_vertices_normal = torch.matmul(T_little1_2_w_normal,
                                           f41_vertices_normal.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        f42_vertices = self.f42.repeat(batch_size, 1, 1)
        T_J13 = self.forward_kinematics(self.J13_a, self.J13_alpha,
                                        self.J13_d, (1.3588 * theta[:, 7] + 0.24) + self.J13_theta, batch_size)
        T_little2_2_w = torch.matmul(T_little1_2_w, T_J13)
        f42_vertices = torch.matmul(T_little2_2_w,
                                    f42_vertices.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        f42_vertices_normal = self.f42_normal.repeat(batch_size, 1, 1)
        T_little2_2_w_normal = torch.matmul(T_little1_2_w_normal, T_J13)
        f42_vertices_normal = torch.matmul(T_little2_2_w_normal,
                                           f42_vertices_normal.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        f43_vertices = self.f43.repeat(batch_size, 1, 1)
        T_J17 = self.forward_kinematics(self.J17_a, self.J17_alpha,
                                        self.J17_d, 1.42307 * theta[:, 7] + self.J17_theta, batch_size)
        T_little3_2_w = torch.matmul(T_little2_2_w, T_J17)
        f43_vertices = torch.matmul(T_little3_2_w,
                                    f43_vertices.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        f43_vertices_normal = self.f43_normal.repeat(batch_size, 1, 1)
        T_little3_2_w_normal = torch.matmul(T_little2_2_w_normal, T_J17)
        f43_vertices_normal = torch.matmul(T_little3_2_w_normal,
                                           f43_vertices_normal.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        return right_hand_vertices, h10_vertices, h11_vertices, \
               d10_vertices, d11_vertices, d12_vertices, d13_vertices, \
               f10_vertices, f11_vertices, f12_vertices, f13_vertices, \
               f20_vertices, f21_vertices, f22_vertices, f23_vertices, \
               f30_vertices, f31_vertices, f32_vertices, f33_vertices, \
               f40_vertices, f41_vertices, f42_vertices, f43_vertices, \
               righthand_vertices_normal, h10_vertices_normal, h11_vertices_normal, \
               d10_vertices_normal, d11_vertices_normal, d12_vertices_normal, d13_vertices_normal, \
               f10_vertices_normal, f11_vertices_normal, f12_vertices_normal, f13_vertices_normal, \
               f20_vertices_normal, f21_vertices_normal, f22_vertices_normal, f23_vertices_normal, \
               f30_vertices_normal, f31_vertices_normal, f32_vertices_normal, f33_vertices_normal, \
               f40_vertices_normal, f41_vertices_normal, f42_vertices_normal, f43_vertices_normal

    # @timeCalc
    def forward_kinematics(self, A, alpha, D, theta, batch_size=1):
        c_theta = torch.cos(theta)
        s_theta = torch.sin(theta)
        c_alpha = torch.cos(alpha)
        s_alpha = torch.sin(alpha)
        l_1_to_l = torch.zeros((batch_size, 4, 4), device=self.device)
        l_1_to_l[:, 0, 0] = c_theta
        l_1_to_l[:, 0, 1] = -s_theta
        l_1_to_l[:, 0, 3] = A
        l_1_to_l[:, 1, 0] = s_theta * c_alpha
        l_1_to_l[:, 1, 1] = c_theta * c_alpha
        l_1_to_l[:, 1, 2] = -s_alpha
        l_1_to_l[:, 1, 3] = -s_alpha * D
        l_1_to_l[:, 2, 0] = s_theta * s_alpha
        l_1_to_l[:, 2, 1] = c_theta * s_alpha
        l_1_to_l[:, 2, 2] = c_alpha
        l_1_to_l[:, 2, 3] = c_alpha * D
        l_1_to_l[:, 3, 3] = 1
        return l_1_to_l

    def get_hand_mesh(self, vertices_list, faces, save_mesh=True, path='./output_mesh'):
        if save_mesh:
            assert os.path.exists(path)

        right_hand_verts = vertices_list[0]
        right_hand_faces = faces[0]
        palm1_verts = vertices_list[1]
        palm1_faces = faces[1]
        palm2_verts = vertices_list[2]
        palm2_faces = faces[2]

        thumb_base_verts = vertices_list[3]
        thumb_base_faces = faces[3]
        thumb_proximal_verts = vertices_list[4]
        thumb_proximal_faces = faces[4]
        thumb_medial_verts = vertices_list[5]
        thumb_medial_faces = faces[5]
        thumb_distal_verts = vertices_list[6]
        thumb_distal_faces = faces[6]

        fore_base_verts = vertices_list[7]
        fore_base_faces = faces[7]
        fore_proximal_verts = vertices_list[8]
        fore_proximal_faces = faces[8]
        fore_medial_verts = vertices_list[9]
        fore_medial_faces = faces[9]
        fore_distal_verts = vertices_list[10]
        fore_distal_faces = faces[10]

        middle_base_verts = vertices_list[11]
        middle_base_faces = faces[11]
        middle_proximal_verts = vertices_list[12]
        middle_proximal_faces = faces[12]
        middle_medial_verts = vertices_list[13]
        middle_medial_faces = faces[13]
        middle_distal_verts = vertices_list[14]
        middle_distal_faces = faces[14]

        ring_base_verts = vertices_list[15]
        ring_base_faces = faces[15]
        ring_proximal_verts = vertices_list[16]
        ring_proximal_faces = faces[16]
        ring_medial_verts = vertices_list[17]
        ring_medial_faces = faces[17]
        ring_distal_verts = vertices_list[18]
        ring_distal_faces = faces[18]

        little_base_verts = vertices_list[19]
        little_base_faces = faces[19]
        little_proximal_verts = vertices_list[20]
        little_proximal_faces = faces[20]
        little_medial_verts = vertices_list[21]
        little_medial_faces = faces[21]
        little_distal_verts = vertices_list[22]
        little_distal_faces = faces[22]

        if save_mesh:
            save_to_mesh(right_hand_verts, right_hand_faces, '{}/svh_base.obj'.format(path))
            save_to_mesh(palm1_verts, palm1_faces, '{}/svh_palm1.obj'.format(path))
            save_to_mesh(palm2_verts, palm2_faces, '{}/svh_palm2.obj'.format(path))
            save_to_mesh(thumb_base_verts, thumb_base_faces, '{}/svh_thumb_base.obj'.format(path))
            save_to_mesh(thumb_proximal_verts, thumb_proximal_faces, '{}/svh_thumb_proximal.obj'.format(path))
            save_to_mesh(thumb_medial_verts, thumb_medial_faces, '{}/svh_thumb_medial.obj'.format(path))
            save_to_mesh(thumb_distal_verts, thumb_distal_faces, '{}/svh_thumb_distal.obj'.format(path))
            save_to_mesh(fore_base_verts, fore_base_faces, '{}/svh_fore_base.obj'.format(path))
            save_to_mesh(fore_proximal_verts, fore_proximal_faces, '{}/svh_fore_proximal.obj'.format(path))
            save_to_mesh(fore_medial_verts, fore_medial_faces, '{}/svh_fore_medial.obj'.format(path))
            save_to_mesh(fore_distal_verts, fore_distal_faces, '{}/svh_fore_distal.obj'.format(path))
            save_to_mesh(middle_base_verts, middle_base_faces, '{}/svh_middle_base.obj'.format(path))
            save_to_mesh(middle_proximal_verts, middle_proximal_faces, '{}/svh_middle_proximal.obj'.format(path))
            save_to_mesh(middle_medial_verts, middle_medial_faces, '{}/svh_middle_medial.obj'.format(path))
            save_to_mesh(middle_distal_verts, middle_distal_faces, '{}/svh_middle_distal.obj'.format(path))
            save_to_mesh(ring_base_verts, ring_base_faces, '{}/svh_ring_base.obj'.format(path))
            save_to_mesh(ring_proximal_verts, ring_proximal_faces, '{}/svh_ring_proximal.obj'.format(path))
            save_to_mesh(ring_medial_verts, ring_medial_faces, '{}/svh_ring_medial.obj'.format(path))
            save_to_mesh(ring_distal_verts, ring_distal_faces, '{}/svh_ring_distal.obj'.format(path))
            save_to_mesh(little_base_verts, little_base_faces, '{}/svh_little_base.obj'.format(path))
            save_to_mesh(little_proximal_verts, little_proximal_faces, '{}/svh_little_proximal.obj'.format(path))
            save_to_mesh(little_medial_verts, little_medial_faces, '{}/svh_little_medial.obj'.format(path))
            save_to_mesh(little_distal_verts, little_distal_faces, '{}/svh_little_distal.obj'.format(path))

            hand_mesh = []
            for root, dirs, files in os.walk('{}'.format(path)):
                for filename in files:
                    if filename.endswith('.obj'):
                        filepath = os.path.join(root, filename)
                        mesh = trimesh.load_mesh(filepath)
                        hand_mesh.append(mesh)
            hand_mesh = np.sum(hand_mesh)
        else:
            right_hand_mesh = trimesh.Trimesh(right_hand_verts, right_hand_faces)
            palm1_mesh = trimesh.Trimesh(palm1_verts, palm1_faces)
            palm2_mesh = trimesh.Trimesh(palm2_verts, palm2_faces)
            thumb_base_mesh = trimesh.Trimesh(thumb_base_verts, thumb_base_faces)
            thumb_proximal_mesh = trimesh.Trimesh(thumb_proximal_verts, thumb_proximal_faces)
            thumb_medial_mesh = trimesh.Trimesh(thumb_medial_verts, thumb_medial_faces)
            thumb_distal_mesh = trimesh.Trimesh(thumb_distal_verts, thumb_distal_faces)
            fore_base_mesh = trimesh.Trimesh(fore_base_verts, fore_base_faces)
            fore_proximal_mesh = trimesh.Trimesh(fore_proximal_verts, fore_proximal_faces)
            fore_medial_mesh = trimesh.Trimesh(fore_medial_verts, fore_medial_faces)
            fore_distal_mesh = trimesh.Trimesh(fore_distal_verts, fore_distal_faces)
            middle_base_mesh = trimesh.Trimesh(middle_base_verts, middle_base_faces)
            middle_proximal_mesh = trimesh.Trimesh(middle_proximal_verts, middle_proximal_faces)
            middle_medial_mesh = trimesh.Trimesh(middle_medial_verts, middle_medial_faces)
            middle_distal_mesh = trimesh.Trimesh(middle_distal_verts, middle_distal_faces)
            ring_base_mesh = trimesh.Trimesh(ring_base_verts, ring_base_faces)
            ring_proximal_mesh = trimesh.Trimesh(ring_proximal_verts, ring_proximal_faces)
            ring_medial_mesh = trimesh.Trimesh(ring_medial_verts, ring_medial_faces)
            ring_distal_mesh = trimesh.Trimesh(ring_distal_verts, ring_distal_faces)
            little_base_mesh = trimesh.Trimesh(little_base_verts, little_base_faces)
            little_proximal_mesh = trimesh.Trimesh(little_proximal_verts, little_proximal_faces)
            little_medial_mesh = trimesh.Trimesh(little_medial_verts, little_medial_faces)
            little_distal_mesh = trimesh.Trimesh(little_distal_verts, little_distal_faces)

            hand_mesh = [right_hand_mesh, palm1_mesh, palm2_mesh,
                         thumb_base_mesh, thumb_proximal_mesh, thumb_medial_mesh, thumb_distal_mesh,
                         fore_base_mesh, fore_proximal_mesh, fore_medial_mesh, fore_distal_mesh,
                         middle_base_mesh, middle_proximal_mesh, middle_medial_mesh, middle_distal_mesh,
                         ring_base_mesh, ring_proximal_mesh, ring_medial_mesh, ring_distal_mesh,
                         little_base_mesh, little_proximal_mesh, little_medial_mesh, little_distal_mesh
                         ]

        return hand_mesh

    def get_forward_hand_mesh(self, pose, theta, save_mesh=False, path='./output_mesh'):
        batch_size = pose.size()[0]
        outputs = self.forward(pose, theta)

        hand_vertices_list = [[outputs[j][i].detach().cpu().numpy() for j in range(23)]
                              for i in range(batch_size)]

        hand_meshes = [self.get_hand_mesh(hand_vertices, self.gripper_faces, save_mesh=save_mesh, path=path) for
                       hand_vertices in hand_vertices_list]
        hand_meshes = [np.sum(hand_mesh) for hand_mesh in hand_meshes]

        return hand_meshes

    # @timeCalc
    def get_forward_vertices(self, pose, theta):
        batch_size = pose.size()[0]
        outputs = self.forward(pose, theta)
        # s = 0
        # l = []
        # for i in range(23):
        #     s += outputs[i].shape[1]
        #     print(outputs[i].shape[1], s)
        #     l.append(s)
        # np.save('idx_list', l)
        # exit()

        hand_vertices = torch.cat((outputs[0].view(batch_size, -1, 3),
                                   outputs[1].view(batch_size, -1, 3),
                                   outputs[2].view(batch_size, -1, 3),
                                   outputs[3].view(batch_size, -1, 3),
                                   outputs[4].view(batch_size, -1, 3),
                                   outputs[5].view(batch_size, -1, 3),
                                   outputs[6].view(batch_size, -1, 3),
                                   outputs[7].view(batch_size, -1, 3),
                                   outputs[8].view(batch_size, -1, 3),
                                   outputs[9].view(batch_size, -1, 3),
                                   outputs[10].view(batch_size, -1, 3),
                                   outputs[11].view(batch_size, -1, 3),
                                   outputs[12].view(batch_size, -1, 3),
                                   outputs[13].view(batch_size, -1, 3),
                                   outputs[14].view(batch_size, -1, 3),
                                   outputs[15].view(batch_size, -1, 3),
                                   outputs[16].view(batch_size, -1, 3),
                                   outputs[17].view(batch_size, -1, 3),
                                   outputs[18].view(batch_size, -1, 3),
                                   outputs[19].view(batch_size, -1, 3),
                                   outputs[20].view(batch_size, -1, 3),
                                   outputs[21].view(batch_size, -1, 3),
                                   outputs[22].view(batch_size, -1, 3)), 1)

        hand_vertices_normal = torch.cat((outputs[23].view(batch_size, -1, 3),
                                          outputs[24].view(batch_size, -1, 3),
                                          outputs[25].view(batch_size, -1, 3),
                                          outputs[26].view(batch_size, -1, 3),
                                          outputs[27].view(batch_size, -1, 3),
                                          outputs[28].view(batch_size, -1, 3),
                                          outputs[29].view(batch_size, -1, 3),
                                          outputs[30].view(batch_size, -1, 3),
                                          outputs[31].view(batch_size, -1, 3),
                                          outputs[32].view(batch_size, -1, 3),
                                          outputs[33].view(batch_size, -1, 3),
                                          outputs[34].view(batch_size, -1, 3),
                                          outputs[35].view(batch_size, -1, 3),
                                          outputs[36].view(batch_size, -1, 3),
                                          outputs[37].view(batch_size, -1, 3),
                                          outputs[38].view(batch_size, -1, 3),
                                          outputs[39].view(batch_size, -1, 3),
                                          outputs[40].view(batch_size, -1, 3),
                                          outputs[41].view(batch_size, -1, 3),
                                          outputs[42].view(batch_size, -1, 3),
                                          outputs[43].view(batch_size, -1, 3),
                                          outputs[44].view(batch_size, -1, 3),
                                          outputs[45].view(batch_size, -1, 3)), 1)

        return hand_vertices, hand_vertices_normal


if __name__ == "__main__":
    device = 'cuda'
    svh = SvhHandLayer(device, show_mesh=True).to(device)
    pose = torch.from_numpy(np.identity(4)).to(device).reshape(-1, 4, 4).float()
    # 0:Thumb Opposition, 1:Thumb Flexion, 2:Index Finger Proximal, 3:Index Finger Distal, 4:Middle Finger Proximal,
    # 5:Middle Finger Distal, 6:Ring Finger, 7:Pinky, 8:Spread
    # theta = torch.ones((1, 9), dtype=torch.float32).to(device) * 0.1
    theta = torch.zeros((1, 9), dtype=torch.float32).to(device)
    theta[0][0] = 0.99
    theta[0][2] = 0
    theta[0][8] = 0.18

    vertices, normal = svh.get_forward_vertices(pose, theta)
    # import pytorch3d.transforms
    # vertices = vertices.matmul(
    #     pytorch3d.transforms.euler_angles_to_matrix(torch.tensor([[3.14, -1.57, -0.25]], device='cuda'), 'XYZ'))
    vertices = vertices.detach().cpu()
    normal = normal.detach().cpu()
    mesh = svh.get_forward_hand_mesh(pose, theta)
    mesh[0].show()
    exit()

    point_cloud = trimesh.PointCloud(vertices[0], colors=[0, 255, 0])
    color = np.asarray(point_cloud.colors)
    color[1834:2156, :] = [255, 0, 0, 255]
    color[2186:2359, :] = [125, 125, 0, 255]
    color[2386:2573, :] = [0, 125, 125, 255]
    color[2602:2796, :] = [125, 0, 125, 255]
    color[2826:3000, :] = [0, 0, 255, 255]
    point_cloud.colors = color
    point_cloud.show()
    exit()

    v1 = vertices[:, 1834:2156, :]
    v2 = vertices[:, 2186:2359, :]
    v3 = vertices[:, 2386:2573, :]
    v4 = vertices[:, 2602:2796, :]
    v5 = vertices[:, 2826:3000, :]
    v = [v1, v2, v3, v4, v5]

    n1 = normal[:, 1834:2156, :]
    n2 = normal[:, 2186:2359, :]
    n3 = normal[:, 2386:2573, :]
    n4 = normal[:, 2602:2796, :]
    n5 = normal[:, 2826:3000, :]
    n = [n1, n2, n3, n4, n5]

    for i in range(5):
        for j in range(5):
            if i == j:
                continue
            _, i2j, _, _ = point2point_signed(v[i], v[j], n[i], n[j])
            # print(len(i2j[0]))
            dist_neg = i2j < 0.0
            print(sum(sum(dist_neg)))
    exit()
    # idxs = fps(point_cloud.vertices, 3000)
    # np.save('idxs.npy', idxs)
    # idxs = np.load('idxs.npy')
    #
    # idx = np.where(idxs > 43910, idxs, 43910)
    # idx = np.where(idx < 55101, idx, 55100)

    # point_cloud = trimesh.PointCloud(point_cloud.vertices[idx], colors=[0, 255, 0])
    # exit()
    # point_cloud.show()
    # exit()

    # sub_mesh = mesh[0].subdivide_to_size(0.004)
    # exit()
    # sub_mesh.show()

    # mesh[0].show()
    # vg = trimesh.voxel.creation.local_voxelize(mesh[0], point=[0, 0, 0], radius=160, pitch=0.0015)
    # vg.show()
    # vg_mesh = vg.marching_cubes
    # vg_mesh.show()
    # exit()

    ray_origins = vertices.squeeze()
    ray_directions = normal.squeeze()
    ray_visualize = trimesh.load_path(np.hstack((ray_origins, ray_origins + ray_directions / 100)).reshape(-1, 2, 3))
    pc = trimesh.PointCloud(ray_origins, colors=[0, 255, 0])
    scene = trimesh.Scene([pc, ray_visualize])
    scene.show()
