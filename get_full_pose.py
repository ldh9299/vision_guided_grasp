import numpy as np
import open3d.open3d.visualization
import torch
import trimesh

from manotorch.manolayer import ManoLayer, MANOOutput
from manotorch.utils.visutils import display_hand_open3d, display_hand_pyrender
from manotorch.axislayer import AxisLayer
from manotorch.anchorlayer import AnchorLayer
import pytorch3d.transforms
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

    x_near, y_near, xidx_near, yidx_near = ch_dist(x,y)

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


datas = np.load('object_in_use_all.npy', allow_pickle=True)
# print(datas[0].keys())
# exit()
axis_layer = AxisLayer()
anchor_layer = AnchorLayer(anchor_root="assets/anchor")

for i in range(len(datas)):
    mano_layer = ManoLayer(
        rot_mode="axisang",
        use_pca=True,
        side="right",
        center_idx=None,
        mano_assets_root="assets/mano",
        flat_hand_mean=False,
        ncomps=15,
    )
    print(datas[i]['human_id'], datas[i]['object_name'])
    pose = datas[i]['pose']
    betas = datas[i]['betas']
    # print(pose, betas)
    mano_results: MANOOutput = mano_layer(torch.tensor([pose]), torch.tensor([betas]))
    axis_angle = mano_results.full_poses[0][3:].view(-1, 3)
    matrix = pytorch3d.transforms.axis_angle_to_matrix(axis_angle)
    euler = pytorch3d.transforms.matrix_to_euler_angles(matrix, 'XYZ')
    J = np.asarray([euler[12][0],
                    euler[12][2],
                    euler[13][2],
                    euler[14][2],
                    euler[0][1],
                    euler[0][2],
                    euler[1][2],
                    euler[2][2],
                    euler[3][2],
                    euler[4][2],
                    euler[5][2],
                    euler[6][1],
                    euler[6][2],
                    euler[7][2],
                    euler[8][2],
                    euler[9][1],
                    euler[9][2],
                    euler[10][2],
                    euler[11][2]])
    # print(euler)
    # print(J)
    datas[i]['J'] = J
    # exit()
    # verts = mano_results.verts
    # joints = mano_results.joints
    # # print('joints', joints)
    # transforms_abs = mano_results.transforms_abs
    # # print('transforms_abs', transforms_abs)
    #
    # anchors = anchor_layer(verts)
    # bul_axes = axis_layer(joints, transforms_abs)
    # # display_hand_pyrender(mano_results, mano_layer.th_faces, bul_axes=bul_axes, anchors=anchors)
    # # exit()
    #
    import open3d as o3d

    # geometry = o3d.geometry.TriangleMesh()
    # geometry.triangles = o3d.utility.Vector3iVector(mano_layer.th_faces)
    verts, joints = mano_results.verts[0], mano_results.joints[0]
    # verts = verts.matmul(
    #     pytorch3d.transforms.euler_angles_to_matrix(torch.tensor([3.14, -1.57, 0]), 'XYZ'))
    # pc = o3d.geometry.PointCloud()
    # pc.points = o3d.utility.Vector3dVector(verts.detach().cpu().numpy())
    # pc.paint_uniform_color([0, 1, 0])
    # # pc.transform(datas[i]['root_mat'])
    # open3d.open3d.visualization.draw_geometries([pc])
    pc = trimesh.PointCloud(verts.detach().cpu().numpy())

    pc.show()
    # pc.apply_transform(datas[i]['root_mat'])
    # pc.show()
    continue
    exit()

    # geometry.vertices = o3d.utility.Vector3dVector(verts.detach().cpu().numpy())
    # geometry.transform(datas[i]['root_mat'])
    # geometry.compute_vertex_normals()
    #
    contactmap_filename = '/home/ldh/ContactPose/data/contactpose_data/full{}_use/{}/{}.ply'.format(
        datas[i]['human_id'], datas[i]['object_name'], datas[i]['object_name'])
    idx = np.load('dataset/idx_3000/{}.npy'.format(datas[i]['object_name']))
    mesh = o3d.io.read_triangle_mesh(contactmap_filename)
    pc_o = o3d.geometry.PointCloud()
    pc_o.points = o3d.utility.Vector3dVector(np.asarray(mesh.vertices)[idx])
    pc_o.paint_uniform_color([1, 0, 0])
    #
    # # print(np.asarray(mesh.vertices).shape)
    # # exit()
    mesh.compute_vertex_normals()
    # print(datas[i].keys())
    # print(datas[i]['root_mat'])
    # open3d.open3d.visualization.draw_geometries([pc, pc_o])

    v_o = torch.unsqueeze(torch.tensor(mesh.vertices)[idx], 0).float()
    n_o = torch.unsqueeze(torch.tensor(mesh.vertex_normals)[idx], 0).float()
    v_h = torch.unsqueeze(torch.tensor(geometry.vertices), 0).float()
    n_h = torch.unsqueeze(torch.tensor(geometry.vertex_normals), 0).float()
    # print(v_o.shape, n_o.shape, v_h.shape, n_h.shape)
    o2h_signed, h2o, _, _ = point2point_signed(v_h, v_o, n_h, n_o)

    w_dist_neg = o2h_signed < 0.0
    v_dist_neg = torch.logical_and(h2o.abs() < 0.015, h2o < 0.0)
    color = np.asarray(pc_o.colors)
    color[w_dist_neg[0], :] = [0, 0, 1]
    pc_o.colors=o3d.utility.Vector3dVector(color)
    color = np.asarray(pc.colors)
    color[v_dist_neg[0], :] = [0, 0, 1]
    pc.colors = o3d.utility.Vector3dVector(color)
    open3d.open3d.visualization.draw_geometries([pc, pc_o])

    # exit()
    # assert len(mano_results.full_poses[0]) == 48
    # datas[i]['full_hand_pose'] = np.asarray(mano_results.full_poses[0])
    # print(datas[i]['full_hand_pose'])
    #
    # exit()

    # display_hand_open3d(mano_results, mano_layer.th_faces)
    # exit()

# np.save('object_in_use_all.npy', datas)