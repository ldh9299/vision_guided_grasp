import os
import sys

import pytorch3d.transforms
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import trimesh

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
from pointnet2.pointnet2.pointnet2_modules import PointnetSAModule
from svh_kinematics.svh_layer.svhhand_layer import SvhHandLayer
from tool.train_tool import point2point_signed


# quat_loss
def quaternion_loss(y_true, y_pred):
    dist1 = torch.mean(torch.abs(y_true - y_pred), dim=-1)
    dist2 = torch.mean(torch.abs(y_true + y_pred), dim=-1)
    loss = torch.where(dist1 < dist2, dist1, dist2)
    return torch.mean(loss)


class backbone_pointnet2(nn.Module):
    def __init__(self, config):
        super(backbone_pointnet2, self).__init__()
        self.config = config
        self.only_touch_idxs = np.load('dataset/only_touch_idxs.npy')

        self.sa1 = PointnetSAModule(mlp=[24, 64, 64, 128], npoint=512, radius=0.025, nsample=64, bn=not True)
        self.sa2 = PointnetSAModule(mlp=[128, 128, 128, 256], npoint=128, radius=0.05, nsample=64, bn=not True)
        self.sa3 = PointnetSAModule(mlp=[256, 256, 256, 512], bn=not True)
        # self.sa4 = PointnetSAModule(mlp=[256, 256, 256, 512], npoint=None, radius=None, nsample=None, bn=True)

        # fc layer
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc_translation = nn.Linear(128, 3)
        self.fc_quat = nn.Linear(128, 4)
        self.fc_joints = nn.Linear(128, 9)
        self.tanh = nn.Tanh()
        self.joints_mean = torch.tensor([0.99 / 2, 0.97 / 2, 0.8 / 2, 1.33 / 2, 0.8 / 2, 1.33 / 2, 0.98 / 2, 0.98 / 2,
                                         0.40 / 2 + 0.18])
        self.joints_range = torch.tensor([0.99, 0.97, 0.8, 1.33, 0.8, 1.33, 0.98, 0.98, 0.40])

        # svh_hand layer
        self.svh = SvhHandLayer()

    def forward(self, xyz, points):
        B = xyz.shape[0]
        device = xyz.device
        l1_xyz, l1_points = self.sa1(xyz.contiguous(), points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        feature = l3_points.view(-1, 512)
        feature = F.leaky_relu(self.bn1(self.fc1(feature)), negative_slope=0.2)
        # feature = F.leaky_relu(self.fc1(feature), negative_slope=0.2)
        feature = F.leaky_relu(self.bn2(self.fc2(feature)), negative_slope=0.2)
        # feature = F.leaky_relu(self.fc2(feature), negative_slope=0.2)

        hand_translation = self.fc_translation(feature)
        hand_quat = F.normalize(self.fc_quat(feature))
        hand_joints = self.tanh(self.fc_joints(feature))

        pose = torch.from_numpy(np.zeros([B, 4, 4])).float().to(device)
        pose[:, :3, :3] = pytorch3d.transforms.quaternion_to_matrix(hand_quat)
        pose[:, :3, 3] = hand_translation

        theta = self.joints_mean.to(device) + hand_joints * self.joints_range.to(device) * 0.5

        J = torch.cat([(theta[:, 0] - 0.26).view(B, -1),
                       (theta[:, 1] - 0.17).view(B, -1),
                       (1.01511 * theta[:, 1] + 0.35).view(B, -1),
                       (1.44889 * theta[:, 1] + 0.087).view(B, -1),
                       ((0.8 - theta[:, 2]) / 0.8 * (theta[:, 8] - 0.18)).view(B, -1),
                       (theta[:, 2] + 0.10).view(B, -1),
                       (theta[:, 3] + 0.26).view(B, -1),
                       (1.045 * theta[:, 3]).view(B, -1),
                       (theta[:, 4] + 0.10).view(B, -1),
                       (theta[:, 5] + 0.26).view(B, -1),
                       (1.0434 * theta[:, 5]).view(B, -1),
                       (-(theta[:, 8] - 0.18)).view(B, -1),
                       (theta[:, 6] + 0.23).view(B, -1),
                       (1.3588 * theta[:, 6] + 0.42).view(B, -1),
                       (1.42093 * theta[:, 6] + 0.17).view(B, -1),
                       (-2 * (theta[:, 8] - 0.18)).view(B, -1),
                       (theta[:, 7] + 0.23).view(B, -1),
                       (1.3588 * theta[:, 7] + 0.24).view(B, -1),
                       (1.42307 * theta[:, 7]).view(B, -1)], dim=1)
        vertices, normals = self.svh.get_forward_vertices(pose, theta)

        return dict(J=J, vertices=vertices, normals=normals, pose=pose, theta=theta, quat=hand_quat)
        # return dict(J=J, vertices=vertices, normals=normals, pose=pose, theta=theta, joints=joints, feature=feature,
        #             xyz=xyz)

    def get_loss(self, pred, data, debug_loss=False):
        """
        pred: {J, vertices, normals, quat}
        data: {J, contactmap, points, normals, root_mat}
        """
        bs = pred['vertices'].size()[0]

        # collision loss of hand and object
        o2h_signed, h2o_signed, _, _ = point2point_signed(pred['vertices'], data['points'], pred['normals'], data['normals'])

        o2h_dist_neg = torch.logical_and(o2h_signed.abs() < 0.015, o2h_signed < 0.0)
        h2o_dist_neg = torch.logical_and(h2o_signed.abs() < 0.015, h2o_signed < 0.0)
        loss_collision_h2o = torch.sum(h2o_signed * h2o_dist_neg) / bs
        loss_collision_o2h = torch.sum(o2h_signed * o2h_dist_neg) / bs
        if debug_loss:
            selected_idx = np.random.randint(bs)
            object_points = data['points'].cpu().numpy()[selected_idx]
            # green for object points
            object_pc = trimesh.PointCloud(object_points, colors=[0, 255, 0])

            hand_vertices = pred['vertices'].detach().cpu().numpy()[selected_idx]
            # blue for object points
            hand_pc = trimesh.PointCloud(hand_vertices, colors=[0, 0, 255])

            scene = trimesh.Scene([object_pc, hand_pc])

            object_points_in = data['points'][selected_idx][o2h_dist_neg[selected_idx]].cpu().numpy()

            if len(object_points_in) > 0:
                object_pc_in = trimesh.PointCloud(object_points_in, colors=[0, 255, 255])
                scene.add_geometry(object_pc_in)

            hand_vertices_in = pred['vertices'][selected_idx][h2o_dist_neg[selected_idx]].detach().cpu().numpy()

            if len(hand_vertices_in) > 0:
                hand_pc_in = trimesh.PointCloud(hand_vertices_in, colors=[255, 255, 0])
                scene.add_geometry(hand_pc_in)

            scene.show()

        # distant loss , from contactmap_gt(>0.4) to nearest hand vertices
        touched = data['contactmap'] > 0.4
        line2ver = [0, 16, 63, 80, 98,
                    110, 143, 160, 178,
                    190, 233, 255, 277,
                    289, 340, 364, 384,
                    393, 438, 455, 477]

        loss_dist = 0
        for i in range(20):
            if i % 4 == 0:
                continue
            idx = (data['line_idx'] == i) & touched

            if torch.sum(idx):
                o2f, f2o, _, _ = point2point_signed(pred['vertices'][:, self.only_touch_idxs[line2ver[i]:line2ver[i+1]], :], data['points'],
                                                  pred['normals'][:, self.only_touch_idxs[line2ver[i]:line2ver[i+1]], :], data['normals'])
                loss_dist += torch.sum(torch.abs(o2f)[idx]) / bs * self.config.contact_prob[i]

        # self-collision loss
        self_collison_loss = 0

        v1 = pred['vertices'][:, 1438:1857, :]
        v2 = pred['vertices'][:, 1905:2135, :]
        v3 = pred['vertices'][:, 2177:2433, :]
        v4 = pred['vertices'][:, 2471:2735, :]
        v5 = pred['vertices'][:, 2780:3000, :]
        v = [v1, v2, v3, v4, v5]

        n1 = pred['normals'][:, 1438:1857, :]
        n2 = pred['normals'][:, 1905:2135, :]
        n3 = pred['normals'][:, 2177:2433, :]
        n4 = pred['normals'][:, 2471:2735, :]
        n5 = pred['normals'][:, 2780:3000, :]
        n = [n1, n2, n3, n4, n5]

        for i in range(5):
            for j in range(i+1, 5):
                # print(i, j)
                j2i_signed, i2j_signed, _, _ = point2point_signed(v[i], v[j], n[i], n[j])
                j2i_signed_dist_neg = torch.logical_and(j2i_signed.abs() < 0.01, j2i_signed < 0.0)
                i2j_signed_dist_neg = torch.logical_and(i2j_signed.abs() < 0.01, i2j_signed < 0.0)

                if debug_loss:
                    selected_idx = np.random.randint(bs)
                    finger_1 = v[i].detach().cpu().numpy()[selected_idx]
                    finger_1_pc = trimesh.PointCloud(finger_1, colors=[0, 255, 0])
                    finger_2 = v[j].detach().cpu().numpy()[selected_idx]
                    finger_2_pc = trimesh.PointCloud(finger_2, colors=[0, 0, 255])
                    scene = trimesh.Scene([finger_1_pc, finger_2_pc])
                    points_in = v[i][selected_idx][i2j_signed_dist_neg[selected_idx]].detach().cpu().numpy()
                    print(points_in.shape)
                    if len(points_in) > 0:
                        points_in_pc = trimesh.PointCloud(points_in, colors=[0, 255, 255])
                        scene.add_geometry(points_in_pc)
                    scene.show()
                self_collison_loss += torch.sum(i2j_signed * i2j_signed_dist_neg) / bs
                self_collison_loss += torch.sum(j2i_signed * j2i_signed_dist_neg) / bs
                # print(self_collison_loss)
        # self_collison_loss *= -self.config.loss_weight[2]

        # joint loss
        joint_angle_loss = torch.mean(torch.abs(data['J'] - pred['J']))
        # print(joint_angle_loss.grad)
        # print(data['J'])
        # exit()

        # loss between contactmap_gt and contactmap_pred
        touched = o2h_signed.abs() < 0.005
        # touched_gt = data['contactmap'] > 0.4
        # not_touch = touched ^ touched_gt & touched
        repulsion_loss = torch.sum(o2h_signed * touched) / bs

        quat_gt = pytorch3d.transforms.matrix_to_quaternion(data['root_mat'])
        quat_pred = pred['quat']
        quat_loss = quaternion_loss(quat_gt, quat_pred)

        total_loss = -self.config.loss_weight[0] * (loss_collision_h2o + loss_collision_o2h) + \
                     self.config.loss_weight[1] * loss_dist + \
                     -self.config.loss_weight[2] * self_collison_loss + \
                     self.config.loss_weight[3] * joint_angle_loss + \
                     -self.config.loss_weight[4] * repulsion_loss + \
                     self.config.loss_weight[5] * quat_loss

        return dict(total_loss=total_loss, collison_loss=loss_collision_h2o+loss_collision_o2h,
                    dist_loss=loss_dist, self_collison_loss=self_collison_loss, joint_angle_loss=joint_angle_loss,
                    repulsion_loss=repulsion_loss, quat_loss=quat_loss)


if __name__ == '__main__':
    import argparse
    from util.config import cfg, cfg_from_yaml_file

    parser = argparse.ArgumentParser(description='Training Config')
    parser.add_argument('--cfg', default='config/base_config.yaml', type=str)
    args = parser.parse_args()
    config = cfg_from_yaml_file(args.cfg, cfg)
    batch_size = 2
    data1 = torch.rand(batch_size, 5000, 3).cuda()
    data2 = torch.rand(batch_size, 5000, 4).permute(0, 2, 1).cuda()
    model = backbone_pointnet2(config).cuda()
    model.eval()
    j, v, n = model(data1, data2)
    print(j.shape)
    print(v.shape)
    print(n.shape)
    # print(v)
