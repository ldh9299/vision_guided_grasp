import os
import sys

import pytorch3d.transforms
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
from pointnet2.pointnet2.pointnet2_modules import PointnetSAModule
from svh_kinematics.svh_layer.svhhand_layer import SvhHandLayer
from tool.train_tool import point2point_signed


class backbone_pointnet2(nn.Module):
    def __init__(self, config, device='cuda'):
        super(backbone_pointnet2, self).__init__()
        self.config = config
        self.device = device

        self.sa1 = PointnetSAModule(mlp=[6, 32, 32, 64], npoint=1024, radius=0.1, nsample=32, bn=not True)
        self.sa2 = PointnetSAModule(mlp=[64, 64, 64, 128], npoint=256, radius=0.2, nsample=64, bn=not True)
        self.sa3 = PointnetSAModule(mlp=[128, 128, 128, 256], npoint=64, radius=0.4, nsample=128, bn=not True)
        self.sa4 = PointnetSAModule(mlp=[256, 256, 256, 512], npoint=None, radius=None, nsample=None, bn=not True)

        # fc layer
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 16)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.drop2 = nn.Dropout(0.5)
        self.tanh = nn.Tanh()
        self.joints_mean = torch.tensor([0, 0, 0, 0, 0, 0, 0,
                                         0.99 / 2, 0.97 / 2, 0.8 / 2, 1.33 / 2, 0.8 / 2, 1.33 / 2, 0.98 / 2, 0.98 / 2,
                                         0.40 / 2 + 0.18]).to(self.device)
        self.joints_range = torch.tensor([0.5, 0.5, 0.5, 2, 2, 2, 2,
                                          0.99, 0.97, 0.8, 1.33, 0.8, 1.33, 0.98, 0.98, 0.40]).to(self.device)

        # svh_hand layer
        self.svh = SvhHandLayer()

    def forward(self, xyz, points):
        B = xyz.shape[0]
        l1_xyz, l1_points = self.sa1(xyz.contiguous(), points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        _, l4_points = self.sa4(l3_xyz, l3_points)
        feature = l4_points.view(-1, 512)
        # feature = F.leaky_relu(self.bn1(self.fc1(feature)), negative_slope=0.2)
        feature_ = F.leaky_relu(self.fc1(feature), negative_slope=0.2)
        # feature = self.drop1(feature)
        # feature = F.leaky_relu(self.bn2(self.fc2(feature)), negative_slope=0.2)
        feature_ = F.leaky_relu(self.fc2(feature_), negative_slope=0.2)
        # feature = self.drop2(feature)
        joints_debug = self.tanh(self.fc3(feature_))
        joints = self.joints_mean + joints_debug * self.joints_range * 0.5
        # print('joints', joints)
        pose = torch.from_numpy(np.zeros([B, 4, 4])).float().to(self.device)
        pose[:, :3, :3] = pytorch3d.transforms.quaternion_to_matrix(joints[:, 3:7])
        pose[:, :3, 3] = joints[:, :3]
        # pose[:, 2, 3] -= 0.1
        theta = joints[:, 7:]
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
        return dict(J=J, vertices=vertices, normals=normals, pose=pose, theta=theta, joints=joints, feature=feature, xyz=xyz)

    def get_loss(self, pred, data):
        """
        pred: {J, vertices, normals, quat}
        data: {J, contactmap, points, normals, root_mat}
        """
        # collision loss of hand and object
        o2h, h2o, _, _ = point2point_signed(pred['vertices'], data['points'], pred['normals'], data['normals'])

        w_dist_neg = o2h < 0.0
        v_dist_neg = torch.logical_and(h2o.abs() < 0.015, h2o < 0.0)
        loss_collision_h2o = torch.sum(h2o * v_dist_neg)
        loss_collision_o2h = torch.sum(o2h * w_dist_neg)

        # distant loss , from contactmap_gt(>0.4) to nearest hand vertices
        touched = data['contactmap'] > 0.4
        line2ver = [1745, 1834, 2004, 2098,
                    2156, 2186, 2273, 2323,
                    2359, 2386, 2481, 2538,
                    2573, 2602, 2701, 2758,
                    2796, 2826, 2925, 2965, 3000]
        # o2f, _, _, _ = point2point_signed(pred['vertices'][:, 1834:, :], data['points'], pred['normals'][:, 1834:, :], data['normals'])
        # loss_dist = torch.sum(torch.abs(o2f)[touched]) / torch.sum(touched)
        loss_dist = 0
        for i in range(20):
            idx = (data['line_idx'] == i) & touched
            if torch.sum(idx):
                o2f, _, _, _ = point2point_signed(pred['vertices'][:, line2ver[i]:line2ver[i+1], :], data['points'],
                                                  pred['normals'][:, line2ver[i]:line2ver[i+1], :], data['normals'])
                loss_dist += torch.sum(torch.abs(o2f)[idx]) / torch.sum(idx) * self.config.contact_prob[i]

        # self-collision loss
        self_collison_loss = 0

        v1 = pred['vertices'][:, 1834:2156, :]
        v2 = pred['vertices'][:, 2186:2359, :]
        v3 = pred['vertices'][:, 2386:2573, :]
        v4 = pred['vertices'][:, 2602:2796, :]
        v5 = pred['vertices'][:, 2826:3000, :]
        v = [v1, v2, v3, v4, v5]

        n1 = pred['normals'][:, 1834:2156, :]
        n2 = pred['normals'][:, 2186:2359, :]
        n3 = pred['normals'][:, 2386:2573, :]
        n4 = pred['normals'][:, 2602:2796, :]
        n5 = pred['normals'][:, 2826:3000, :]
        n = [n1, n2, n3, n4, n5]

        for i in range(5):
            for j in range(5):
                if i == j:
                    continue
                _, i2j, _, _ = point2point_signed(v[i], v[j], n[i], n[j])
                dist_neg = i2j < 0.0
                self_collison_loss += torch.sum(i2j * dist_neg)
        # self_collison_loss *= -self.config.loss_weight[2]

        # joint loss
        joint_angle_loss = torch.mean(torch.abs(data['J'] - pred['J']))
        # print(joint_angle_loss.grad)
        # print(data['J'])
        # exit()

        # loss between contactmap_gt and contactmap_pred
        touched = o2h < 0.005
        touched_gt = data['contactmap'] > 0.4
        not_touch = touched ^ touched_gt & touched
        diff_map_loss = torch.sum(o2h * not_touch)

        # quat_loss
        def quaternion_loss(y_true, y_pred):
            dist1 = torch.mean(torch.abs(y_true - y_pred), dim=-1)
            dist2 = torch.mean(torch.abs(y_true + y_pred), dim=-1)
            loss = torch.where(dist1 < dist2, dist1, dist2)
            return torch.mean(loss)

        quat_gt = pytorch3d.transforms.matrix_to_quaternion(data['root_mat'])
        # print('quat_gt', quat_gt)
        quat_pred = pred['joints'][:, 3:7]
        quat_pred = quat_pred / torch.linalg.norm(quat_pred)
        # print('quat_pred', quat_pred)
        quat_loss =  quaternion_loss(quat_gt, quat_pred)

        total_loss = -self.config.loss_weight[0] * (loss_collision_h2o + loss_collision_o2h) + \
                     self.config.loss_weight[1] * loss_dist + \
                     -self.config.loss_weight[2] * self_collison_loss + \
                     self.config.loss_weight[3] * joint_angle_loss + \
                     -self.config.loss_weight[4] * diff_map_loss + \
                     self.config.loss_weight[5] * quat_loss

        return dict(total_loss=total_loss, collison_loss=loss_collision_h2o+loss_collision_o2h,
                    dist_loss=loss_dist, self_collison_loss=self_collison_loss, joint_angle_loss=joint_angle_loss,
                    diff_map_loss=diff_map_loss, quat_loss=quat_loss)


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
