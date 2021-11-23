import numpy as np

from model import backbone_pointnet2
from dataset import Dataset
import torch
import trimesh
import open3d as o3d
from svh_kinematics.svh_layer.svhhand_layer import SvhHandLayer

# torch seed
torch.manual_seed(0)
svh = SvhHandLayer(show_mesh=True)


def train(dataloader, model):
    model.train()
    num_epoches = 200
    optimizer = torch.optim.Adam(params=model.parameters(), lr=3e-4)
    for epoch in range(num_epoches):
        for i, data in enumerate(dataloader):

            for k in data.keys():
                data[k] = data[k].cuda().float()
            optimizer.zero_grad()
            pred = model(data['points'],
                         torch.cat([data['normals'],
                                    data['contactmap'].unsqueeze(dim=-1),
                                    data['finger_idx'].unsqueeze(dim=-1),
                                    data['part_idx'].unsqueeze(dim=-1)], dim=-1).transpose(1, 2))

            loss_dict = model.get_loss(pred, data)
            loss = loss_dict['total_loss']
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0 and i % 1 == 0:
                for k in loss_dict.keys():
                    loss_dict[k] = loss_dict[k].detach().cpu().numpy().item()
                print(epoch, i, loss_dict)
                # print(pred['joints_debug'].detach().cpu().numpy())
                vis = not True
                if vis:
                    # touched_gt = data['contactmap'] > 0.4
                    pc_o = trimesh.PointCloud(data['points'][0].cpu().detach().numpy(), colors=[255, 0, 0])
                    # pc_o_t = trimesh.PointCloud(data['points'][touched_gt].squeeze().cpu().detach().numpy(), colors=[0, 0, 255])
                    pc_h = trimesh.PointCloud(pred['vertices'][0].cpu().detach().numpy(), colors=[0, 255, 0])
                    scene = trimesh.Scene()
                    scene.add_geometry([pc_h, pc_o])
                    scene.show()
                    # pc_o = o3d.geometry.PointCloud()
                    # pc_o.points = o3d.utility.Vector3dVector(data['points'][0].cpu().detach().numpy())
                    # pc_o.paint_uniform_color([1, 0, 0])
                    # pc_o_t = o3d.geometry.PointCloud()
                    # pc_o_t.points = o3d.utility.Vector3dVector(data['points'][touched_gt][0].cpu().detach().numpy())
                    # pc_o_t.paint_uniform_color([0, 0, 1])
                    # pc_o_t_p = o3d.geometry.PointCloud()
                    # pc_o_t_p.points = o3d.utility.Vector3dVector(
                    #     data['points'][t_p][0].cpu().detach().numpy())
                    # pc_o_t_p.paint_uniform_color([1, 1, 0])
                    # pc_h = o3d.geometry.PointCloud()
                    # pc_h.points = o3d.utility.Vector3dVector(pred['vertices'][0].cpu().detach().numpy())
                    # pc_h.paint_uniform_color([0, 1, 0])
                    # o3d.visualization.draw_geometries([pc_h, pc_o])

    # torch.save(model.state_dict(), 'checkpoint/{}_{}.pth'.format('model_2_part', num_epoches))


def test(dataloader, model, vis=False):
    model.eval()
    for i, data in enumerate(dataloader):
        index = data['index']
        for k in data.keys():
            data[k] = data[k].cuda().float()
        pred = model(data['points'],
                     torch.cat([data['normals'],
                                data['contactmap'].unsqueeze(dim=-1),
                                data['finger_idx'].unsqueeze(dim=-1),
                                data['part_idx'].unsqueeze(dim=-1)], dim=-1).transpose(1, 2))
        loss = model.get_loss(pred, data)
        for k in loss.keys():
            loss[k] = loss[k].detach().cpu().numpy().item()
        print(loss)
        # print(pred['xyz'].detach().cpu().numpy())
        if vis:
            # pc_o = trimesh.PointCloud(data['points'].squeeze().cpu().detach().numpy(), colors=[255, 0, 0])
            touched_gt = data['contactmap'] > 0.4
            # pc_o_t = trimesh.PointCloud(data['points'][touched_gt].squeeze().cpu().detach().numpy(), colors=[0, 0, 255])
            # pc_h = trimesh.PointCloud(pred['vertices'].squeeze().cpu().detach().numpy(), colors=[0, 255, 0])
            # scene = trimesh.Scene()
            # scene.add_geometry([pc_h, pc_o, pc_o_t])
            # scene.show()
            pc_o = o3d.geometry.PointCloud()
            pc_o.points = o3d.utility.Vector3dVector(data['points'][0].cpu().detach().numpy())
            pc_o.paint_uniform_color([1, 0, 0])
            pc_o_t = o3d.geometry.PointCloud()
            pc_o_t.points = o3d.utility.Vector3dVector(data['points'][touched_gt].cpu().detach().numpy())
            pc_o_t.paint_uniform_color([0, 0, 1])
            pc_h = o3d.geometry.PointCloud()
            pc_h.points = o3d.utility.Vector3dVector(pred['vertices'][0].cpu().detach().numpy())
            pc_h.paint_uniform_color([0, 1, 0])
            o3d.visualization.draw_geometries([pc_h, pc_o, pc_o_t])
            # hand_mesh = svh.get_forward_hand_mesh(pred['pose'], pred['theta'])
            # contactmap_filename = '/home/ldh/ContactPose/data/contactpose_data/full{}_use/{}/{}.ply'.format(
            #     dataset.data[index]['human_id'], dataset.data[index]['object_name'], dataset.data[index]['object_name'])
            # obj_mesh = trimesh.load_mesh(contactmap_filename)
            # scene.add_geometry([hand_mesh, obj_mesh])
            # scene.show()
            # exit()


if __name__ == '__main__':
    import argparse
    from util.config import cfg, cfg_from_yaml_file

    parser = argparse.ArgumentParser(description='Training Config')
    parser.add_argument('--cfg', default='config/config_2.yaml', type=str)
    args = parser.parse_args()
    config = cfg_from_yaml_file(args.cfg, cfg)

    dataset = Dataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False)
    test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    model = backbone_pointnet2(config).cuda()

    train(dataloader, model)
    # model.load_state_dict(torch.load('checkpoint/model_2_part_500.pth'))
    test(test_dataloader, model, vis=True)
