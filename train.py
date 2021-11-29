import numpy as np
from model import backbone_pointnet2
from dataset import Contact_dataset
import torch
import trimesh
import open3d as o3d
from svh_kinematics.svh_layer.svhhand_layer import SvhHandLayer

# torch seed
torch.manual_seed(0)
svh = SvhHandLayer(show_mesh=True)


def train(dataloader, model):
    model.train()

    for i, data in enumerate(dataloader):
        index = data['index']

        for k in data.keys():
            data[k] = data[k].cuda().float()
        optimizer.zero_grad()
        pred = model(data['points'],
                     torch.cat([data['normals'],
                                data['contactmap'].unsqueeze(dim=-1),
                                data['finger_idx'].unsqueeze(dim=-1),
                                data['part_idx'].unsqueeze(dim=-1)], dim=-1).transpose(1, 2))

        loss_dict = model.get_loss(pred, data, debug_loss=False)
        loss = loss_dict['total_loss']
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 and i % 1 == 0:
            for k in loss_dict.keys():
                loss_dict[k] = loss_dict[k].detach().cpu().numpy().item()
            print(epoch, i, loss_dict)
            # print(pred['joints_debug'].detach().cpu().numpy())

        vis = False
        if vis:
            hand_mesh = svh.get_forward_hand_mesh(pred['pose'], pred['theta'])[0]
            object_filepath = 'dataset/object_models/{}.ply'.format(dataset.data[index[0]]['object_name'])
            object_mesh = trimesh.load_mesh(object_filepath)
            scene = trimesh.Scene([hand_mesh, object_mesh])
            scene.show()

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

        if vis:
            # touched_gt = data['contactmap'] > 0.4
            contact_map = dataset.data[index]['contact_map'].reshape(-1, 1)

            contact_map = contact_map.repeat(3, axis=1)

            hand_mesh = svh.get_forward_hand_mesh(pred['pose'], pred['theta'])[0]
            hand_mesh.export('./hand_mesh.stl')
            hand_mesh = o3d.io.read_triangle_mesh('./hand_mesh.stl')
            hand_mesh.compute_vertex_normals()
            object_filepath = 'dataset/object_models/{}.ply'.format(dataset.data[index]['object_name'])
            object_mesh = o3d.io.read_triangle_mesh(object_filepath)
            object_mesh.compute_vertex_normals()
            object_mesh.vertex_colors = o3d.utility.Vector3dVector(contact_map)
            o3d.visualization.draw_geometries([object_mesh, hand_mesh])

            # object_mesh = trimesh.load_mesh(object_filepath)
            # object_mesh.visual.vertex_colors =
            # scene = trimesh.Scene([hand_mesh, object_mesh])
            # scene.show()


if __name__ == '__main__':
    import argparse
    from util.config import cfg, cfg_from_yaml_file

    parser = argparse.ArgumentParser(description='Training Config')
    parser.add_argument('--cfg', default='config/config_2.yaml', type=str)
    args = parser.parse_args()
    config = cfg_from_yaml_file(args.cfg, cfg)

    dataset = Contact_dataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=16, drop_last=False,
                                             persistent_workers=True)
    test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    model = backbone_pointnet2(config).cuda()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1.5e-5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.milestones, gamma=config.gamma)

    num_epochs = config.num_epochs

    for epoch in range(num_epochs):
        train(dataloader, model)
        scheduler.step()
    # model.load_state_dict(torch.load('checkpoint/model_2_part_500.pth'))
    test(test_dataloader, model, vis=True)
