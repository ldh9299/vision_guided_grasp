import os
import copy
import pickle
from utils_ import mesh_util
import numpy as np
import torch
import trimesh
from hitdlr_kinematics.hitdlr_layer.hitdlr_layer import HitdlrLayer
import time
# from utils_.taxonomy_20dof import grasp_dict_20f


vis = True
mesh_prefix_dir = '/home/v-wewei/code/isaacgym/assets/mjcf/BHAM_split_stl/'
device = 'cpu'
mesh_save_path = './output_mesh_tmp'
if vis:
    hit_hand = HitdlrLayer(device)
R_hand = np.load('./R_hand.npy')

# file_path = '/home/v-wewei/dlr_grasping/'
file_path = '../../pickle_256'
for root, dirs, files in os.walk(file_path):
    for filename in files:
        if filename.endswith('_final.pickle'):
            if 'Parallel_Extension' not in filename:
                continue
            grasp_configuration = []
            filepath = os.path.join(root, filename)
            print(filepath)
            if not filename.startswith('D_1_full'):
                continue
            with open(filepath, 'rb') as f:
                grasp_dicts = pickle.load(f)
                tmp = filepath.split('/')[-1].split('_')[:-3]
                mesh_filepath = mesh_prefix_dir + '_'.join(tmp) +'_vhacd/' + '_'.join(tmp) +'_smooth.stl'
                assert os.path.exists(mesh_filepath)
                # print(mesh_filepath)
                if vis:
                    obj_mesh = mesh_util.Mesh(filepath=mesh_filepath)
                for grasp_dict in grasp_dicts:
                    if not grasp_dict:
                        continue

                    time_start = time.time()
                    joint_configuration = grasp_dict['joint_configuration']
                    pose = np.hstack((grasp_dict['pos'], grasp_dict['quat']))
                    if vis:
                        mesh_copy = copy.deepcopy(obj_mesh)

                    R = trimesh.transformations.quaternion_matrix([pose[3], pose[4], pose[5], pose[6]])
                    t = trimesh.transformations.translation_matrix([pose[0], pose[1], pose[2]])
                    R_obj = trimesh.transformations.concatenate_matrices(t, R)

                    # T_transform_2 = trimesh.transformations.euler_matrix(-0.32, 0, 0, 'rxyz')
                    # T_transform_3 = trimesh.transformations.translation_matrix([-0.015, 0.1, 0.01])
                    # T_transform_4 = trimesh.transformations.quaternion_matrix([0, 1, 0, 0])
                    # T_transform_1 = trimesh.transformations.euler_matrix(np.pi/2, 0, np.pi, 'rxyz')
                    # T_transform_5 = trimesh.transformations.quaternion_matrix([0, 1, 0, 0])
                    # R_hand = trimesh.transformations.concatenate_matrices(T_transform_5,
                    #                                                       T_transform_1, T_transform_4, T_transform_3,
                    #                                                       T_transform_2)

                    inv_R_obj = trimesh.transformations.inverse_matrix(R_obj)
                    hand_in_obj = trimesh.transformations.concatenate_matrices(inv_R_obj, R_hand)

                    translation = copy.deepcopy(hand_in_obj[:3, 3])
                    quat = trimesh.transformations.quaternion_from_matrix(hand_in_obj)
                    print('time cost is :', time.time() - time_start)

                    if vis:

                        T = trimesh.transformations.quaternion_matrix(quat)
                        translation_matrix = trimesh.transformations.translation_matrix(translation)
                        T_ = trimesh.transformations.concatenate_matrices(translation_matrix, T)
                        theta_ = joint_configuration
                        theta_tensor = torch.from_numpy(theta_).to(device).reshape(-1, 20)

                        pose_tensor = torch.from_numpy(T_).to(device).reshape(-1, 4, 4).float()
                        if not os.path.exists(mesh_save_path):
                            os.mkdir(mesh_save_path)
                        hand_meshes = hit_hand.get_forward_hand_mesh(pose_tensor, theta_tensor, save_mesh=False, path=mesh_save_path)
                        # hand_mesh = np.sum(hand_mesh)
                        # hand_mesh.apply_transform(T)
                        # hand_mesh.apply_translation(translation)
                        (hand_meshes[0] + mesh_copy.mesh).show()
                    configuration = np.hstack((translation, quat, joint_configuration))
                    grasp_configuration.append(configuration)
            grasp_configuration_array = np.array(grasp_configuration)
            filename_prefix = filename.split('_')[:-1]
            filename = '_'.join(filename_prefix)
            if not os.path.exists('../grasp_dataset'):
                os.mkdir('../grasp_dataset')
            np.save('../grasp_dataset/{}.npy'.format(filename), grasp_configuration_array)
            exit()
            # print(grasp_configuration_array.shape)
            # exit()
