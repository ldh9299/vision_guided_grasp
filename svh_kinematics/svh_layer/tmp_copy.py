import trimesh
import numpy as np
import os


mesh_dir = '../meshes/svh_hand_ori'
for root, dirs, files in os.walk(mesh_dir):
    for filename in files:
        filepath = os.path.join(root, filename)
        if filename.endswith('.stl'):
            mesh = trimesh.load(filepath)
            mesh.export(filepath.replace('.stl', '.obj'))

exit()


# mesh = trimesh.load_mesh('../meshes/svh_hand/d11.stl')
# mesh.apply_translation(np.array([0.032, 0, 0.0]))
# mesh.show()
# mesh.export('../meshes/svh_hand/d11.stl')
#
# exit()

def euler_to_rotMat(yaw, pitch, roll):
    Rz_yaw = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [          0,            0, 1]])
    Ry_pitch = np.array([
        [ np.cos(pitch), 0, np.sin(pitch)],
        [             0, 1,             0],
        [-np.sin(pitch), 0, np.cos(pitch)]])
    Rx_roll = np.array([
        [1,            0,             0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll),  np.cos(roll)]])
    # R = RzRyRx
    rotMat = np.dot(Rz_yaw, np.dot(Ry_pitch, Rx_roll))
    return rotMat

# # 0 -1.5707 1.5707
# roll = 0
# pitch = 0.9704
# yaw = 0
# rotMat = euler_to_rotMat(yaw, pitch, roll)
# print(rotMat)
