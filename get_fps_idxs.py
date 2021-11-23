import numpy as np
import trimesh
import multiprocessing as mp
import os
import open3d.open3d.io


def fps(points, npoint):
    """
    Input:
        mesh: input mesh
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


def get_fps_idx(obj_name):
    print(obj_name)
    filename = 'object_models/{}.ply'.format(obj_name)
    mesh = open3d.open3d.io.read_triangle_mesh(filename)
    mesh.compute_vertex_normals()
    open3d.visualization.draw_geometries([mesh])
    print(np.asarray(mesh.vertex_normals))
    # exit()
    v = np.asarray(mesh.vertices)
    # mesh = trimesh.load_mesh(filename)
    # print(v.shape)
    idxs = fps(v, 3000)
    # np.save('dataset/idx_3000/{}.npy'.format(obj_name), idxs)


if __name__ == "__main__":
    # filename = '/home/ldh/ContactPose/data/contactpose_data/full29_use/ps_controller/ps_controller.ply'
    # mesh = trimesh.load_mesh(filename)
    # print(mesh.vertices.shape)
    # idxs = fps(mesh.vertices, 3000)
    # print(max(idxs))
    # np.save('dataset/idx_3000/{}.npy'.format('ps_controller'), idxs)
    # exit()
    # pool = mp.Pool(processes=int(mp.cpu_count()))
    pool = mp.Pool(processes=1)

    path = "object_models"
    for _, dirname, files in os.walk(path):
        for file in files:
            obj_name = file[:-4]
            pool.apply(get_fps_idx, args=(obj_name,))
            # pool.apply_async(get_fps_idx, args=(obj_name,))

    pool.close()
    pool.join()



