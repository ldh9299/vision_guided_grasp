import trimesh
import os
import numpy as np
import networkx as nx


def build_graph(mesh):
    edges = mesh.edges_unique
    length = mesh.edges_unique_length
    g = nx.Graph()
    for edge, L in zip(edges, length):
        g.add_edge(*edge, length=L)
    return g


def fps(points, npoint, mesh=None, use_geodesic=False):
    """
    Input:
        mesh: input mesh
        graph: graph for mesh
        npoint: target point number to sample
    Return:
        centroids: sampled pointcloud index, [npoint]
    """
    # print(npoint)
    if use_geodesic:
        assert mesh is not None
        graph = build_graph(mesh)
        N, C = mesh.vertices.shape
    else:
        N, C = points.shape
    centroids = np.zeros(npoint, dtype=np.int)
    distance = np.ones(N) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        # centroid = mesh.vertices[farthest, :].reshape(1, 3)
        centroid = points[farthest, :].reshape(1, 3)
        if not use_geodesic:
            # dist = np.sum((mesh.vertices - centroid) ** 2, -1)
            dist = np.sum((points - centroid) ** 2, -1)
            mask = dist < distance
            distance[mask] = dist[mask]
        else:
            dist = nx.shortest_path_length(graph, source=farthest, weight='length')
            # dist = length_geodesic
            for idx in range(0, N):
                if dist[idx] < distance[idx]:
                    distance[idx] = dist[idx]

        farthest = np.argmax(distance, -1)
    return centroids


def main():
    mesh = trimesh.load_mesh('/home/v-wewei/code/two_stage_pointnet/data/grippers/yumi/finger.obj')
    # trimesh.smoothing.filter_humphrey(mesh)
    # graph = build_graph(mesh)
    npoint = 40
    sample_point_index = fps(mesh.vertices, npoint, mesh, use_geodesic=False)

    pointcloud_sampling = trimesh.PointCloud(mesh.vertices[sample_point_index], colors=[255, 0, 0])
    mesh.visual.face_colors = [0, 255, 0]
    scene = trimesh.Scene([mesh, pointcloud_sampling])
    scene.show()


if __name__ == '__main__':
    main()