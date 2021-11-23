import trimesh
import numpy as np
import networkx as nx
from utils_ import common_util
import sklearn


def build_graph(mesh):
    edges = mesh.edges_unique
    length = mesh.edges_unique_length
    g = nx.Graph()
    for edge, L in zip(edges, length):
        g.add_edge(*edge, length=L)
    return g


def fps(points, npoint, mesh, use_geodesic=False):
    """
    Input:
        mesh: input mesh
        graph: graph for mesh
        npoint: target point number to sample
    Return:
        centroids: sampled pointcloud index, [npoint]
    """
    print('sampled point num is :', npoint)
    assert use_geodesic == False
    if use_geodesic:
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


class Mesh(object):
    def __init__(self, filepath, use_embree=True, **kwargs):
        self.mesh_filepath = filepath
        self.mesh_ = trimesh.load_mesh(self.mesh_filepath)
        self.use_embree = use_embree

    @property
    def mesh(self):
        return self.mesh_

    @property
    def mesh_name(self):
        name = self.mesh_filepath.split('/')[-1].split('.')[0]
        return name

    def smoothing(self, type='humphrey', **kwargs):
        assert type in ['humphrey', 'laplacian', 'taubin']
        if type == 'humphrey':
            return trimesh.smoothing.filter_humphrey(self.mesh_, **kwargs)
        if type == 'laplacian':
            return trimesh.smoothing.filter_laplacian(self.mesh_, **kwargs)
        if type == 'taubin':
            return trimesh.smoothing.filter_taubin(self.mesh_, **kwargs)

    def ray_intersect(self, ray_origins, ray_direction, multiple_hits=True):
        if self.use_embree:
            locations, index_ray, index_tri = self.mesh_.ray.intersects_location(
                ray_origins=ray_origins,
                ray_directions=ray_direction, multiple_hits=multiple_hits)
        else:
            locations, index_ray, index_tri = self.mesh_.ray.intersects_location(
                ray_origins=ray_origins,
                ray_directions=ray_direction)
        return locations, index_ray, index_tri

    def compute_stable_poses(self, n_samples=1):
        transforms, probs = self.mesh_.compute_stable_poses(n_samples=n_samples)
        return transforms, probs

    def apply_transform(self, matrix):
        self.mesh_.apply_transform(matrix)

    def show(self):
        self.mesh_.show()

    def sample_surface(self, numpoints, vis=False):
        normals_ = self.mesh_.face_normals
        points, face_index = trimesh.sample.sample_surface(self.mesh_, numpoints)
        normals = normals_[face_index]
        result = np.concatenate((points, -normals), axis=1)

        if vis:
            pc = trimesh.PointCloud(points, colors=[255, 0, 0])
            ray_origins = points
            ray_directions = normals
            vis_path = np.hstack((ray_origins, ray_origins + ray_directions / 50)).reshape(-1, 2, 3)
            ray_visualize = trimesh.load_path(vis_path)
            scene = trimesh.Scene()
            scene.add_geometry(pc)
            scene.add_geometry(self.mesh_)
            scene.add_geometry(ray_visualize)
            scene.show()
        return result

    def normalize_pose(self):

        """
        Transforms the vertices and normals of the mesh such that the origin of the resulting mesh's coordinate frame is at the
        centroid and the principal axes are aligned with the vertical Z, Y, and X axes.

        Returns:
        Nothing. Modified the mesh in place (for now)
        """

        vertex_array_cent = np.array(self.mesh_.vertices)

        # find principal axes
        pca = sklearn.decomposition.PCA(n_components=3)
        pca.fit(vertex_array_cent)

        # count num vertices on side of origin wrt principal axes
        comp_array = pca.components_
        norm_proj = vertex_array_cent.dot(comp_array.T)
        opposite_aligned = np.sum(norm_proj < 0, axis=0)
        same_aligned = np.sum(norm_proj >= 0, axis=0)

        # create rotation from principal axes to standard basis
        z_axis = comp_array[0, :]
        y_axis = comp_array[1, :]
        if opposite_aligned[2] > same_aligned[2]:
            z_axis = -z_axis
        if opposite_aligned[1] > same_aligned[1]:
            y_axis = -y_axis
        x_axis = np.cross(y_axis, z_axis)
        R_pc_obj = np.c_[x_axis, y_axis, z_axis]

        # rotate vertices, normals and reassign to the mesh
        vertex_array_rot = R_pc_obj.T.dot(vertex_array_cent.T)
        vertex_array_rot = vertex_array_rot.T
        self.mesh_.vertices = vertex_array_rot
        return self.mesh_

    def fps_sample_surface(self, numpoints, num_scale=8, vis=False):
        assert num_scale >= 1
        # set random seed
        common_util.set_seed(seed=1)  # seed=0
        points_tmp, tri_index_tmp = trimesh.sample.sample_surface(self.mesh_, numpoints*num_scale)
        sampled_points_index = fps(points_tmp, numpoints, self.mesh_, use_geodesic=False)

        normals = self.mesh_.face_normals[tri_index_tmp][sampled_points_index]
        points = points_tmp[sampled_points_index]
        result = np.concatenate((points, -normals), axis=1)
        if vis:
            pc = trimesh.PointCloud(points, colors=[255, 0, 0])
            ray_origins = points
            ray_directions = normals
            vis_path = np.hstack((ray_origins, ray_origins + ray_directions / 40)).reshape(-1, 2, 3)
            ray_visualize = trimesh.load_path(vis_path)
            scene = trimesh.Scene()
            scene.add_geometry(pc)
            scene.add_geometry(self.mesh_)
            scene.add_geometry(ray_visualize)
            scene.show()
        return result


if __name__ == "__main__":
    mesh_file_path = '/home/v-wewei/hand/BHAM_split_stl_new/D_105_full_vhacd/D_105_full_smooth.stl'

    mesh = Mesh(filepath=mesh_file_path)
    # mesh.smoothing(type='humphrey')
    # mesh.show()
    # exit()
    # compute stable poses
    # transforms, probs = mesh.compute_stable_poses(n_samples=100)
    # print(transforms)
    # print(probs)
    #
    # for transform, prob in zip(transforms, probs):
    #     if prob > 0.1:
    #         mesh.apply_transform(transform)
    #         mesh.show()

    result = mesh.fps_sample_surface(numpoints=250, vis=True)
