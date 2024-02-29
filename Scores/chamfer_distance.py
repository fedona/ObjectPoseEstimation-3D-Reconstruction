# Compute the distance between two pointclouds

import numpy as np
import open3d as o3d
from simpleicp import PointCloud, SimpleICP

def read_obj(file_path):
    """Reads an OBJ file and returns the vertices as a NumPy array."""
    vertices = []

    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                vertex = list(map(float, line.split()[1:]))
                vertices.append(vertex)

    return np.array(vertices)

def obj_to_xyz1(input_obj_file, output_xyz_file):
    # Read the OBJ file, converts it to a .xyz format 
    point_cloud = o3d.io.read_point_cloud(input_obj_file)
    vertices = np.asarray(point_cloud.points)

    print(np.asarray(vertices))

    with open(output_xyz_file, 'w') as xyz_file:
        for vertex in vertices:
            xyz_file.write(f"{vertex[0]} {vertex[1]} {vertex[2]}\n")

def obj_to_xyz(input_obj_file, output_xyz_file):
    vertices = []

    with open(input_obj_file, 'r') as f:
        for line in f:
            if line.startswith('v '):
                # Extracting vertex coordinates
                vertex = list(map(float, line.split()[1:]))
                vertices.append(vertex)

    vertices_array = np.array(vertices)

    with open(output_xyz_file, 'w') as xyz_file:
        for vertex in vertices_array:
            xyz_file.write(f"{vertex[0]} {vertex[1]} {vertex[2]}\n")


def transform_points(vertices, transformation_matrix):
    # Add a column of ones to the vertices to make them homogeneous coordinates
    homogeneous_vertices = np.hstack((vertices, np.ones((vertices.shape[0], 1))))

    # Apply the transformation matrix
    transformed_vertices_homogeneous = np.dot(transformation_matrix, homogeneous_vertices.T).T

    # Remove the added column of ones and return the result
    transformed_vertices = transformed_vertices_homogeneous[:, :3]

    return transformed_vertices

def compute_chamfer_distance(pts1, pts2):

    count = 0
    tot = len(pts1) + len(pts2)

    # Translate points to center them
    translated_pts1 = pts1 
    translated_pts2 = pts2 
    print('')

    # Compute chamfer distance 1 to 2
    distances1 = []
    for point in translated_pts1:
        count += 1
        distances1.append(np.min(np.linalg.norm(translated_pts2 - point, axis=1)))
        print(f'Computing distance 1 to 2 {round(count/tot, 3)*100}%', end="\r")

    # Compute chamfer distance 2 to 1
    distances2 = []
    for point in translated_pts2:
        count += 1
        distances2.append(np.min(np.linalg.norm(translated_pts1 - point, axis=1)))
        print(f'Computing distance 2 to 1 {round(count/tot, 3)*100}%', end="\r")

    d1to2 = np.mean(distances1)
    d2to1 = np.mean(distances2)

    print(f'Mean distance first obj to second obj: {d1to2}')
    print(f'Mean distance second obj to first obj: {d2to1}')

    chamfer_distance = (d1to2 + d2to1) / 2
    chamfer_distance_per_point = (d1to2 / len(translated_pts1) + d2to1 / len(translated_pts2))

    return chamfer_distance, chamfer_distance_per_point

# Example
obj1 = 'Mug_Teddy_Andrea.obj'
obj2 = 'teddyMugBundleSDF.obj'

xyz1 = obj_to_xyz(obj1, 'file1.xyz')
xyz2 = obj_to_xyz(obj2, 'file2.xyz')

X_fix = np.genfromtxt('file1.xyz')
X_mov = np.genfromtxt('file2.xyz')

print(X_fix)
print(X_mov)

# Create the point cloud objects
pc_fix = PointCloud(X_fix, columns=["x", "y", "z"])
pc_mov = PointCloud(X_mov, columns=["x", "y", "z"])

# Compute transformation matrix through ICP method
icp = SimpleICP()
icp.add_point_clouds(pc_fix, pc_mov)
H, X_mov_transformed, rigid_body_transformation_params, distance_residuals = icp.run(max_overlap_distance=1)

# Compute chamfer distance from the transformed coordinates of second object
chamfer_distance, chamfer_distance_per_point = compute_chamfer_distance(X_fix, X_mov_transformed)
print(f"Chamfer Distance: {chamfer_distance}")
print(f"Chamfer Distance per point: {chamfer_distance_per_point}")
