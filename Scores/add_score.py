import argparse
import numpy as np
import json
from scipy.spatial.transform import Rotation as R
import plyfile
import misc
from scipy import spatial
import open3d as o3d
import trimesh

import plotly.graph_objects as go

"""
    Short code to compute ADD score of predicted poses vs gt poses.
    It requires any object to be used as model to be projected in poses
    and to compute differences. In this case the default model is taken from the 
    LM dataset.

"""

#python add_score.py --gt_pose_file "/local/home/fedona/Desktop/ETH/Bachelor/Datasets/tudl_test_all/test/000003/scene_gt.json" --pred_pose_file "../Downloads/poses.json"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--object_name', type=str, default='000001')
    parser.add_argument('--gt_pose_file')
    parser.add_argument('--pred_pose_file')
    #parser.add_argument('--prediction_file', type=str, default='scene_prediction4.json')
    args = parser.parse_args()
    return args


def read_3d_points(object_name):
    filename = 'lm_models/models/obj_{}.ply'.format(object_name)
    with open(filename) as f:
        in_vertex_list = False
        vertices = []
        in_mm = False
        for line in f:
            if in_vertex_list:
                vertex = line.split()[:3]
                vertex = np.array([float(vertex[0]),
                                   float(vertex[1]),
                                   float(vertex[2])], dtype=np.float32)
                if in_mm:
                    vertex = vertex / np.float32(10) # mm -> cm
                #vertex = vertex / np.float32(100)           # ?perche?
                vertices.append(vertex)
                if len(vertices) >= vertex_count:
                    break
            elif line.startswith('element vertex'):
                vertex_count = int(line.split()[-1])
            elif line.startswith('end_header'):
                in_vertex_list = True
            elif line.startswith('element face'):
                in_mm = True
    return np.matrix(vertices)

def read_3d_points_tri(object_name):
    file_path = 'lm_models/models/obj_{}.ply'.format(object_name)
    #file_path = '/local/home/fedona/Desktop/ETH/Bachelor/obj_{}.ply'.format(object_name)
    mesh = trimesh.load_mesh(file_path)

    # Extract vertices
    vertices = np.array(mesh.vertices)

    return vertices

def read_3d_points_tri_default():
    file_path = 'tudl_models/models_eval/obj_000001.ply'
    #file_path = 'tless_models/models/obj_000014.ply'
    mesh = trimesh.load_mesh(file_path)

    # Extract vertices
    vertices = np.array(mesh.vertices)

    return vertices

def read_3d_points_ply(object_name):
    filename = 'lm_models/models/obj_{}.ply'.format(object_name)

    # Read the PLY file
    ply_data = plyfile.PlyData.read(filename)

    # Extract vertices
    vertices = np.vstack([ply_data['vertex']['x'], ply_data['vertex']['y'], ply_data['vertex']['z']]).T

    # Convert units if needed (e.g., from mm to cm)
    vertices = vertices / 10.0  # assuming mm to cm conversion

    return vertices.astype(np.float32)

def read_diameter(object_name):
    # this is the same for linemod and occlusion linemod
    filename = './tudl_models/models/models_info.json'
    
    with open(filename, 'r') as f:
        data = json.load(f)

    # Remove leading zeros and convert to integer
    object_id = int(object_name.lstrip('0'))

    # Access the diameter field from the JSON data
    diameter_in_mm = data.get(str(object_id), {}).get('diameter', None)

    if diameter_in_mm is not None:
        return diameter_in_mm #* 0.1
    else:
        print(f"Diameter not found for object '{object_name}' in the JSON file.")
        return None

def read_gt(frame_id, object_name):
    filename = object_name

    with open(filename, 'r') as f:
        data = json.load(f)

    # Check if data is a dictionary and the frame_id is present
    if not isinstance(data, dict) or str(frame_id) not in data:
        print(f"Frame {frame_id} not found for object '{object_name}' in the JSON file.")
        return None

    # Extract the relevant information from the frame entry
    frame_entry = data[str(frame_id)][0]  #
    R_gt = frame_entry.get('cam_R_m2c', None)
    t_gt = frame_entry.get('cam_t_m2c', None)

    if R_gt is None:
        print(f"R matrix not found for object '{object_name}' in the JSON file at frame id {frame_id}.")
        return None

    return np.array(R_gt), np.array(t_gt)

def read_pred(frame_id, object_name):
    filename = object_name

    with open(filename, 'r') as f:
        data = json.load(f)

    # Check if data is a dictionary and the frame_id is present
    if not isinstance(data, dict) or str(frame_id) not in data:
        print(f"Frame {frame_id} not found for object '{object_name}' in the JSON file.")
        return None

    # Extract the relevant information from the frame entry
    frame_entry = data[str(frame_id)][0]  #
    R_pred = frame_entry.get('cam_R_m2c', None)
    t_pred = frame_entry.get('cam_t_m2c', None)

    if R_pred is None:
        print(f"R matrix not found for object '{object_name}' in the JSON file at frame id {frame_id}.")
        return None

    return np.array(R_pred), np.array(t_pred)

def extract_frame_ids(object_name):

    json_file_path = object_name

    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)

        # Extract frame IDs from the keys of the dictionary
        frame_ids = [int(frame_id) for frame_id in data.keys()]

        return frame_ids

    except FileNotFoundError:
        print(f"File '{json_file_path}' not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON in file '{json_file_path}'.")
        return None
    
def compute_add_score(pts3d, gt_pose, pred_pose, R_rel, t_rel):
    """
    Compute the ADD score between two poses.

    Parameters:
    - pts3d: Numpy matrix of 3D points of model object. nx3
    - gt_pose: Tuple containing ground truth pose (R_gt, t_gt).
    - pred_pose: Tuple containing predicted pose (R_pred, t_pred).

    Returns:
    - mean_distance
    """
    R_gt, t_gt = gt_pose        # 3x3, 3x1
    R_est, t_est = pred_pose

    t_est = t_rel + t_est
    R_est = R.from_matrix(R_est).as_matrix() @ R_rel

    # Transform 3D points to camera coordinate system
    pts3d_camera_gt = R_gt @ (pts3d.T) + t_gt

    # Transform 3D points to camera coordinate system
    pts3d_camera_pred = R_est @ (pts3d.T) + t_est

    # Compute distances between corresponding points
    distance = np.linalg.norm(pts3d_camera_gt - pts3d_camera_pred, axis=0)  # compute norm among the columns (distance between each point)
    # Meanof the distances
    mean_distance = np.mean(distance)
    max_distance = np.max(distance)
    min_distance = np.min(distance)

    #print('  this is the GT        : {}'.format(R_gt))
    #print('  this is the prediction: {}'.format(R_pred))
    print('  distance between points: {}'.format(distance))
    print('  max distance between some points: {}'.format(max_distance))
    print('  min distance between some points: {}'.format(min_distance))

    return mean_distance

def compute_add_score_norel(pts3d, gt_pose, pred_pose):
    """
    Compute the ADD score between two poses.

    Parameters:
    - pts3d: Numpy matrix of 3D points of model object. nx3
    - gt_pose: Tuple containing ground truth pose (R_gt, t_gt).
    - pred_pose: Tuple containing predicted pose (R_pred, t_pred).

    Returns:
    - mean_distance
    """
    R_gt, t_gt = gt_pose        # 3x3, 3x1
    R_pred, t_pred = pred_pose

    # Transform 3D points to camera coordinate system
    pts3d_camera_gt = R_gt @ (pts3d.T) + t_gt

    # Transform 3D points to camera coordinate system
    pts3d_camera_pred = R_pred @ (pts3d.T) + t_pred

    # Compute distances between corresponding points
    distance = np.linalg.norm(pts3d_camera_gt - pts3d_camera_pred, axis=0)  # compute norm among the columns (distance between each point)
    # Meanof the distances
    mean_distance = np.mean(distance)
    max_distance = np.max(distance)
    min_distance = np.min(distance)

    #print('  this is the GT        : {}'.format(R_gt))
    #print('  this is the prediction: {}'.format(R_pred))
    print('  distance between points: {}'.format(distance))
    print('  max distance between some points: {}'.format(max_distance))
    print('  min distance between some points: {}'.format(min_distance))

    return mean_distance

def add_norel(R_est, t_est, R_gt, t_gt, pts):
    """Average Distance of Model Points for objects with no indistinguishable
    views - by Hinterstoisser et al. (ACCV'12).

    :param R_est: 3x3 ndarray with the estimated rotation matrix.
    :param t_est: 3x1 ndarray with the estimated translation vector.
    :param R_gt: 3x3 ndarray with the ground-truth rotation matrix.
    :param t_gt: 3x1 ndarray with the ground-truth translation vector.
    :param pts: nx3 ndarray with 3D model points.
    :return: The calculated error.
    """

    pts_est = misc.transform_pts_Rt(pts, R_est, t_est)
    pts_gt = misc.transform_pts_Rt(pts, R_gt, t_gt)
    e = np.linalg.norm(pts_est - pts_gt, axis=1).mean()
    return e

# BOP TOOLKIT
def add(R_est, t_est, R_gt, t_gt, R_rel, t_rel, pts):
    """Average Distance of Model Points for objects with no indistinguishable
    views - by Hinterstoisser et al. (ACCV'12).

    :param R_est: 3x3 ndarray with the estimated rotation matrix.
    :param t_est: 3x1 ndarray with the estimated translation vector.
    :param R_gt: 3x3 ndarray with the ground-truth rotation matrix.
    :param t_gt: 3x1 ndarray with the ground-truth translation vector.
    :param pts: nx3 ndarray with 3D model points.
    :return: The calculated error.
    """

    t_est = t_rel + t_est
    R_est = R.from_matrix(R_est).as_matrix() @ R_rel

    pts_est = misc.transform_pts_Rt(pts, R_est, t_est)
    pts_gt = misc.transform_pts_Rt(pts, R_gt, t_gt)
    e = np.linalg.norm(pts_est - pts_gt, axis=1).mean()
    return e

def adi_norel(R_est, t_est, R_gt, t_gt, pts):
    """Average Distance of Model Points for objects with indistinguishable views
    - by Hinterstoisser et al. (ACCV'12).

    :param R_est: 3x3 ndarray with the estimated rotation matrix.
    :param t_est: 3x1 ndarray with the estimated translation vector.
    :param R_gt: 3x3 ndarray with the ground-truth rotation matrix.
    :param t_gt: 3x1 ndarray with the ground-truth translation vector.
    :param pts: nx3 ndarray with 3D model points.
    :return: The calculated error.
    """

    pts_est = misc.transform_pts_Rt(pts, R_est, t_est)
    pts_gt = misc.transform_pts_Rt(pts, R_gt, t_gt)

    # Calculate distances to the nearest neighbors from vertices in the
    # ground-truth pose to vertices in the estimated pose.
    nn_index = spatial.cKDTree(pts_est)
    nn_dists, _ = nn_index.query(pts_gt, k=1)

    e = nn_dists.mean()
    return e

# BOP TOOLKIT
def adi(R_est, t_est, R_gt, t_gt, R_rel, t_rel, pts):
    """Average Distance of Model Points for objects with indistinguishable views
    - by Hinterstoisser et al. (ACCV'12).

    :param R_est: 3x3 ndarray with the estimated rotation matrix.
    :param t_est: 3x1 ndarray with the estimated translation vector.
    :param R_gt: 3x3 ndarray with the ground-truth rotation matrix.
    :param t_gt: 3x1 ndarray with the ground-truth translation vector.
    :param pts: nx3 ndarray with 3D model points.
    :return: The calculated error.
    """
    ## Transform predicted pose to relative prediction pose
    t_est = t_rel + t_est
    R_est = R.from_matrix(R_est).as_matrix() @ R_rel

    pts_est = misc.transform_pts_Rt(pts, R_est, t_est)
    pts_gt = misc.transform_pts_Rt(pts, R_gt, t_gt)

    # Calculate distances to the nearest neighbors from vertices in the
    # ground-truth pose to vertices in the estimated pose.
    nn_index = spatial.cKDTree(pts_est)
    nn_dists, _ = nn_index.query(pts_gt, k=1)

    e = nn_dists.mean()
    return e

# main function
if __name__ == '__main__':
    args = parse_args()
    #record = np.load(args.prediction_file, allow_pickle=True).item()

    pts3d = read_3d_points_tri(args.object_name)
    #pts3d = read_3d_points_ply(args.object_name)
    #pts3d = read_3d_points(args.object_name)

    gt_path = args.gt_pose_file
    pred_path = args.pred_pose_file

    diameter = read_diameter(args.object_name)
    frames = extract_frame_ids(pred_path)
    all_distances = [] 
    count = 0
    R_rel = []
    t_rel = []

    for frame_id in frames:
        print('')
        print('Frame {}: '.format(frame_id))

        R_gt, t_gt = read_gt(frame_id, gt_path)
        R_est, t_est = read_pred(frame_id, pred_path)

        t_est = t_est * 1000 # transform t_est from m to mm # super important!!

        # Ensure R_gt is a proper 3x3 rotation matrix and t_gt a 3x1 vector
        R_gt = R_gt.reshape(3, 3) 
        t_gt = t_gt.reshape(3, 1) 

        # Ensure R_est is a proper 3x3 rotation matrix and t_est a 3x1 vector
        R_est = R_est.reshape(3, 3)
        t_est = t_est.reshape(3, 1)

        if (count==0): ## do it in the first frame
            # compute relative rotation matrix and vector
            R_rel = R.from_matrix(R_est).inv().as_matrix() @ R.from_matrix(R_gt).as_matrix()
            t_rel = t_gt - t_est
        count += 1
        #mean_distance = compute_add_score(pts3d,
        #                        (R_gt, t_gt),
        #                        (R_est, t_est), R_rel, t_rel)
        mean_distance = adi(R_est, t_est, R_gt, t_gt, R_rel, t_rel, pts3d)
        
        print('Mean distance is: {}'.format(mean_distance))
        all_distances.append(mean_distance)

    all_distances = np.array(all_distances)
    threshold = diameter * 0.1
    passed = (all_distances < threshold).sum()
    score = passed / all_distances.shape[0]

    frame_indices = np.arange(1, len(all_distances) + 1)

    print('')
    print('==============================================================================')
    print('')
    print(f'diameter is: {diameter}')
    print(f'threshold is: {threshold}')
    print(f'Mean distance between the points in all the frames: {np.mean(all_distances)}')
    print(f'Points under threshold: {passed}')
    print(f'total number of points: {all_distances.shape[0]}')
    print(f'ADD-S score: {score}')
    print('')

    # Create line plot for distances
    fig = go.Figure()

    # Add distances trace
    fig.add_trace(go.Scatter(x=frame_indices, y=all_distances, mode='lines', name='Distances', marker=dict(color='blue')))

    # Generate threshold values
    threshold_line = [threshold] * len(all_distances)

    # Add threshold line trace
    fig.add_trace(go.Scatter(x=frame_indices, y=threshold_line, mode='lines', name='Threshold', line=dict(color='red', dash='dash')))

    # Set axis labels and title
    fig.update_layout(xaxis_title='Frame Index', yaxis_title='Distance', title='Distances over Frame Index')

    # Set y-axis range
    fig.update_layout(yaxis=dict(range=[0, 1200]))


    # Show plot
    fig.show()

