import argparse
import numpy as np
import json
from scipy.spatial.transform import Rotation as R
import plyfile


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--object_name', type=str, default='000004')
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
    filename = 'lm_models/models/models_info.json'
    
    with open(filename, 'r') as f:
        data = json.load(f)

    # Remove leading zeros and convert to integer
    object_id = int(object_name.lstrip('0'))

    # Access the diameter field from the JSON data
    diameter_in_cm = data.get(str(object_id), {}).get('diameter', None) # here is in mm

    if diameter_in_cm is not None:
        return diameter_in_cm * 0.1
    else:
        print(f"Diameter not found for object '{object_name}' in the JSON file.")
        return None

def read_gt(frame_id, object_name):
    filename = './{}/scene_gt.json'.format(object_name)

    with open(filename, 'r') as f:
        data = json.load(f)

    # Check if data is a dictionary and the frame_id is present
    if not isinstance(data, dict) or str(frame_id) not in data:
        print(f"Frame {frame_id} not found for object '{object_name}' in the JSON file.")
        return None

    # Extract the relevant information from the frame entry
    frame_entry = data[str(frame_id)][0]  # Assuming there's only one entry per frame
    R_gt = frame_entry.get('cam_R_m2c', None)
    t_gt = frame_entry.get('cam_t_m2c', None)

    if R_gt is None:
        print(f"R matrix not found for object '{object_name}' in the JSON file at frame id {frame_id}.")
        return None

    return np.array(R_gt), np.array(t_gt)

def read_pred(frame_id, object_name):
    filename = './scene_prediction{}.json'.format(object_name)

    with open(filename, 'r') as f:
        data = json.load(f)

    # Check if data is a dictionary and the frame_id is present
    if not isinstance(data, dict) or str(frame_id) not in data:
        print(f"Frame {frame_id} not found for object '{object_name}' in the JSON file.")
        return None

    # Extract the relevant information from the frame entry
    frame_entry = data[str(frame_id)][0]  # Assuming there's only one entry per frame
    R_pred = frame_entry.get('cam_R_m2c', None)
    t_pred = frame_entry.get('cam_t_m2c', None)

    if R_pred is None:
        print(f"R matrix not found for object '{object_name}' in the JSON file at frame id {frame_id}.")
        return None

    return np.array(R_pred), np.array(t_pred)

def extract_frame_ids(object_name):

    json_file_path = '{}/scene_gt.json'.format(object_name)

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
    - pts3d: Numpy matrix of 3D points. nx3
    - gt_pose: Tuple containing ground truth pose (R_gt, t_gt).
    - pred_pose: Tuple containing predicted pose (R_pred, t_pred).

    Returns:
    - mean_distance
    """
    R_gt, t_gt = gt_pose        # 3x3, 3x1
    R_pred, t_pred = pred_pose

    ## Transform predicted pose to relative prediction pose
    R_pred = R.from_matrix(R_rel).as_matrix() @ R_pred
    t_pred = t_pred + t_rel

    # Transform 3D points to camera coordinate system
    pts3d_camera_gt = R_gt @ (pts3d.T) + t_gt

    # Transform 3D points to camera coordinate system
    pts3d_camera_pred = R_pred @ (pts3d.T) + t_pred

    # Compute distances between corresponding points
    distance = np.linalg.norm(pts3d_camera_gt - pts3d_camera_pred, axis=0)  # compute norm among the columns (distance between each point)
    mean_distance = np.mean(distance)   # mean of the distances
    max_distance = np.max(distance)
    min_distance = np.min(distance)

    #print('  this is the GT        : {}'.format(R_gt))
    #print('  this is the prediction: {}'.format(R_pred))
    print('  distance between points: {}'.format(distance))
    print('  max distance between some points: {}'.format(max_distance))
    print('  min distance between some points: {}'.format(min_distance))

    return mean_distance

# main function
if __name__ == '__main__':
    args = parse_args()
    #record = np.load(args.prediction_file, allow_pickle=True).item()

    pts3d = read_3d_points(args.object_name)
    #pts3d = read_3d_points_ply(args.object_name)

    diameter = read_diameter(args.object_name)
    frames = extract_frame_ids(args.object_name)
    all_distances = [] 
    count = 0
    R_rel = []
    t_rel = []

    for frame_id in frames:
        print('')
        print('Frame {}: '.format(frame_id))

        R_gt, t_gt = read_gt(frame_id, args.object_name)
        R_pred, t_pred = read_pred(frame_id, args.object_name)

        # Ensure R_gt is a proper 3x3 rotation matrix and t_gt a 3x1 vector
        R_gt = R_gt.reshape(3, 3) 
        t_gt = t_gt.reshape(3, 1) 

        # Ensure R_pred is a proper 3x3 rotation matrix and t_pred a 3x1 vector
        R_pred = R_pred.reshape(3, 3)
        t_pred = t_pred.reshape(3, 1)

        if (count==0): ## do it in the first frame
            # compute relative rotation matrix and vector
            R_rel = R.from_matrix(R_gt).inv().as_matrix() @ R.from_matrix(R_pred).as_matrix()
            t_rel = t_gt - t_pred

        mean_distance = compute_add_score(pts3d,
                                (R_gt, t_gt),
                                (R_pred, t_pred), R_rel, t_rel)
        
        print('Mean distance is: {}'.format(mean_distance))
        all_distances.append(mean_distance)

    all_distances = np.array(all_distances)
    threshold = diameter * 0.1
    passed = (all_distances < threshold).sum()
    score = passed / all_distances.shape[0]

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
