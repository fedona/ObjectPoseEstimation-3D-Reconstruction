import os
import json
import numpy as np
import sys

# Import the compute_add function
from add_score import add_norel, adi_norel
from add_score import read_3d_points_tri_default


import plotly.graph_objects as go

#python score_script.py /local/home/fedona/Desktop/ETH/Bachelor/Datasets/dst_for_megapose/tudl_test_megapose_dragon /local/home/fedona/Desktop/ETH/Bachelor/Datasets/tudl_test_all/test/000001/scene_gt.json 

'''

    Script for computing add and adi scores from megapose output folders
    inputs are the dataset's megapose format folder (with outputs ready) and the gt_file (either from bop datasets, or blenderproc generated datasets)

    uses "norel" from add_score file beacuse there is no need to transform the predicted poses, add_norel and adi_norel are exact copies of the methods from the BOP toolkit

'''

def read_object_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
        for item in data:
            label = item['label']
            if 'TWO' in item:
                quaternion = np.array(item['TWO'][0])
                t_pred = np.array(item['TWO'][1]) # Multiply by 1000
                return quaternion, t_pred
    AssertionError

def quaternion_to_rotation_matrix(q):
    q = np.array(q)
    x, y, z, w = q
    rotation_matrix = np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ])
    return rotation_matrix

def main(prediction_folder, gt_file):


    # Read ground data from the gt file
    gt_frame_data = {}
    with open(gt_file, 'r') as f:
        gt_data = json.load(f).items()

    for frame_number, entries in gt_data:
        frame_number = int(frame_number)
        entry = entries[0]
        gt_frame_data[frame_number] = [
            entry['cam_R_m2c'],
            entry['cam_t_m2c']
        ]

    prediction_folder = prediction_folder + '/examples'

    add_scores = []
    adi_scores = []

    # Process each example folder in the prediction folder
    for folder in os.listdir(prediction_folder):
        frame_folder_path = os.path.join(prediction_folder, folder)
        print(folder)
        if os.path.isdir(frame_folder_path):
            output_folder = os.path.join(frame_folder_path, 'outputs')
            if os.path.exists(output_folder):
                #print(output_folder)
                object_data_file = os.path.join(output_folder, 'object_data.json')
                if os.path.exists(object_data_file):
                    print('this is folder: {}'.format(folder))

                    # Read quaternion and t_pred from object_data.json
                    quaternion, t_pred = read_object_data(object_data_file)
                    t_pred = 1000 * t_pred
                    R_gt = np.array(gt_frame_data[int(folder)][0])
                    t_gt = np.array(gt_frame_data[int(folder)][1])

                    # Convert quaternion to rotation matrix
                    R_pred = quaternion_to_rotation_matrix(quaternion)

                    '''lista = R_pred.tolist()
                    R_pred = []

                    for l in lista:
                        for el in l:
                            R_pred.append(el)'''
                    
                    R_gt = R_gt.reshape(3, 3)

                    # Debug outputs
                    print('this is R_pred \n {}'.format(R_pred))
                    print('this this is R_gt \n {}'.format(R_gt))
                    print('this is t_pred \n {}'.format(t_pred))
                    print('this this is t_gt \n {}'.format(t_gt))

                    # Call compute_add function with R_pred, t_pred, R_gt, and t_gt
                    #score  = add_norel(R_pred, t_pred, R_gt, t_gt, read_3d_points_tri_default())
                    add_score  = add_norel(R_pred, t_pred, R_gt, t_gt, read_3d_points_tri_default())
                    add_scores.append(add_score)
                    print("Add result for frame ", folder, ":", add_score)

                    adi_score  = adi_norel(R_pred, t_pred, R_gt, t_gt, read_3d_points_tri_default())
                    adi_scores.append(adi_score)
                    print("Adi result for frame ", folder, ":", adi_score)
    return add_scores, adi_scores              

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py prediction_folder gt_file")
        sys.exit(1)
    
    prediction_folder = sys.argv[1]
    gt_file = sys.argv[2]
    add_scores, adi_scores = main(prediction_folder, gt_file)
    #diameter = 430.31 # mm for tudl dragon
    #diameter = 175.704 # mm for tudl blob/frog/gnome
    diameter = 352.356 # mm for tudl watercan
    #diameter = 71.9471 # mm for tless 14

    print("this is the current diameter: {}".format(diameter))

    print('this is the average of the add scores: \n {}'.format(np.mean(add_scores)))
    add_lower_than_threshold = len([score for score in add_scores if score < diameter*0.1]) # alpha = 0.1, is the default value when computing add score
    print('overall add score is {}'.format(add_lower_than_threshold/len(add_scores)))

    print('this is the average of the adi scores: \n {}'.format(np.mean(adi_scores)))
    adi_lower_than_threshold = len([score for score in adi_scores if score < diameter*0.1]) # alpha = 0.1, is the default value when computing add score
    print('overall adi score is {}'.format(adi_lower_than_threshold/len(adi_scores)))

    threshold = diameter * 0.1
    all_distances = add_scores 
    #all_distances = adi_scores 
    frame_indices = np.arange(1, len(all_distances) + 1)

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
