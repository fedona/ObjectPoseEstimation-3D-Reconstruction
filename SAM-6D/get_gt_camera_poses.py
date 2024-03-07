import os
import json
import numpy as np
import argparse
import sys

"""
    Short python script to get camera gt poses from the file cam_poses_level0.npy that is being used in the SAM6D demo
    to produce the demo's dataset:
        the object is fixed and there are 42 different cameras rendering the object in different poses
"""
def  main():
    
    folder_path = "./cam_poses_level0.npy"
    poses = {}
    big_np_stuff = np.load(folder_path)
    #print(os.listdir(folder_path))
    frame_number = 0

    for matrix in big_np_stuff:
        R = matrix[:3, :3]
        t = matrix[:3, 3]
        #print(matrix)
        #print(R)
        #print(t)
        poses[frame_number] = [{
                "cam_R_m2c": np.ravel(R).tolist(), "cam_t_m2c": np.ravel(t).tolist(), "obj_id": 1
            }]
        frame_number += 1
        #break
            
    total_frames = frame_number     
    output_file = "poses.json"
    with open(output_file, "w") as outfile:
        outfile.write('{\n')
        for i, (frame_number, pose_info) in enumerate(poses.items()):
            outfile.write('\"' + str(frame_number) + '\": ')
            json.dump(pose_info, outfile)
            if i < total_frames - 1:
                outfile.write(',\n')
            else:
                outfile.write('\n')
        outfile.write('}')

    print(f"Pose information has been saved to {output_file}")

if __name__ == "__main__":
    """parser = argparse.ArgumentParser(description='')
    parser.add_argument('--pose_path', type=str, help='Path to the folder containing predicted poses')

    args = parser.parse_args()"""
    main()

