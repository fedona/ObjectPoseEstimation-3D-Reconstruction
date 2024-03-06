import os
import json
import numpy as np
import argparse
import sys

"""
    Short python script to extract poses computed by BundleSDF and put them
    in the same json file following the BOP's datasets standard
"""
def  main(args):
    # Path to the folder containing predicted poses

    if not args.pose_path:
        sys.exit("Error: Please specify the path to the output folder using the --pose_path argument. (No need to add /ob_in_cam/)")

    folder_path = args.pose_path + 'ob_in_cam/'
    poses = {}
    sorted_files = sorted(os.listdir(folder_path))
    #print(os.listdir(folder_path))

    # Iterate over each text file in the folder
    for filename in sorted_files:
        if filename.endswith(".txt"):
            frame_number = int(filename.split(".")[0])
            with open(os.path.join(folder_path, filename), "r") as file:
                lines = file.readlines()
                R = []
                t = []
                for i, line in enumerate(lines):
                    if i < 3:
                        R.append(list(map(float, line.strip().split()))[:3])
                        t.append(list(map(float, line.strip().split()))[3])
            poses[frame_number] = [{
                "cam_R_m2c": np.ravel(R).tolist(), "cam_t_m2c": np.ravel(t).tolist(), "obj_id": 1
            }]
            
    total_frames = len(os.listdir(folder_path))        
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
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--pose_path', type=str, help='Path to the folder containing predicted poses')

    args = parser.parse_args()
    main(args)

