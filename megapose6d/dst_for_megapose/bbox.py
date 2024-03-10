import os
import cv2
import numpy as np
import json
import sys

def find_bounding_box(image_path):
    # Read the grayscale image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Threshold the image to get a binary mask
    _, binary_mask = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)

    # Find contours in the binary mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the bounding box of the largest contour
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        top_left = [x, y]
        bottom_right = [x + w, y + h]
        return top_left, bottom_right
    else:
        return None

# Function to generate object_data.json
def generate_object_data(folder_path):
    
    for root, dirs, files in os.walk(folder_path):
        for directory in dirs:
            object_data = []
            image_path = os.path.join(root, directory, 'image_mask.png')
            print('this is the image path {}'.format(image_path))
            bounding_box = find_bounding_box(image_path)
            print('bbox compute successfully!{}'.format(bounding_box))
            if bounding_box:
                label = directory
                bbox_modal = [bounding_box[0][0], bounding_box[0][1], bounding_box[1][0], bounding_box[1][1]]
                object_data.append({"label": label, "bbox_modal": bbox_modal})

                # Write object_data to JSON file
                output_file = os.path.join(folder_path, label,'inputs/object_data.json')
                print(output_file)
                with open(output_file, 'w') as f:
                    json.dump(object_data, f)
                    print('file written! ')
    


#folder_path = './blenderproc_dragon_megapose'       # change here
#generate_object_data(folder_path)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py folder_path")
        sys.exit(1)
    folder_path = sys.argv[1]
    generate_object_data(folder_path)
