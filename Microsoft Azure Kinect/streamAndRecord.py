from argparse import ArgumentParser

import pyk4a
from helpers import colorize
from pyk4a import Config, ImageFormat, PyK4A, PyK4ARecord
import numpy as np
import cv2
import os
from PIL import Image
import subprocess

parser = ArgumentParser(description="pyk4a recorder")
parser.add_argument("--device", type=int, help="Device ID", default=0)
args = parser.parse_args()

print('What is the name of the recording?')
name = input()

print(f"Starting device #{args.device}")
config = Config(
            color_resolution=pyk4a.ColorResolution.RES_720P,
            depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
            synchronized_images_only=False,
            )
device = PyK4A(config=config, device_id=args.device)
device.start()

rgb_folder = name + "/rgb_images_jpg"
rgb_folder_png = name + "/rgb_images"
depth_folder = name + "/depth_images"
depthColored_folder = name + "/depthColored_images"

os.makedirs(rgb_folder, exist_ok=True)
os.makedirs(rgb_folder_png, exist_ok=True)
os.makedirs(depth_folder, exist_ok=True)
os.makedirs(depthColored_folder, exist_ok=True)
imagesCount = 1

try:
    print("Recording... Press CTRL-C to stop recording.")
    print("================== Start ==================")
    while True:
        capture = device.get_capture()

        if np.any(capture.color):
            print(f"Current Frame: {imagesCount}", end='\r')

            # Stream current colored captured image
            cv2.imshow("k4a", capture.color[:, :, :3])

            # Stream current colored depth image
            #cv2.imshow("Transformed Depth", colorize(capture.transformed_depth, (None, 5000)))
            
            key = cv2.waitKey(10)
            
            # RGB Image
            rgb = capture.color[:, :, :3]
            rgb = rgb[:, :, ::-1]   # switch RB channels to match pillow's format
            current_rgb = Image.fromarray(rgb, "RGB")
            
            # Colored Depth Image
            #depthColored = colorize(capture.transformed_depth, (None, 5000))
            #depthColored = depthColored[:, :, ::-1]   # switch RB channels
            #current_depthColored = Image.fromarray(depthColored)

            # Depth Image
            depth = capture.transformed_depth
            current_depth = Image.fromarray(depth)
            current_depth = current_depth.convert('I')

            image_number = str(imagesCount).zfill(5)

            # Set output paths
            path_1 = os.path.join(rgb_folder, f"{image_number}.jpg")
            #path_2 = os.path.join(depthColored_folder, f"{image_number}.jpg")
            path_3 = os.path.join(depth_folder, f"{image_number}.png")
            
            # Save captures
            current_rgb.save(path_1)
            #current_depthColored.save(path_2)
            current_depth.save(path_3)

            imagesCount += 1
            if key != -1:
                cv2.destroyAllWindows()
                break
        
except KeyboardInterrupt:
    print("CTRL-C pressed. Exiting.")
    device.stop()

print(f"{imagesCount} frames written.")

# Convert all JPG files with ffmpeg
ffmpeg_command = [
    "ffmpeg",
    "-hide_banner",
    "-i",
    os.path.join(rgb_folder, "%05d.jpg"),  # Input
    os.path.join(rgb_folder_png, "%05d.png"),  # Output 
]

subprocess.run(ffmpeg_command, check=True)