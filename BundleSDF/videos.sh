#!/bin/bash

# Iterate over each subfolder in the current directory
for folder in */; do
    # Check if the subfolder "pose_vis" exists
    if [ -d "$folder/pose_vis" ]; then
        # Enter the "pose_vis" subfolder
        cd "$folder/pose_vis"
        
        # Check if the "resized" subfolder exists
        if [ -d "resized" ]; then
            # Enter the "resized" subfolder
            cd "resized"
            
            # Run the ffmpeg command to create a video
            ffmpeg -framerate 30 -pattern_type glob -i '*.png' -c:v libx264 -pix_fmt yuv420p pose.mp4
            
            # Move back to the "pose_vis" subfolder
            cd ..
        fi
        
        # Move back to the parent directory
        cd ../../
    fi
done

