#!/bin/bash

# Iterate over each subfolder in the current directory
for folder in */; do
    # Check if the subfolder "pose_vis" exists
    if [ -d "$folder/pose_vis" ]; then
        # Enter the "pose_vis" subfolder
        cd "$folder/pose_vis"

        # Create folder for resized images
        mkdir -p resized
        
        # Execute the ffmpeg command to resize each PNG file
        for file in *.png; do
            ffmpeg -i "$file" -vf "scale=854:480" "resized/$file" 
        done
        
        
        # Enter the "resized" subfolder
        cd "resized"
        
        # Run the ffmpeg command to create a video
        ffmpeg -framerate 30 -pattern_type glob -i '*.png' -c:v libx264 -pix_fmt yuv420p pose.mp4
        
        # Move back to the parent directory
        cd ../../
    fi

    if [ -d "$folder/color_segmented" ]; then
		cd "$folder/color_segmented"

        # Generate video of the segmented rgb frames
		ffmpeg -framerate 30 -i %05d.png -vf "scale=854:480" -c:v libx264 -preset slow -crf 22 -pix_fmt yuv420p color_segmented.mp4
		cd ..
	fi

     # Generate side to side video using the segmented rgb video and the pose video
     ffmpeg -i ./color_segmented/color_segmented.mp4 -i ./pose_vis/resized/pose.mp4 -filter_complex hstack=inputs=2 -c:v libx264 -crf 22 -preset slow -c:a aac -strict experimental output.mp4
done

