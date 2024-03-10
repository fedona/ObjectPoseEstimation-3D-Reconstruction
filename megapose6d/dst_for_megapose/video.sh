#!/bin/bash

input="$1"
fps="$2"

# Directory containing subdirectories
base_dir="/local/home/fedona/Desktop/ETH/Bachelor/Datasets/dst_for_megapose/$input/examples"

# Output directory for the video
output_dir="/local/home/fedona/Desktop/ETH/Bachelor/Datasets/dst_for_megapose/videos/$input"

# Create output directory if it doesn't exist
mkdir -p "$output_dir"

# Iterate over subdirectories
for subdirectory in "$base_dir"/*/; do
    #echo "Processing subdirectory: $subdirectory"
    
    # Check if the visualizations folder exists
    visualizations_folder="$subdirectory/visualizations"
    if [ -d "$visualizations_folder" ]; then
        # Check if the 'all_results.png' file exists
        all_results_image="$visualizations_folder/all_results.png"
        if [ -f "$all_results_image" ]; then
            # Copy 'all_results.png' to the output directory
            cp "$all_results_image" "$output_dir/$(basename "$subdirectory").png"
        fi
    fi
done

# Use ffmpeg to generate a video from the images
ffmpeg -framerate $fps -pattern_type glob -i "$output_dir/*.png" -c:v libx264 -pix_fmt yuv420p "$output_dir/video.mp4"

echo "Video generated: $output_dir/video.mp4"

