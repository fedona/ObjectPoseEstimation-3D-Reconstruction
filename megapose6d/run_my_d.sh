#!/bin/bash

# Small bash script used to run megapose6d demo on multiple images from the same (converted) dataset
# it is necessary to change the paths accordingly

# Directory containing subfolders
input="$1"

# This command is necessary to set an environment variable that tells the megapose demo where to find the dataset
export MEGAPOSE_DATA_DIR="/local/home/fedona/Desktop/ETH/Bachelor/Datasets/dst_for_megapose/$input"

# This is the location of the folder
base_dir="/local/home/fedona/Desktop/ETH/Bachelor/Datasets/dst_for_megapose/$input/examples/"

# Iterate over subfolders
for subfolder in "$base_dir"/*/; do
    subfolder_name=$(basename "$subfolder")
    echo $subfolder_name
    # Execute first command
    echo "Running inference for $subfolder_name"
    python -m megapose.scripts.run_inference_on_example "$subfolder_name" --run-inference
    
    # Execute second command
    echo "Visualizing outputs for $subfolder_name"
    python -m megapose.scripts.run_inference_on_example "$subfolder_name" --vis-outputs
done

