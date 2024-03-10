#!/bin/bash

# Directory containing subfolders
input="$1"

export MEGAPOSE_DATA_DIR="/local/home/fedona/Desktop/ETH/Bachelor/Datasets/dst_for_megapose/$input"


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

