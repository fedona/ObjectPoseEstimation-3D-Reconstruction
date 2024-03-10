
# Script for generating a suitable dataset for megapose6d starting from 
# a dataset generated with the StreamAndRecord script


# Function to organize the dataset
extract_and_copy_images() {
    start_folder=$(pwd)
    input_folder=$1
    output_folder=$2
    model=$3
    camera_data=$4

    echo $output_folder

    mkdir $output_folder
    mkdir $output_folder/megapose-models/

    cp -r /local/home/fedona/Desktop/ETH/Bachelor/Datasets/dst_for_megapose/tudl_test_megapose_dragon/megapose-models $output_folder

    mkdir $output_folder/examples

    output_folder="$output_folder/examples"

    # Go into the RGB folder of the first folder
    rgb_folder="$input_folder/rgb"
    if [ -d "$rgb_folder" ]; then
        cd "$rgb_folder" || exit

        # Iterate over all files in the RGB folder
        for file_path in *; do
            if [ -f "$file_path" ]; then
                frame_number=$(basename "$file_path" | cut -d'.' -f1)
                frame_output_folder="$output_folder/$frame_number"

                # Create folder for frame if it doesn't exist
                if [ ! -d "$frame_output_folder" ]; then
                    mkdir -p "$frame_output_folder"
                fi

                # Copy image to frame folder
                cp "$file_path" "$frame_output_folder/image_rgb.png"
            fi
        done
    fi

    cd $start_folder

    masks_folder="$input_folder/masks"
    if [ -d "$masks_folder" ]; then
        cd "$masks_folder" || exit

        # Iterate over all files in the masks folder
        for file_path in *; do
            if [ -f "$file_path" ]; then
                frame_number=$(basename "$file_path" | cut -d'.' -f1)
                frame_output_folder="$output_folder/$frame_number"

                # Create folder for frame if it doesn't exist
                if [ ! -d "$frame_output_folder" ]; then
                    mkdir -p "$frame_output_folder"
                    echo WARNING masks frame
                fi

                # Copy image to frame folder
                cp "$file_path" "$frame_output_folder/image_mask.png"
            fi
        done
    fi

    cd $start_folder

    depth_folder="$input_folder/depth"
    if [ -d "$depth_folder" ]; then
        cd "$depth_folder" || exit

        # Iterate over all files in the masks folder
        for file_path in *; do
            if [ -f "$file_path" ]; then
                frame_number=$(basename "$file_path" | cut -d'.' -f1)
                frame_output_folder="$output_folder/$frame_number"

                # Create folder for frame if it doesn't exist
                if [ ! -d "$frame_output_folder" ]; then
                    mkdir -p "$frame_output_folder"
                    echo WARNING depth frame
                fi

                # Copy image to frame folder
                cp "$file_path" "$frame_output_folder/image_depth.png"
            fi
        done
    fi

    cd $output_folder
    for file_path in *; do
        # Create folder for frame if it doesn't exist
        if [ ! -d "meshes" ]; then
            mkdir -p "$file_path/meshes"
        fi

        # Create folder for frame if it doesn't exist
        if [ ! -d "inputs" ]; then
            mkdir -p "$file_path/inputs"
        fi

        frame_number=$(basename "$file_path" | cut -d'.' -f1)

        mkdir -p "$file_path/meshes/$frame_number"

        cp $start_folder/models/$model $file_path/meshes/$frame_number/hope_000002.ply
        cp $start_folder/$camera_data $file_path/camera_data.json
    done
}

# Example usage
start_folder=$(pwd)


input_folder_name="$1"
output_folder_name="$2"
model_name="$3"
camera_d="$4"

input_dataset_folder=$start_folder/$input_folder_name                                                       
output_dataset_folder=$start_folder/$output_folder_name   

extract_and_copy_images "$input_dataset_folder" "$output_dataset_folder" "$model_name" "$camera_d"
cd $start_folder
folder_name=$(basename "$output_dataset_folder")
echo $output_dataset_folder
python bbox.py $output_dataset_folder/examples

