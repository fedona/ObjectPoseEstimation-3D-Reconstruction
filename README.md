# MOT-3D-Reconstruction
This short repository contains the code snippets and modifications that have been necessary to generate a RGBD dataset using a Microsoft Kinect camera, setup
some MOT & 3D Reconstruction SOTA codes, and produce ADD and CD scores for respectively the accuracy of the pose estimation and 3D reconstruction.

TODO
bundlesdf all the bash files for the reuslts
images
pretty touches

#### Machine specifications:
* CPU: Intel Core i7-9700 @ 3.6 GHz
* OS: Ubuntu 20.04
* GPU: NVIDIA GeForce RTX 2080
* CUDA:
  * Driver Version: 545.23
  * Release: 12.3

## Microsoft Kinect Azure SDK, Streaming and Recording RGBD
The Microsoft Kinect Azure Sensor is a Microsoft product for developers that has been discontinued from October 2023, the product is not available anymore and its software library is not being updated. The sensor has two cameras, one records rgb 3840x2160 images while the other captures depth informations with resolution 1024x1024.

<div align='center'><img src='https://github.com/fedona/MOT-3D-Reconstruction/blob/main/images/azure_kinect.jpg' height=200></div>

The official [Azure-Kinect-Sensor-SDK](https://github.com/microsoft/Azure-Kinect-Sensor-SDK) library, necessary to stream live images from the camera, is available just for Microsoft 10 and Ubuntu 18.04, but it is still possible to use it on Ubuntu 20.04  following the directions provided by [this](https://github.com/microsoft/Azure-Kinect-Sensor-SDK/issues/1263#issuecomment-710698591) comment on the SDK's official github repository. Following are copied down the same instructions.

Connect with the microsoft package repository:
```
curl -sSL https://packages.microsoft.com/keys/microsoft.asc | sudo apt-key add -
sudo apt-add-repository https://packages.microsoft.com/ubuntu/18.04/prod
curl -sSL https://packages.microsoft.com/config/ubuntu/18.04/prod.list | sudo tee /etc/apt/sources.list.d/microsoft-prod.list
curl -sSL https://packages.microsoft.com/keys/microsoft.asc | sudo apt-key add -
sudo apt-get update
```

Install the necessary packages:
```
sudo apt install libk4a1.3-dev
sudo apt install libk4abt1.0-dev
sudo apt install k4a-tools=1.3.0
```

Check if the installation worked and connects to the camera:

```
k4aviewer
```

The library provides helpful funtions and a list of useful examples written in C++, but to be more flexible and comfortable it is possible to also exploit [pyk4a](https://github.com/etiennedub/pyk4a/), a python 3 wrapper for the Azure-Kinect-Sensor-SDK, and directly write code in python (it is anyway necessary to have compiled the SDK). Pyk4a does provide other similar code examples and functions. The most important things to do for streaming and recording the RGBD captures from the Kinect Sensor are the following:


Configure the cameras settings and start the device:
```
config = Config(
            color_resolution=pyk4a.ColorResolution.RES_720P,
            depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
            synchronized_images_only=False,
            )
device = PyK4A(config=config, device_id=args.device)
device.start()
```

Capture images from the cameras:
```
capture = device.get_capture()
rgb = capture.color[:, :, :3]
depth = capture.transformed_depth
```

The function `transformed_depth` automatically transforms the depth camera images to the same format as the rgb camera images, this is not necessary but simplifies the work for later use, as this makes necessary to specify only one single camera calibration file instead of two. Popular datasets as the ones used in the BOP Challenge all specify one single camera calibration configuration for both rgb and depth images.

Here follows an overlay of a rgb video with the colored depth images produced with the above function.

<div align='center'><img src='https://github.com/fedona/MOT-3D-Reconstruction/blob/main/videos/rgbd_blend.gif'></div>

It is good to notice that the uncolored parts of the frames corresponds to those parts that are invisible from the point of view of the depth camera but visible from the one of the rgb camera.

Check out the [streamAndRecord.py](https://github.com/fedona/MOT-3D-Reconstruction/blob/main/Microsoft%20Azure%20Kinect/streamAndRecord.py) file for a complete version  of it!

### Camera Calibration File
Camera-recorded datasets need to specify their camera calibration matrix such that datas can be normalized. It is possible to simply compute the camera calibration matrix of a camera when its focal length and resolution specifications are well known, here is how the matrix is composed:


<img src="https://github.com/fedona/MOT-3D-Reconstruction/blob/main/images/cam_k.png">


where 
* fx and fy are the focal length in the x and y axis directions
* cx and cy are the coordinates of the principal points in pixels
* s is the image skew, usually zero for most cameras

To get the same matrix informations it is possible to exploit camera calibration algorithms, for example Zhang's Method. The advantage is that they can be very preciese and subjective to the used camera, which could always differ a bit from the manufacturer specifications.

Many different implementations of camera calibration algorithms are avaiable online, for example the official Azure Kinect SDK provides its own code [here](https://github.com/microsoft/Azure-Kinect-Sensor-SDK/tree/develop/examples/calibration_registration).

An other convenient method is provided by the [video2calibration](https://github.com/microsoft/Azure-Kinect-Sensor-SDK/tree/develop/examples/calibration_registration) repository, this short python code requires a recorded video of a simple checked board as input, and as long as the board is visible through most of the frames, it returns an accurate camera calibration matrix.

This can be installed through pip:

```
pip install video2calibration
```

Here is the video recorded with the Kinect, the checkerboard has to be twisted especially around the corners of the frames to favorite the algorithm successfulness.

<div align='center'><img src="https://github.com/fedona/MOT-3D-Reconstruction/blob/main/videos/calibration.gif"></div>

## BundleSDF on Ubuntu 20.04
[BundleSDF](https://bundlesdf.github.io/) is a method for 6-Dof tracking and 3D reconstruction of unknown objects from a monocular RGBD video sequence.
At the time of February 2024, its official [github repository](https://github.com/NVlabs/BundleSDF) provides straightforward instructions to setup their docker image and run the code successfully. 

### Setup

Although this should be machine-independent as docker is installed and there is an available CUDA GPU, it has been necessary to install `g++-11` before building the docker image, here follows a list of commands to do so:

```bash
# check if g++-11 is installed
which g++-11

# if not installed do:
sudo apt install build-essential manpages-dev software-properties-common
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt update && sudo apt install gcc-11 g++-11
```

### Modifications

The authors tested BundleSDF on ubuntu 18.04 with a GeForce RTX 3090 as mentioned [here](https://github.com/NVlabs/BundleSDF/issues/18#issuecomment-1612242390) and suggest to have lot of [memory available](https://github.com/NVlabs/BundleSDF/issues/58#issuecomment-1671929927).

In order to run successfully BundleSDF on the demo dataset using a machine with limited GPU memory the following modifications are mandatory:

* modify loftr_wrapper.py reducing the batch size from 64 to 1
```python
    #batch_size = 64  FEDONA
    batch_size = 1
```

An other necessary modification to the code has to be made in run_custom.py, where the user has to set the distance of the object from the camera:
```python
    cfg_bundletrack['depth_processing']["zfar"] = 4  # object distant 4 meters from the camera 
```
Otherwise BundleSDF will fail to initialize the point cloud and will output this error code:
```bash
   [pcl::PLYWriter::writeASCII] Input point cloud has no data!
   [pcl::KdTreeFLANN::setInputCloud] Cannot create a KDTree with an empty input cloud!
   [pcl::PLYWriter::writeASCII] Input point cloud has no data!
```
on this same file is possible to tweak other parameters, for examples the ones that determine the minimal angle between different keyframes.

### Prepare your own dataset
The official repository provides a small example of a gallon of milk video available [here](https://drive.google.com/file/d/1akutk_Vay5zJRMr3hVzZ7s69GT4gxuWN/view?usp=share_link). This can be used to test if the code correctly runs without having to download large datasets as the suggested HO3D and YCBInEOAT.

If the "milk dataset" correctly runs then it is possible to prepare your own dataset; this has to be organized in the following way:
```
   root
    ├──rgb/    (PNG files)
    ├──depth/  (PNG files, stored in mm, uint16 format. Filename same as rgb)
    ├──masks/       (PNG files. Filename same as rgb. 0 is background. Else is foreground)
    └──cam_K.txt   (3x3 intrinsic matrix, use space and enter to delimit)
```
BundleSDF uses a wrapper for [XMem](https://github.com/hkchengrex/XMem) to generate the masks, but this is not included in the code due to license issues, so to generate the masks it is necessary to download XMem from their official repository.

An other fast and interactive way to get the masks of the subjects is to use other segmentation alternatives as [MiVOS](https://github.com/hkchengrex/MiVOS) or [Cutie](https://github.com/hkchengrex/Cutie).

### Produce pose_vis Videos
BundleSDF does generate under the folder pose_vis images of the predicted pose's bounding box around the subject. In order to comfortably see these results it is possible to generate a video using [ffmpeg](https://ffmpeg.org/).

<div align='center'><img src="https://github.com/fedona/MOT-3D-Reconstruction/blob/main/images/ffmpeg_logo.png" height=100></div>

Ffmpeg is a useful tool when working with images and audio files, it is also used for image trasformations, frames extracions, overlaying, format conversion, ...

For example, with this simple and intuitive command it is possible to generate an output video from a list of input png images:

```bash
   ffmpeg -framerate 30 -pattern_type glob -i '*.png' -c:v libx264 -pix_fmt yuv420p output.mp4
```

To produce videos using ffmpeg it is necessary to have frames of same resolution, and since BundleSDF happends to generate frames of odd resolution it might be necessary to resize them or ffmpeg will fail to generate a video.

This command is used to resize any frame to a correct resolution:
```bash
   for file in *.png; do
       ffmpeg -i "$file" -vf "scale=854:480" "resized/$file"
   done
```

This is the final result:

<div align='center'><img src="https://github.com/fedona/MOT-3D-Reconstruction/blob/main/videos/dragon.gif"></div>

Check out [videos.sh](https://github.com/fedona/MOT-3D-Reconstruction/blob/main/BundleSDF/videos.sh) for a complete script that resizes the frames in the pose_vis folder and generates a mp4 video.

## SAM6D

## BlenderProc

Check out also BlendedMVS

## ADD and CD Scores
In the field of pose estimation and 3D reconstruction there are many different errors/scores used to evaluate the results, for example, in the [BOP Challenge 2023](https://bop.felk.cvut.cz/challenges/bop-challenge-2023/#task4) there are considered three of them:
* VSD (Visible Surface Discrepancy)
* MSSD (Maximum Symmetry-Aware Surface Distance)
* MSPD (Maximum Symmetry-Aware Projection Distance)

To keep it short and simple here there are the implementations for computing two basic scores, one for the pose estimation and one for 3D reconstruction:
* ADD (Average Distance of the Data), having both ground truth and predicted poses, it projects any object to this poses and compares the results computing the mean distance between points, the score is the results of the ratio between poses that are below an acceptance threshold (typically 10% of the object's diameter) and the total number of poses.

  <img src="https://github.com/fedona/MOT-3D-Reconstruction/blob/main/images/addScore.png">
  
  ```python
  def compute_add_score(pts3d, gt_pose, pred_pose, R_rel, t_rel):
    """
    Compute the ADD score between two poses.

    Parameters:
    - pts3d: Numpy matrix of 3D points. nx3
    - gt_pose: Tuple containing ground truth pose (R_gt, t_gt).
    - pred_pose: Tuple containing predicted pose (R_pred, t_pred).

    Returns:
    - mean_distance
    """
    R_gt, t_gt = gt_pose        # 3x3, 3x1
    R_pred, t_pred = pred_pose

    ## Transform predicted pose to relative prediction pose
    R_pred = R.from_matrix(R_rel).as_matrix() @ R_pred
    t_pred = t_pred + t_rel

    # Transform 3D points to camera coordinate system
    pts3d_camera_gt = R_gt @ (pts3d.T) + t_gt

    # Transform 3D points to camera coordinate system
    pts3d_camera_pred = R_pred @ (pts3d.T) + t_pred

    # Compute distances between corresponding points
    distance = np.linalg.norm(pts3d_camera_gt - pts3d_camera_pred, axis=0)  # compute norm among the columns (distance between each point)
    mean_distance = np.mean(distance)   # mean of the distances
  
    print('  distance between points: {}'.format(distance))

    return mean_distance
  ```
  
* CD (Chamfer Distance), having both the ground truth and predicted 3D reconstruction of the object, computes the distance of each point of the objects to the closest point of the other object.
  
  <img src="https://github.com/fedona/MOT-3D-Reconstruction/blob/main/images/CDScore.png">
  
  ```python
  def compute_chamfer_distance(pts1, pts2):

    count = 0
    tot = len(pts1) + len(pts2)

    # Translate points to center them
    translated_pts1 = pts1 
    translated_pts2 = pts2 
    print('')

    # Compute chamfer distance 1 to 2
    distances1 = []
    for point in translated_pts1:
        count += 1
        distances1.append(np.min(np.linalg.norm(translated_pts2 - point, axis=1)))
        print(f'Computing distance 1 to 2 {round(count/tot, 3)*100}%', end="\r")

    # Compute chamfer distance 2 to 1
    distances2 = []
    for point in translated_pts2:
        count += 1
        distances2.append(np.min(np.linalg.norm(translated_pts1 - point, axis=1)))
        print(f'Computing distance 2 to 1 {round(count/tot, 3)*100}%', end="\r")

    d1to2 = np.mean(distances1)
    d2to1 = np.mean(distances2)

    print(f'Mean distance first obj to second obj: {d1to2}')
    print(f'Mean distance second obj to first obj: {d2to1}')

    chamfer_distance = (d1to2 + d2to1) / 2
    chamfer_distance_per_point = (d1to2 / len(translated_pts1) + d2to1 / len(translated_pts2))

    return chamfer_distance, chamfer_distance_per_point
  ```
for a complete version of the scores check out [add_score.py](https://github.com/fedona/MOT-3D-Reconstruction/blob/main/Scores/add_score.py) and [chamfer_distance.py](https://github.com/fedona/MOT-3D-Reconstruction/blob/main/Scores/chamfer_distance.py)
