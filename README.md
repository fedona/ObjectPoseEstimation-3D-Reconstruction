# MOT-3D-Reconstruction
This short repository contains the code snippets and modifications that have been necessary to generate a RGBD dataset using a Microsoft Kinect camera, setup
some MOT & 3D Reconstruction SOTA codes, and produce ADD and CD scores for respectively the accuracy of the pose estimation and 3D reconstruction.

#### Machine specifications:
* CPU: Intel Core i7-9700 @ 3.6 GHz
* OS: Ubuntu 20.04
* GPU: NVIDIA GeForce RTX 2080
* CUDA:
  * Driver Version: 545.23
  * Release: 12.3

## Microsoft Kinect Azure SDK, Streaming and Recording RGBD
The Microsoft Kinect Azure Sensor is a Microsoft product for developers that has been discontinued from October 2023, the product is not available anymore and its software library is not being updated anymore. The official [Azure-Kinect-Sensor-SDK](https://github.com/microsoft/Azure-Kinect-Sensor-SDK) library, necessary to stream live images from the camera, is available just for Microsoft 10 and Ubuntu 18.04, but it is still possible to use it on Ubuntu 20.04  following the directions provided by [this](https://github.com/microsoft/Azure-Kinect-Sensor-SDK/issues/1263#issuecomment-710698591) comment on the SDK's official github repository. Following are copied down the same instructions.

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

## Generate Videos from Images
While working with images, a very useful tool for videos generation, image trasformations, frames extracions, overlaying, ect... can be [ffmpeg](https://ffmpeg.org/). For example, with this simple and intuitive command it is possible to generate an output video from a list of input png images:

```bash
ffmpeg -framerate 30 -pattern_type glob -i '*.png' -c:v libx264 -pix_fmt yuv420p output.mp4
```

## BundleSDF on Ubuntu 20.04, Setup and Modifications
[BundleSDF](https://bundlesdf.github.io/) is a method for 6-Dof tracking and 3D reconstruction of unknown objects from a monocular RGBD video sequence.
At the time of February 2024, its official [github repository](https://github.com/NVlabs/BundleSDF) provides straightforward instructions to setup their docker image and run the code successfully. 

Although this should be machine-independent as docker is installed and there is an available CUDA GPU, it has been necessary to install `g++-11` before building the docker image, here follows a list of commands to do so:

```bash
# check if g++-11 is installed
which g++-11

# if not installed do:
sudo apt install build-essential manpages-dev software-properties-common
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt update && sudo apt install gcc-11 g++-11
```

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
on this same file is possible to tweak other parameters, for examples the ones that determine the minimal angle between different keyframes.


### Resize Output Images
When BundleSDF happends to produce output pose_vis images with odd numer of pixels, ffmpeg will fail to generate a video. For this case it is necessary to first resize those images to the correct resolution with this command: 

```bash
    ffmpeg -i input.png -vf scale=1280:720 output.png # with final resolution 1280x720
```

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
