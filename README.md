# MOT-3D-Reconstruction
This short repository contains the code snippets and modifications that have been necessary to generate a RGBD dataset using a Microsoft Kinect camera, setup
some MOT & 3D Reconstruction SOTA codes, and produce ADD and CD scores for respectively the accuracy of the pose estimation and 3D reconstruction of the outputs.

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

K = | fx   s  cx |
    |  0  fy  cy |
    |  0   0   1 |

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

## BundleSDF on Ubuntu 20.04, Setup and Modifications

### Resize Output Images

## ADD and CD Scores

## Generate Videos from Images
