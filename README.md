# MOT-3D-Reconstruction
This short repository contains the code snippets and modifications that have been necessary to generate a RGBD dataset using a Microsoft Kinect camera, setup
some MOT & 3D Reconstruction SOTA codes, and produce ADD and CD scores for respectively the accuracy of the pose estimation and 3D reconstruction of the outputs.

#### Machine spec... ubuntu 20.04 ...

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

The library provides helpful funtions and a list of useful examples written in C++, but to be more flexible and comfortable it is possible to also exploit [pyk4a](https://github.com/etiennedub/pyk4a/), a python 3 wrapper for the Azure-Kinect-Sensor-SDK, and directly write code in python (it is anyway necessary to have compiled the SDK). Pyk4a does provide other similar code examples and functions. Check out the code StreamAndRecord.py that exploits pyk4a to stream and record live images from the camera!


### Camera Calibration File

## BundleSDF on Ubuntu 20.04, Setup and Modifications

### Resize Output Images

## ADD and CD Scores

## Generate Videos from Images
