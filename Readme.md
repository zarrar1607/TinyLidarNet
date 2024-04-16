# TinyLidarNet

[![Read Paper](https://img.shields.io/badge/Read-Paper-blue)](Link_to_Paper)
[![View Slides](https://img.shields.io/badge/View-Slides-green)](Link_to_Slides)
[![Watch Video](https://img.shields.io/badge/Watch-Video-red)](Link_to_Video)

## Setup
This repository is designed for use with ROS Melodic. Hardware we use include an NVIDIA Jetson Xavier NX, Hokuyo Lidar UST10-LX, and VESC-MKIV. The operating system used is Linux 5.10.104-tegra aarch64. Follow the steps below to set up the environment:

### Install Dependcies
Please install:
- ROS Melodic: Install from the official [official ROS Melodic](https://wiki.ros.org/melodic/Installation/Ubuntu) and follow the provided instructions.
- TensorFlow: If using an NVIDIA Jetson Xavier NX, install TensorFlow using the JetPack provided by NVIDIA. Follow the instructions on the [NVIDIA website](https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/index.html)

Clone this repository:
```
git clone https://github.com/zarrar1607/TinyLidarNet.git
```

The ROS_Workspace folder is dedicated to setting up the ROS environment. It follows the standard steps of creating a devel folder, builder folder, and CMakeLists.txt file. Within this folder, you'll find code sourced from the [f1tenth_system repository](https://github.com/f1tenth/f1tenth_system/tree/melodic).

This code has been customized to include a global joystick button functionality for seamless switching between autonomous and manual control modes. Additionally, the ROS laser filter package has been integrated into the system for enhanced sensor data processing.

## Citation
If you have found our work useful, please consider citing:

```
cite
```

