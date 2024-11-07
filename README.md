# DEllipsoid-SLAM
An Object-level SLAM Method Based on Ellipsoidal Representation in Dynamic Scenes.

<div align="center">
  <img src="img/framework.png">
</div>

video address at: [https://www.bilibili.com/video/BV1jt421G7G9/?vd_source=737506656063f049ffdc96e35434c482](https://www.bilibili.com/video/BV1jt421G7G9/?vd_source=737506656063f049ffdc96e35434c482)/<br />

Our environments: ubuntu18.04/GCC7.5/ GPU: RTX2080ti.<br />
Our sourse code: coming soon.

System critical step:
<div align="center">
  <img src="img/association.png">
</div>

<div align="center">
  <img src="img/relocal.png">
</div>

## Installation

Our environments: ubuntu18.04/GCC7.5/GPU: RTX2080ti.

Our code is tested on:
* CMake 3.10.0
* Eigen 3.3.7
* NVIDIA CUDA 10.2
* OpenCV 4.5.5
* pytorch 1.6.0
* pcl 1.8.1
* ceres 1.14
* Sophus 1.0
## build on linux:
####### Build Raft.
sudo apt-get install libncurses5-dev\
virtualenv python-environment\
virtualenv yolo-env -p /usr/bin/python3.8
source yolo-env/bin/activate
pip install torch==1.6.0 torchvision==0.7.0\
pip install matplotlib tensorboard scipy opencv-python
####### Raft could be run here. Test it
python demo.py --model=models/raft-things.pth --path=demo-frames
####### Provide numpy headers to C++
ln -s `python -c "import numpy as np; print(np.__path__[0])"`/core/include/numpy include/raft/ || true 
####### Install ceres 's dependency 
sudo apt-get install libvtk7-dev libsuitesparse-dev liblapack-dev libblas-dev libgtk2.0-dev
####### Build ceres 's dependency
gflags 2.2.2 set 'BUILD_SHARED_LIBS' to ON\
glog 0.6.0\
boost 1.65
####### Build ceres
ceres 1.14.0
####### Build Opt
    git clone https://github.com/niessner/Opt.git
    as in dynamic fusion
Optional:

* Pangolin

```
mkdir build && cd build
cmake -DVISUALIZATION=ON ..
make -j8
```

We use -DVISUALIZATION=OFF/ON to switch visualization plug.
