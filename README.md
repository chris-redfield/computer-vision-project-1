# Computer Vision - Project 1

Stereo vision is the computer vision branch that allows the discovery of dimensions, shapes and positions of objects based on a pair of images. 

In this project we develop and explore stereo vision algorithms for: 

1. extracting depth maps using camera calibration parameters, both in parallel and convergent cameras; 
2. extracting objects measures in the 3D world.

## prerequisites
* python > 3.6
* OpenCV > 3
* numpy
* sklearn
* matplotlib

All available on requirements.txt file:

```console
pip install -r requirements.txt
```

## How to run
1. Disparity and depth map estimation from rectified stereo images

```console
python src/stereo_disparity.py
```

2. Disparity and depth map estimation from converged stereo cameras

3. 3D objects measurements from stereo cameras

