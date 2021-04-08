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

Place the dataset image folders in data/Middlebury/ and run the following:
```console
python src/stereo_disparity.py
```

The program will run individually for each folder found in data/Middlebury/. For example:

```console
Processing folder /home/user/proj/computer-vision-project-1/data/Middlebury/Playtable-perfect/
Computing disparity...
Applying filter...
File disparidade.pgm saved on current folder
Comparing to ground truth
bad 2.0 for image pair /home/user/proj/computer-vision-project-1/data/Middlebury/Playtable-perfect/: 0.45
Computing depth map
depth map saved at /home/user/proj/computer-vision-project-1/data/Middlebury/Playtable-perfect/profundidade.png

Processing folder /home/user/proj/computer-vision-project-1/data/Middlebury/Jadeplant-perfect/
Computing disparity...
Applying filter...
File disparidade.pgm saved on current folder
Comparing to ground truth
bad 2.0 for image pair /home/user/proj/computer-vision-project-1/data/Middlebury/Jadeplant-perfect/: 0.61
Computing depth map
depth map saved at /home/user/proj/computer-vision-project-1/data/Middlebury/Jadeplant-perfect/profundidade.png
```
2 files will be generated in each folder:
 - disparidade.pgm: The disparity map saved in pixels
 - profundidade.png: The depth map computed from the disparity map

2. Disparity and depth map estimation from converged stereo cameras

3. 3D objects measurements from stereo cameras

