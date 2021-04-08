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

# How to run

## 1. Disparity and depth map estimation from rectified stereo images

Place the dataset image folders in data/Middlebury/, according to middlebury's 2014 stereo dataset standards and run the following:
```console
python src/stereo_disparity.py
```

The program will run individually for each folder found in data/Middlebury/. 

Example output with 2 folders inside:

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
 - profundidade.png: The depth map computed from the disparity map, with a colormap representing the distance in mm

stereomatching and filtering parameters are set dinamically, based on distance offset and focal length.

### Results:
- Disparity map:
![Disparity Map](https://raw.githubusercontent.com/chris-redfield/computer-vision-project-1/main/relatorio/disparity_playtable_SGBM_post_filtered.png)

- Depth map:
![Depth Map](https://raw.githubusercontent.com/chris-redfield/computer-vision-project-1/main/relatorio/disparity_playtable_SGBM_post_filtered_colormap.png)

## 2. Disparity and depth map estimation from converged stereo cameras

Place the Morpheus image pair from Yasutaka Furukawa and Jean Ponce dataset in data/FurukawaPonce/, along with the cameras calibration parameters and run:

```console
python src/converged_stereo_disparity.py
```

This will repeat the steps in last section, but since these images created from converged cameras, we'll have to preprocess the images before stereo matching.

example output:
```console
original images dims: ((1300, 1400, 3), (1200, 1200, 3))
images dims after reshape: ((1300, 1400, 3), (1300, 1400, 3))
Finding keypoints...
Finding homography...
Warping perspective...
Computing disparity...
Applying filter...
File disparidade.pgm saved on current folder
depth map saved at /home/user/proj/computer-vision-project-1/data/FurukawaPonce/profundidade.png
```

2 files will be generated in the same folder:
 - disparidade.pgm: The disparity map saved in pixels
 - profundidade.png: The depth map computed from the disparity map, with a colormap representing the distance in mm

### results
- Point matching after rectification:

![Point matching after rectification](https://raw.githubusercontent.com/chris-redfield/computer-vision-project-1/main/relatorio/morpheus_rectified_matching_points.png)

- Depth map:

![Depth Map](https://raw.githubusercontent.com/chris-redfield/computer-vision-project-1/main/relatorio/morpheus_depth_map.png)

## 3. 3D objects measurements from stereo cameras

Place the Morpheus image pair from Yasutaka Furukawa and Jean Ponce dataset in data/FurukawaPonce/, along with the cameras calibration parameters and run:

```console
python src/3d_measure.py
```

2 windows will open, the left image (img1) one and the right image (img2).

Double click 2 points on the left image, a line will form between them. Then select the same 2 points on the right image. A new line will show on the right image, and then the line measure will appear in cm.

Repeat the process for having a good idea about the object size.

### results
![3D Measures](https://raw.githubusercontent.com/chris-redfield/computer-vision-project-1/main/relatorio/morpheus_3D_measure2.png)