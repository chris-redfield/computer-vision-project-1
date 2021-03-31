import numpy as np
import cv2 as cv
from matplotlib import cm
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import glob

DATA_DIR = '../data/'
MIDDLEBURY_DIR = DATA_DIR + "Middlebury/"

def load_images(dir_path):
    imgL = cv.imread(dir_path + 'im0.png')
    imgR = cv.imread(dir_path + 'im1.png')
    return imgL, imgR

def compute_disparity(imgL, imgR):
    window_size = 3
    left_matcher = cv.StereoSGBM_create(
        minDisparity=0,
        numDisparities=160,             
        blockSize=5,
        P1=8 * 3 * window_size ** 2,    
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=0,
        speckleRange=2,
        preFilterCap=63,
        mode=cv.STEREO_SGBM_MODE_SGBM_3WAY
    )
    right_matcher = cv.ximgproc.createRightMatcher(left_matcher)
    print('Computing disparity...')
    displ = left_matcher.compute(imgL, imgR)
    dispr = right_matcher.compute(imgR, imgL)
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    return displ, dispr, left_matcher

def apply_filter(displ, imgL, dispr, left_matcher):
    lmbda = 80000
    sigma = 1.2
    visual_multiplier = 1.0

    wls_filter = cv.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)
    print('Applying filter...')
    filteredImg = wls_filter.filter(displ, imgL, None, dispr)
    filteredImg = cv.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv.NORM_MINMAX)
    filteredImg = np.uint8(filteredImg)
    return filteredImg

def load_calib_dict(dir_path):
    d = {}
    with open(dir_path + "calib.txt") as f:
        for line in f:
            (key, val) = line.split("=")
            d[key] = val.replace("\n","")
    return d

def main():
    folder_list = glob.glob(MIDDLEBURY_DIR + "*/")
    for folder in folder_list:
        print("Processing folder", folder)
        imgL, imgR = load_images(folder)
        displ, dispr, left_matcher = compute_disparity(imgL, imgR)
        disparity_map = apply_filter(displ, imgL, dispr, left_matcher)
        cv.imwrite(folder +"disparidade.pgm",disparity_map)
        print("File disparidade.pgm saved on current folder")

        calib_dict = load_calib_dict(folder)

        # Z = baseline * f / (d + doffs)


    # # Window name in which image is displayed
    # window_name = 'image'
    
    # # Using cv2.imshow() method 
    # # Displaying the image 
    # cv.imshow(window_name, disparity_map)
    
    # #waits for user to press any key 
    # #(this is necessary to avoid Python kernel form crashing)
    # cv.waitKey(0) 
    
    # #closing all open windows 
    # cv.destroyAllWindows() 




if __name__ == "__main__":
    main()