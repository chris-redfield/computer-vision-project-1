import numpy as np
import cv2 as cv
from matplotlib import cm
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import glob

DATA_DIR = '../data/'
MIDDLEBURY_DIR = DATA_DIR + "Middlebury/"

def main():
    folder_list = glob.glob(MIDDLEBURY_DIR + "*/")
    for folder in folder_list:
        print("Processing folder", folder)

        calib_dict = load_calib_dict(folder)
        
        imgL, imgR = load_images(folder)
        imgL, imgR = preprocess_images(imgL, imgR, calib_dict)

        displ, dispr, left_matcher = compute_disparity(imgL, imgR, calib_dict)
        cv.imwrite(folder +"displ_TESTE.png",displ)


        disparity_map = apply_filter(displ, imgL, dispr, left_matcher, calib_dict)
        disparity_map = postprocess_disparity_map(disparity_map, calib_dict)

        cv.imwrite(folder +"disparidade.pgm",disparity_map)
        print("File disparidade.pgm saved on current folder")

        bad2 = compare_ground_truth(disparity_map, folder)
        print(f'bad 2.0 for image pair {folder}: {round(bad2,2)}')

        make_depth_map(disparity_map, folder, calib_dict)
        





if __name__ == "__main__":
    main()