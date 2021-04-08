import numpy as np
import cv2 as cv
from matplotlib import cm
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import glob
import os

#DATA_DIR = '../data/'
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'data/'))
MIDDLEBURY_DIR = DATA_DIR + "/Middlebury/"

def relu(m):
    """ rectified linear unit """
    m[m < 0] = 0

def load_images(dir_path):
    """ """
    imgL = cv.imread(dir_path + 'im0.png')
    imgR = cv.imread(dir_path + 'im1.png')
    return imgL, imgR

def preprocess_images(imgL, imgR, calib_dict):
    """ """
    pixel_offset = int(calib_dict['ndisp'])
    pixel_offset = int(pixel_offset / 16) * 16
    imgL = cv.copyMakeBorder(imgL,0,0,pixel_offset,0,cv.BORDER_CONSTANT)
    imgR = cv.copyMakeBorder(imgR,0,0,pixel_offset,0,cv.BORDER_CONSTANT)
    return imgL, imgR

def postprocess_disparity_map(filteredImg, calib_dict):
    """ """
    pixel_offset = int(calib_dict['ndisp'])
    pixel_offset = int(pixel_offset / 16) * 16
    filteredImg = filteredImg[:,pixel_offset:]
    
    ### adds relu to handle negative values
    relu(filteredImg)
    return filteredImg

def compute_disparity(imgL, imgR, calib_dict):
    """   """
    num_disparities = int(calib_dict['ndisp'])
    num_disparities = int(num_disparities / 16) * 16

    window_size = 3
    
    left_matcher = cv.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparities,             
        blockSize=5,
        P1=8 * 3 * window_size ** 2,    
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=0,
        speckleRange=2,
        preFilterCap=2,
        mode=cv.STEREO_SGBM_MODE_SGBM_3WAY
    )
    right_matcher = cv.ximgproc.createRightMatcher(left_matcher)
    
    print('Computing disparity...')
    displ = left_matcher.compute(imgL, imgR)
    dispr = right_matcher.compute(imgR, imgL)
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    
    return displ, dispr, left_matcher

def apply_filter(displ, imgL, dispr, left_matcher, calib_dict):
    """  """
    doffs = float(calib_dict['doffs'])
    f = float(calib_dict['cam0'].split(' ')[0].replace('[',""))
    x = f * doffs
    
    #print(f'doffs: {doffs}, f: {f}')

    #inverses X, so that large numbers became small
    x = x ** -1
    x = int(x * 10000000)
    lmbda = x * 4000
    #print(f'lambda: {lmbda}')
    #lmbda = 160000
    sigma = 1.2

    wls_filter = cv.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)

    print('Applying filter...')
    filteredImg = wls_filter.filter(displ, imgL, None, dispr)
    
    filteredImg = filteredImg/16
    # filteredImg = cv.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv.NORM_MINMAX)
    # filteredImg = np.uint8(filteredImg)
    return filteredImg

def load_calib_dict(dir_path):
    """ """
    d = {}
    with open(dir_path + "calib.txt") as f:
        for line in f:
            (key, val) = line.split("=")
            d[key] = val.replace("\n","")
    return d

def compare_ground_truth(disparity_map, folder):
    """ """
    print('Comparing to ground truth')
    ground_truth = cv.imread(f'{folder}disp0.pfm',-1)

    ground_truth[ground_truth==np.inf] = 0
    ground_truth = ground_truth / 256

    diff = np.abs(disparity_map - ground_truth)
    total_pixels = disparity_map.shape[0] * disparity_map.shape[1]
    bad2 = np.count_nonzero(diff > 2) / total_pixels
    return bad2

def make_depth_map(disparity_map, folder, calib_dict):
    """Z = baseline * f / (d + doffs)"""
    print('Computing depth map')
    doffs = float(calib_dict['doffs'])
    baseline = float(calib_dict['baseline'])
    f = float(calib_dict['cam0'].split(' ')[0].replace('[',""))

    depth_map = disparity_map + doffs
    depth_map = baseline * f / depth_map

    fig, ax = plt.subplots(figsize=(20, 10))
    fig.axes[0].get_xaxis().set_visible(False)
    fig.axes[0].get_yaxis().set_visible(False)
    plt.axis('off')
    
    hot = cm.hot
    m = cm.ScalarMappable(cmap=hot)
    m.set_array([depth_map.min(),depth_map.max()])
    plt.colorbar(m, ax=ax)
    plt.imshow(depth_map, 'hot')
    
    fig.savefig(f'{folder}profundidade.png')
    print(f'depth map saved at {folder}profundidade.png')

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