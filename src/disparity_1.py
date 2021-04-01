import numpy as np
import cv2 as cv
from matplotlib import cm
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import glob

DATA_DIR = '../data/'
MIDDLEBURY_DIR = DATA_DIR + "Middlebury/"

def load_images(dir_path):
    """ """
    imgL = cv.imread(dir_path + 'im0.png')
    imgR = cv.imread(dir_path + 'im1.png')
    return imgL, imgR

def preprocess_images(imgL, imgR):#calib_dict
    imgL = cv.copyMakeBorder(imgL,0,0,288,0,cv.BORDER_CONSTANT)
    imgR = cv.copyMakeBorder(imgR,0,0,288,0,cv.BORDER_CONSTANT)
    return imgL, imgR

def postprocess_disparity_map(filteredImg):#calib_dict
    filteredImg = filteredImg[:,288:]
    return filteredImg

def compute_disparity(imgL, imgR):#calib_dict
    """   """
    window_size = 3
    
    left_matcher = cv.StereoSGBM_create(
        minDisparity=0,
        numDisparities=288,             
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

def apply_filter(displ, imgL, dispr, left_matcher):
    """  """
    lmbda = 160000
    sigma = 1.8

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
    d = {}
    with open(dir_path + "calib.txt") as f:
        for line in f:
            (key, val) = line.split("=")
            d[key] = val.replace("\n","")
    return d

def compare_ground_truth(disparity_map, folder):
    print('Comparing to ground truth')
    ground_truth = cv.imread(f'{folder}disp0.pfm',-1)

    ground_truth[ground_truth==np.inf] = 0
    ground_truth = ground_truth / 256

    diff = np.abs(disparity_map - ground_truth)
    total_pixels = disparity_map.shape[0] * disparity_map.shape[1]
    bad2 = np.count_nonzero(diff > 2) / total_pixels
    return bad2

def make_depth_map(disparity_map, folder):#calib_dict
    print('Computing depth map')
    depth_map = disparity_map + 100.279
    depth_map = 193.006 * 2329.558 / depth_map

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
        imgL, imgR = preprocess_images(imgL, imgR)

        displ, dispr, left_matcher = compute_disparity(imgL, imgR)
        disparity_map = apply_filter(displ, imgL, dispr, left_matcher)
        disparity_map = postprocess_disparity_map(disparity_map)

        cv.imwrite(folder +"disparidade.pgm",disparity_map)
        print("File disparidade.pgm saved on current folder")

        bad2 = compare_ground_truth(disparity_map, folder)
        print(f'bad 2.0 for image {folder}: {round(bad2,2)}')

        make_depth_map(disparity_map, folder)
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