import numpy as np
import cv2 as cv
from matplotlib import cm
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import glob
import os 
from stereo_disparity import *

#DATA_DIR = '../data/'
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'data/'))
FURUKAWA_DIR = DATA_DIR + "/FurukawaPonce/"

def load_images(dir_path):
    """ """
    img1 = cv.imread(dir_path + 'MorpheusL.jpg')
    img2 = cv.imread(dir_path + 'MorpheusR.jpg')
    print(f'original images dims: {img1.shape, img2.shape}')

    ## Resizes img2 to have the same dimensions as 1
    dim = (1400, 1300)
    img2 = cv.resize(img2, dim, interpolation = cv.INTER_AREA)

    print(f'images dims after reshape: {img1.shape, img2.shape}')

    return img1, img2

def preprocess_images(imgL, imgR):
    """ """
    pixel_offset = 29
    imgL = cv.copyMakeBorder(imgL,0,0,pixel_offset,0,cv.BORDER_CONSTANT)
    imgR = cv.copyMakeBorder(imgR,0,0,pixel_offset,0,cv.BORDER_CONSTANT)
    return imgL, imgR

def postprocess_disparity_map(filteredImg):
    """ """
    pixel_offset = 29
    filteredImg = filteredImg[:,pixel_offset:]
    
    ### adds relu to handle negative values
    relu(filteredImg)
    return filteredImg

def process_point_matching(img1, img2):
    """"""
    sift = cv.SIFT_create()

    print("Finding keypoints...")
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)

    pts1 = []
    pts2 = []
    
    # ratio of closest-distance to second-closest distance as Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.8*n.distance:
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
    
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)

    F, mask = cv.findFundamentalMat(pts1,pts2,cv.FM_LMEDS)

    pts1 = pts1[mask.ravel()==1]
    pts2 = pts2[mask.ravel()==1]

    return pts1, pts2

def rectify_images(img1, img2, pts1, pts2):
    """"""
    print("Finding homography...")
    H, mask = cv.findHomography(pts1, pts2)

    print("Warping perspective...")
    img1_warp = cv.warpPerspective(img1, H, (img1.shape[1], img1.shape[0]))
    img2_warp = cv.warpPerspective(img2, H, (img2.shape[1], img2.shape[0]))
    return img1_warp, img2_warp

def compute_disparity(imgL, imgR):
    """   """
    num_disparities = 29

    window_size = 3
    
    left_matcher = cv.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparities,             
        blockSize=6,
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
    sigma = 1.3

    wls_filter = cv.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)

    print('Applying filter...')
    filteredImg = wls_filter.filter(displ, imgL, None, dispr)
    
    return filteredImg

def make_depth_map(disparity_map, folder):
    """Z = baseline * f / (d + doffs)"""

    t_1 = np.array([ -532.285900 , 207.183600 , 2977.408000 ])
    t_2 = np.array([ -614.549000 , 193.240700 , 3242.754000 ])
    baseline = np.linalg.norm(t_1 - t_2)
    
    principal_point = [ 738.251932, 457.560286 ]
    doffs = principal_point[0] - principal_point[1]

    f = 6704.926882

    depth_map = disparity_map + doffs
    depth_map = baseline * f / depth_map
    depth_map = depth_map/10
    
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
    """"""
    img1, img2 = load_images(FURUKAWA_DIR)

    pts1, pts2 = process_point_matching(img1, img2)

    img1_warp, img2_warp = rectify_images(img1, img2, pts1, pts2)
    #cv.imwrite(FURUKAWA_DIR +"rectified_test.png",img1_warp)

    ## Gets only first channel
    img1_warp = img1_warp[:,:,0]
    img2_warp = img2_warp[:,:,0]

    img1_warp, img2_warp = preprocess_images(img1_warp, img2_warp)

    displ, dispr, left_matcher = compute_disparity(img1_warp, img2_warp)

    disparity_map = apply_filter(displ, img1_warp, dispr, left_matcher)

    disparity_map = postprocess_disparity_map(disparity_map)

    cv.imwrite(FURUKAWA_DIR +"disparidade.pgm",disparity_map)
    print("File disparidade.pgm saved on current folder")

    # plt.imshow(disparity_map, 'gray')
    # plt.show()

    make_depth_map(disparity_map, FURUKAWA_DIR)
    



if __name__ == "__main__":
    main()