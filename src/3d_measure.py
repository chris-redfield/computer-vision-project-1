import os
import cv2 as cv
import numpy as np

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'data/'))
FURUKAWA_DIR = DATA_DIR + "/FurukawaPonce/"

points_pair_left = []
points_pair_right = []

def process_distance(points_pair_left,points_pair_right):
    distance = triangulate_points(points_pair_left, points_pair_right)
    
    #divides by 30mm, according to original dataset docs
    distance = distance / 30
    distance = round(distance,2)

    mean_x_left = int( (points_pair_left[0][0] + points_pair_left[1][0]) / 2 )
    mean_y_left = int( (points_pair_left[0][1] + points_pair_left[1][1]) / 2 )
    mean_x_right = int( (points_pair_right[0][0] + points_pair_right[1][0]) / 2 )
    mean_y_right = int( (points_pair_right[0][1] + points_pair_right[1][1]) / 2 )

    cv.putText(img1, str(distance),(mean_x_left,mean_y_left),cv.FONT_HERSHEY_SIMPLEX,1.5, (0,0,0),3)
    cv.putText(img2, str(distance),(mean_x_right,mean_y_right),cv.FONT_HERSHEY_SIMPLEX,1.5, (0,0,0),3)

def draw_line_img1(event,x,y,flags,param):
    global points_pair_left, points_pair_right

    if event == cv.EVENT_LBUTTONDBLCLK:
        cv.circle(img1,(x,y),5,(255,0,0),-1)
        points_pair_left.append([x,y])
        
        if(len(points_pair_left)>1):
            print("Img1:", points_pair_left)
            cv.line(img1, (points_pair_left[0][0],points_pair_left[0][1]),
                (points_pair_left[1][0],points_pair_left[1][1]), (255,0,0), 5)

            if(len(points_pair_right)>1):
                process_distance(points_pair_left,points_pair_right)
                points_pair_left, points_pair_right=[],[]

def draw_line_img2(event,x,y,flags,param):
    global points_pair_left, points_pair_right

    if event == cv.EVENT_LBUTTONDBLCLK:
        cv.circle(img2,(x,y),5,(255,0,0),-1)
        points_pair_right.append([x,y])
        
        if(len(points_pair_right)>1):
            print("Img2:", points_pair_right)
            cv.line(img2, (points_pair_right[0][0],points_pair_right[0][1]),
             (points_pair_right[1][0],points_pair_right[1][1]), (255,0,0), 5)

            if(len(points_pair_left)>1):
                process_distance(points_pair_left,points_pair_right)                
                points_pair_left, points_pair_right=[],[]

def triangulate_points(point_pair_left, point_pair_right):

    #print(point_pair_left, point_pair_right)
    point_pair_left = np.array(point_pair_left).astype(float)
    point_pair_right = np.array(point_pair_right).astype(float)

    mtx_1 = [ 
    [4265.08, 5195.24, -565.379, -1.37072e+06],
    [1636.7, -1954.5, -6218.58, 2.75156e+06],
    [-0.645659, 0.624306, -0.439734, 2977.41] ]
    
    mtx_2 = [
    [2567.28, 6197.89, -641.912, -1.26828e+06],
    [1969.58, -1410.57, -6237.08, 2.45107e+06],
    [-0.803929, 0.431864, -0.40889, 3242.75] ]

    mtx_1 = np.array(mtx_1)
    mtx_2 = np.array(mtx_2)

    points_3D = cv.triangulatePoints(mtx_1, mtx_2, point_pair_left.T, point_pair_right.T)
    #print(points_3D)

    point_3D_start = points_3D.T[0]
    point_3D_end = points_3D.T[1]

    point_3D_start = point_3D_start / point_3D_start[3]
    point_3D_end = point_3D_end / point_3D_end[3]

    #print(point_3D_start,point_3D_end)

    euclidean_dist_3d = np.linalg.norm(point_3D_start - point_3D_end)
    print("Euclidean distance between the points:", euclidean_dist_3d)
    return euclidean_dist_3d


def main():
    
    global img1, img2

    cv.namedWindow("img1", cv.WINDOW_NORMAL)
    cv.namedWindow("img2", cv.WINDOW_NORMAL)

    img1 = cv.imread(FURUKAWA_DIR + 'MorpheusL.jpg')
    img2 = cv.imread(FURUKAWA_DIR + 'MorpheusR.jpg')
    
    cv.setMouseCallback('img1',draw_line_img1)
    cv.setMouseCallback('img2',draw_line_img2)

    while(True):
        cv.imshow('img1', img1) 
        cv.imshow('img2', img2) 
        
        if cv.waitKey(1) & 0xFF == ord('q'):
            break



if __name__ == "__main__":
    main()