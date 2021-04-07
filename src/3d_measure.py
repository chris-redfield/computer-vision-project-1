import os
import cv2 as cv

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'data/'))
FURUKAWA_DIR = DATA_DIR + "/FurukawaPonce/"

points_pair_left = []
points_pair_right = []

def draw_line_img1(event,x,y,flags,param):
    global points_pair_left

    if event == cv.EVENT_LBUTTONDBLCLK:
        cv.circle(img1,(x,y),5,(255,0,0),-1)
        points_pair_left.append([x,y])
        if(len(points_pair_left)>1):
            print("Img1:", points_pair_left)
            cv.line(img1, (points_pair_left[0][0],points_pair_left[0][1]),
             (points_pair_left[1][0],points_pair_left[1][1]), (255,0,0), 5)
            points_pair_left=[]

def draw_line_img2(event,x,y,flags,param):
    global points_pair_right

    if event == cv.EVENT_LBUTTONDBLCLK:
        cv.circle(img2,(x,y),5,(255,0,0),-1)
        points_pair_right.append([x,y])
        if(len(points_pair_right)>1):
            print("Img2:", points_pair_right)
            cv.line(img2, (points_pair_right[0][0],points_pair_right[0][1]),
             (points_pair_right[1][0],points_pair_right[1][1]), (255,0,0), 5)
            points_pair_right=[]



def main():
    
    global img1, img2

    cv.namedWindow("img1", cv.WINDOW_NORMAL)
    cv.namedWindow("img2", cv.WINDOW_NORMAL)

    img1 = cv.imread(FURUKAWA_DIR + 'MorpheusL.jpg') #queryimage # left image
    img2 = cv.imread(FURUKAWA_DIR + 'MorpheusR.jpg') #trainimage # right image
    
    cv.setMouseCallback('img1',draw_line_img1)
    cv.setMouseCallback('img2',draw_line_img2)

    while(True):
        cv.imshow('img1', img1) 
        cv.imshow('img2', img2) 
        
        if cv.waitKey(1) & 0xFF == ord('q'):
            break



if __name__ == "__main__":
    main()