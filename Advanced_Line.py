import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from collections import deque
from Line import Line


#Step1:Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.


# Make a list of calibration images
images = glob.glob('./camera_cal/calibration*.jpg')

# Step through the list and search for chessboard corners
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (9,6), corners, ret)


#Step2:Apply a distortion correction to raw images.
def cal_undistort(img, objpoints, imgpoints):
    # Use cv2.calibrateCamera() and cv2.undistort()
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints,imgpoints,gray.shape[::-1],None,None)
    undist = cv2.undistort(img, mtx, dist, None, mtx) 
    return undist

#step3:Use color transforms, gradients, etc., to create a thresholded binary image.
def color_grandient(img, s_thresh=(100, 255), v_thresh = (50,255),sx_thresh=(50, 100)):
    # Convert to HLS color space and separate the L/S channel
    # Note: img is the undistorted image
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    # Threshold color s channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Threshold color v channel
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    v_channel = hsv[:,:,2]
    v_binary = np.zeros_like(v_channel)
    v_binary[(v_channel >= v_thresh[0]) & (v_channel <= v_thresh[1])] = 1
    
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[((s_binary == 1) & (v_binary ==1 ) | (sxbinary == 1))] = 1

    return combined_binary


def warper(img, src, dst):

    # Compute and apply perpective transform
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image

    return warped,M,Minv

def drawing(warped,undist,ploty,left_fitx,right_fitx,Minv,offset,left_curverad,right_curverad):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    window_width=25
    yvals= range(0,warped.shape[0])

    # Recast the x and y points into usable format for cv2.fillPoly()
    inter_area = np.array(list(zip(np.concatenate((left_fitx+window_width/2,right_fitx[::-1]-window_width/2),axis=0),np.concatenate((yvals,yvals[::-1]),axis=0))),np.int32)
    # Put dots of lane lines together to form a poly
    left_lane = np.array(list(zip(np.concatenate((left_fitx-window_width/2,left_fitx[::-1]+window_width/2),axis=0),np.concatenate((yvals,yvals[::-1]),axis=0))),np.int32)
    right_lane = np.array(list(zip(np.concatenate((right_fitx-window_width/2,right_fitx[::-1]+window_width/2),axis=0),np.concatenate((yvals,yvals[::-1]),axis=0))),np.int32)
    
    road = np.zeros_like(color_warp)
    road_bg = np.zeros_like(color_warp)

    cv2.fillPoly(road,[left_lane],color=[255,0,0])
    cv2.fillPoly(road,[right_lane],color=[0,0,255])

    # Draw the lane onto the warped blank image
    cv2.fillPoly(road, np.int_([inter_area]), (0,255, 0))

    cv2.fillPoly(road_bg,[left_lane],color=[255,255,255])
    cv2.fillPoly(road_bg,[right_lane],color=[255,255,255])

    newwarp = cv2.warpPerspective(road, Minv, (color_warp.shape[1], color_warp.shape[0]))
    newwarp_bg = cv2.warpPerspective(road_bg, Minv, (color_warp.shape[1], color_warp.shape[0]))

    # Combine the result with the original image
    base = cv2.addWeighted(undist, 1, newwarp_bg, -0.6, 0)
    result = cv2.addWeighted(base, 1, newwarp, 0.5, 0)
    

    side_pos = 'left'
    if offset <= 0:
        side_pos = 'right'
    #draw the text 
    cv2.putText(result,'Radius of Curvature = '+str(round(left_curverad,3))+'(m)',(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    cv2.putText(result,'Vehicle is '+str(abs(round(offset,3)))+'m '+side_pos+' of center',(50,100),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    return result

def process_image(image):
    #Undistort the image
    undistorted = cal_undistort(image, objpoints, imgpoints)
    #Apply color and grandient threhold
    combined_binary = color_grandient(undistorted)
    #The four source points
    s1=(image.shape[1]-592,image.shape[0]-270)
    s2=(image.shape[1]-160,image.shape[0]-1)
    s3=(202,image.shape[0]-1)
    s4=(593,image.shape[0]-270)
    #The four desti points
    d1=(960,0)
    d2=(960,image.shape[0]-1)
    d3=(310,image.shape[0]-1)
    d4=(310,0)

    src=np.float32([s1,s2,s3,s4])
    dst=np.float32([d1,d2,d3,d4])
    #warper the binary image 
    binary_warped, M, Minv= warper(combined_binary,src,dst) 
    #Use clss of Line() method slide_fit to find lines
    ploty,left_fitx,right_fitx,left_curverad,right_curverad,offset = lane_line.slide_fit(binary_warped)
    #Draw the results on image
    result = drawing(binary_warped,undistorted,ploty,left_fitx,right_fitx,Minv,offset,left_curverad,right_curverad)
    return result


from moviepy.editor import VideoFileClip
#Create the instance for global
lane_line = Line()
white_output = 'output_video/output1.mp4'
clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(process_image)
white_clip.write_videofile(white_output, audio=False)
