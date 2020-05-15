# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 12:01:28 2020

@author: satrf
"""

import cv2 
import numpy as np
from os.path import join
import matplotlib.pyplot as plt
import math
from scipy import sqrt, pi, arctan2

    

def cal_distance(x2, y2, x1, y1):

    return  math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


        
def capture_sparse_optical_flow(path_in, name_of_sub_dir, grids, time_interval = 5, X_BLOCK_SIZE = 30, Y_BLOCK_SIZE = 30,  if_show = False):
    '''
    Objective: capture optical flow points from video
    input:
        path_in: dir of training data, e.g './data/train'
        name_of_sub_dir: in the training data, there are different days, e.g
                        '/data/train/train001~train034'. 'train001' is the sub diretory
        grids: the corner for each grid created by "make_grid" function
        if_show: a boolin defines if to show the video with optical flow and grid
    Output:
        an n by 4 array [starting x, starting y, magnitude, orientation]
    
    '''
    #%%
    # Parameters for Shi-Tomasi corner detection
    feature_params = dict(maxCorners = 250, qualityLevel = 0.001, minDistance = 1, blockSize = 3)
    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(winSize = (60, 60), maxLevel = 3, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.05))
    # The video feed is read in as a VideoCapture object
    cap = cv2.VideoCapture(join(path_in, name_of_sub_dir + '.avi'))
    # Variable for color to draw optical flow track
    color = (0, 255, 255)
    # ret = a boolean return value from getting the frame, first_frame = the first frame in the entire video sequence
    ret, frame = cap.read()
    # Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Finds the strongest corners in the first frame by Shi-Tomasi method - we will track the optical flow for these corners
    # https://docs.opencv.org/3.0-beta/modules/imgproc/doc/feature_detection.html#goodfeaturestotrack
    
    _, prev_binary = cv2.threshold(prev_gray, 127, 255, cv2.THRESH_BINARY)
    
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask = None, **feature_params)
    # Creates an image filled with zero intensities with the same dimensions as the frame - for later drawing purposes
    mask = np.zeros_like(frame)
    if if_show == True:
        color_grid = (0, 255,0)
        grid_mask = np.zeros_like(frame)
        num_of_grids = grids.shape[1]
        for g in range(num_of_grids):
            grid_mask = cv2.line(grid_mask, 
                             (grids[0][g]+X_BLOCK_SIZE, grids[1][g]),
                             (grids[0][g], grids[1][g]), color_grid, 1)
            grid_mask = cv2.line(grid_mask, 
                             (grids[0][g], grids[1][g]+Y_BLOCK_SIZE),
                             (grids[0][g], grids[1][g]), color_grid, 1)
    #for counting which frame is being processing for each video
    frame_count = 0
    # 20 frames as a set, so record the which set the flows belong to 
    set_of_frames = 1
    flow_points = []
    while cap.isOpened():
        
        # ret = a boolean return value from getting the frame, frame = the current frame being projected in the video
        # 這個.read()一次就換下一張video 的frame
        ret, frame = cap.read()
        if not ret:
            break
        # Converts each frame to grayscale - we previously only converted the first frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # Calculates sparse optical flow by Lucas-Kanade method
        # https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowpyrlk
        next_pts, status, error = cv2.calcOpticalFlowPyrLK(prev_binary, binary, prev_pts, None, **lk_params)
        '''
        if 0 in status:
            print('number of interested corners changed')
            print("previous good corners: ", prev_pts.shape[0])
            print("good corners after calcOpticalFlowPyrLK: ", status.shape[0])
        '''
           
        # Selects good feature points for previous position
        good_old = prev_pts[status == 1]
    
        # Selects good feature points for next position
        good_new = next_pts[status == 1]
    
        # Draws the optical flow tracks
        '''
        plt.scatter(good_new[:,0], good_new[:,1])
        plt.gca().invert_yaxis()
        '''
        
    
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            # Returns a contiguous flattened array as (x, y) coordinates for new point
            x_new, y_new = np.float32(np.clip(new.ravel(),0 ,[frame.shape[1], frame.shape[0]]))
            # Returns a contiguous flattened array as (x, y) coordinates for old point
            x_old, y_old = np.float32(np.clip(old.ravel(),0 ,[frame.shape[1], frame.shape[0]]))
            if cal_distance(x_new,y_new,x_old,y_old) > 0.4:
                # Draws line between new and old position with green color and 2 thickness
                mask = cv2.line(mask, (x_new, y_new), (x_old, y_old), color, 1)
            # Draws filled circle (thickness of -1) at new position with green color and radius of 3
                magnitude = sqrt((x_new-x_old)**2 + (y_new-y_old)**2)
                orientation = categorize_orientation(x_new, y_new, x_old, y_old)
                which_grid = categorize_flow_points(x_old, y_old, grids, X_BLOCK_SIZE, Y_BLOCK_SIZE)
                flow_points.append([x_old, y_old, magnitude, orientation, which_grid, set_of_frames])
            
            
        frame_count += 1
        if (frame_count+1) % time_interval == 0:
            set_of_frames +=1

        # Updates previous frame
        prev_binary = binary.copy()
        # Updates previous good feature points
        prev_pts = good_new.reshape(-1, 1, 2)
  
      
        if if_show == True:    
                
            # Overlays the optical flow tracks on the original frame
            output = cv2.add(frame, mask)
            output = cv2.add(output, grid_mask)
            # Opens a new window and displays the output frame
            cv2.imshow(name_of_sub_dir, output)
            # Frames are read by intervals of 10 milliseconds. The programs breaks out of the while loop when the user presses the 'q' key
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    # The following frees up resources and closes all windows
    cap.release()
    cv2.destroyAllWindows()
    #%% 
    return np.array(flow_points)



def make_grid(first_frame, X_BLOCK_SIZE, Y_BLOCK_SIZE):
    
    height, width = first_frame.shape[0], first_frame.shape[1]
    
    grids = [[],[],[]]
    num_grids = 1
    for x in range(int((width + X_BLOCK_SIZE)/ X_BLOCK_SIZE)):
        for y in range(int(height / Y_BLOCK_SIZE)):
            grids[0].append(x*X_BLOCK_SIZE) 
            grids[1].append(y*Y_BLOCK_SIZE)
            grids[2].append(num_grids)
            num_grids +=1
    return np.array(grids)


def categorize_flow_points(x_old, y_old, grids, X_BLOCK_SIZE, Y_BLOCK_SIZE):
    
    num_y_grids_per_col = len(set(grids[1]))

    x_grid = int(x_old/X_BLOCK_SIZE)
    y_grid = int(y_old/Y_BLOCK_SIZE)
    grid_index = x_grid * num_y_grids_per_col + y_grid
    which_grid = grids[2][grid_index-1] 

    
    return which_grid



def dist_of_grids(grids_to_plot, classified_points):
    
    
    fig, axs = plt.subplots(2, len(grids_to_plot), figsize = (14,8))

    for i, grid in enumerate(grids_to_plot):
        in_this_grid = np.where(classified_points[:,4]==grid)
        axs[0, i].hist(classified_points[in_this_grid][:,2])
        axs[0, i].set_title('Manitude of Grid'+ str(grid))
        axs[1, i].hist(classified_points[in_this_grid][:,3])
        axs[1, i].set_title('Direction of Grid'+ str(grid))

def categorize_orientation(x_new, y_new, x_old, y_old):
    theta = arctan2((y_new-y_old), (x_new-x_old)) * (180 / pi) % 180
    if (y_new-y_old) >0:
        if theta > 45 and theta <= 135:
            return 4
        else:
            return 3
    else:
        if theta > 45 and theta <= 135:
            return 1
        else:
            return 2

'''
#可以用來找最近的點
def search_grid(point, grids):
    
    grids = list(set(grids))
    grids.sort()
    closest_point = min(grids, key=lambda x:abs(x-point))

    if point < closest_point:
        return grids[grids.index(closest_point)-1]
    else:
        return grids[grids.index(closest_point)]
'''
#%%
def capture_dense_optical_flow(path_in, name_of_sub_dir, grids, time_interval = 5,
                               X_BLOCK_SIZE = 30, Y_BLOCK_SIZE = 30,  
                               motion_threshold = 0.75, if_show = False):
    '''
    Objective: capture optical flow points from video
    input:
        path_in: dir of training data, e.g './data/train'
        name_of_sub_dir: in the training data, there are different days, e.g
                        '/data/train/train001~train034'. 'train001' is the sub diretory
        grids: the corner for each grid created by "make_grid" function
        if_show: a boolin defines if to show the video with optical flow and grid
    Output:
        an n by 4 array [starting x, starting y, magnitude, orientation]
    
    '''
    
    #set the Farneback parameters
    fb_params = dict(pyr_scale  = 0.03, levels  = 1, winsize  = 4,
                 iterations = 3, poly_n = 5, poly_sigma  = 1.2, flags = 0)    # The video feed is read in as a VideoCapture object
    cap = cv2.VideoCapture(join(path_in, name_of_sub_dir + '.avi'))

    # ret = a boolean return value from getting the frame, first_frame = the first frame in the entire video sequence
    ret, frame = cap.read()
    # Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    # Creates an image filled with zero intensities with the same dimensions as the frame
    mask = np.zeros_like(frame)
    # Sets image saturation to maximum
    mask[..., 1] = 255

    if if_show == True:
        color_grid = (0, 255,0)
        grid_mask = np.zeros_like(frame)
        num_of_grids = grids.shape[1]
        for g in range(num_of_grids):
            grid_mask = cv2.line(grid_mask, 
                             (grids[0][g]+X_BLOCK_SIZE, grids[1][g]),
                             (grids[0][g], grids[1][g]), color_grid, 1)
            grid_mask = cv2.line(grid_mask, 
                             (grids[0][g], grids[1][g]+Y_BLOCK_SIZE),
                             (grids[0][g], grids[1][g]), color_grid, 1)
    #for counting which frame is being processing for each video
    frame_count = 0
    # 20 frames as a set, so record the which set the flows belong to 
    set_of_frames = 1
    flow_points = []
    while cap.isOpened():
        
        # ret = a boolean return value from getting the frame, frame = the current frame being projected in the video
        # 這個.read()一次就換下一張video 的frame
        ret, frame = cap.read()
        if not ret:
            break

        # Converts each frame to grayscale - we previously only converted the first frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Calculates dense optical flow by Farneback method
        # https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowfarneback
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, **fb_params)
        # Computes the magnitude and angle of the 2D vectors
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=0)
        magnitude = np.array([m if m > motion_threshold else 0 for m in magnitude.reshape(-1)]).reshape(angle.shape[0], angle.shape[1])
        # Sets image hue according to the optical flow direction
        mask[..., 0] = angle * 180 / np.pi / 2
        # Sets image value according to the optical flow magnitude (normalized)
        mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        # Converts HSV to RGB (BGR) color representation
        rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
        # Opens a new window and displays the output frame
        
        # Updates previous frame
        prev_gray = gray.copy()
        
        for y in range(magnitude.shape[0]):
            for x in range(magnitude.shape[1]):
                if magnitude[y][x] > motion_threshold:
                    orientation = divide_angle_to_four_by_weird(angle[y][x])
#                    orientation = categorize_by_bin(mask[..., 0][y][x], 0, 180, 20)
                    which_grid = categorize_flow_points(x, y, grids, X_BLOCK_SIZE, Y_BLOCK_SIZE)
                    flow_points.append([x, y, magnitude[y][x], orientation, which_grid, set_of_frames])
        
        frame_count += 1
        if (frame_count+1) % time_interval == 0:
            set_of_frames +=1

  
      
        if if_show == True:    
            # Opens a new window and displays the input frame
            cv2.imshow("input", frame)
   
            # Overlays the optical flow tracks on the original frame
            output = cv2.add(rgb, grid_mask)
            # Opens a new window and displays the output frame
            cv2.imshow("dense optical flow" + name_of_sub_dir , output) 

            # Frames are read by intervals of 10 milliseconds. The programs breaks out of the while loop when the user presses the 'q' key
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    # The following frees up resources and closes all windows
    cap.release()
    cv2.destroyAllWindows()
    return np.array(flow_points)

def categorize_by_bin(angle, low, high, interval, if_right = False):
    
    bins = np.linspace(low, high, int(((high-low) / interval)+1))
    
    return np.digitize(angle, bins, right=if_right)
    
def divide_angle_to_four_by_weird(angle):
    
    if 0 < np.rad2deg(angle) <=45 or 135 < np.rad2deg(angle) <= 180:
        return 3
    elif 45 < np.rad2deg(angle) <= 135:
        return 4
    elif 225 < np.rad2deg(angle) <= 315:
        return 1
    else:
        return 2
    