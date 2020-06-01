# -*- coding: utf-8 -*-
"""
Created on Wed May  6 11:41:08 2020

@author: satrf
"""
import cv2 
import numpy as np
from os.path import join
import matplotlib.pyplot as plt
import math
from scipy import sqrt, pi, arctan2
import os
import fnmatch

RESIZE_X, RESIZE_Y = 357, 237


def cal_distance(x2, y2, x1, y1):

    return  math.sqrt((x2 - x1)**2 + (y2 - y1)**2)



def visualize_dense_anomalous_gird(path_in, name_of_videos, grids, y_pred, video_info_test,
                                   path_out = None, 
                                   test_item = None, UCSD_to_use = None, 
                                   X_BLOCK_SIZE=30, Y_BLOCK_SIZE=30, if_make_video = False):
    
    
    cap = cv2.VideoCapture(join(path_in, name_of_videos + '.avi'))
    ret, frame = cap.read()
    
    
    
    ####
    # calculate different thresholds for different rows of magnitude to discard
    numbers_y_grid = int(frame.shape[0]/Y_BLOCK_SIZE)
    decreasing_rate = 0.88
    original_threshold = 1.2
    motion_threshold = [original_threshold]
    for row in range(1, numbers_y_grid):
        motion_threshold.insert(0, motion_threshold[0]*decreasing_rate)
        if row%2 ==0:
            decreasing_rate -= .07
    ####

    
    #set the Farneback parameters
    fb_params = dict(pyr_scale  = 0.03, levels  = 1, winsize  = 7,
                 iterations = 3, poly_n = 7, poly_sigma  = 1.5, flags = 0) 
    
    # Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    # Creates an image filled with zero intensities with the same dimensions as the frame
    mask = np.zeros_like(frame)
    # Sets image saturation to maximum
    mask[..., 1] = 255
    
    # creating the green grid
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
    
    
    pred_anomaly = video_info_test.iloc[np.where(y_pred==1)[0]]
    
    which_video = int(name_of_videos[-3:])
    anomaly_this_video = pred_anomaly.loc[pred_anomaly['video']==which_video]
    # for showing the anomalous grid, if the frame and time meet the anomalous grid, then show it 
    frame_count = 1
    time_interval = 1
    
    # for making video with anomaly grid
    frame_array_output = []
    
    # !!! the tuple(x, y), x should be the width, y should be the height
        # whereas in opencv it is reversed
    size = (frame.shape[1], frame.shape[0])
    
    while cap.isOpened():
        
        
        ret, frame = cap.read()
        
        frame_count +=1
        if frame_count%5==0:
            time_interval +=1
            
        if not ret:
            break
        
        # for dense optical visualization
        # Converts each frame to grayscale - we previously only converted the first frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Calculates dense optical flow by Farneback method
        # https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowfarneback
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, **fb_params)
        # Computes the magnitude and angle of the 2D vectors
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=0)
        #scale the magnitude
        for row in range(numbers_y_grid-1):
#            for row in range(int(mask[..., 2].shape[0]/Y_BLOCK_SIZE)):
           thr_idx = magnitude[Y_BLOCK_SIZE*row : Y_BLOCK_SIZE*(row+1), :] < motion_threshold[row]
           magnitude[Y_BLOCK_SIZE*row : Y_BLOCK_SIZE*(row+1), :][thr_idx] = 0  

        thr_idx = magnitude[Y_BLOCK_SIZE*(row+1) : , :] < motion_threshold[-1]
        magnitude[Y_BLOCK_SIZE*(row+1) : , :][thr_idx] = 0
        
        
        # Sets image hue according to the optical flow direction
        mask[..., 0] = angle * 180 / np.pi / 2

        # Sets image value according to the optical flow magnitude (normalized)
        mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        # Converts HSV to RGB (BGR) color representation
        rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
        
        # Updates previous frame
        prev_gray = gray
 ##############################       
        
        
        anomaly_mask = np.zeros_like(frame)

        anomaly_color_grid = (0 ,0 , 255)
        
        
        if time_interval in np.array(anomaly_this_video['time_interval']):
            grids_of_anomaly = np.array(anomaly_this_video.loc[anomaly_this_video['time_interval']==time_interval]['grid'])
            
            
            for g in grids_of_anomaly:
                anomaly_mask = cv2.line(anomaly_mask, 
                                 (grids[0][g-1]+X_BLOCK_SIZE, grids[1][g-1]),
                                 (grids[0][g-1], grids[1][g-1]), anomaly_color_grid, 2)
                anomaly_mask = cv2.line(anomaly_mask, 
                                 (grids[0][g-1], grids[1][g-1]+Y_BLOCK_SIZE),
                                 (grids[0][g-1], grids[1][g-1]), anomaly_color_grid, 2)
                anomaly_mask = cv2.line(anomaly_mask, 
                                 (grids[0][g-1], grids[1][g-1]+Y_BLOCK_SIZE),
                                 (grids[0][g-1]+X_BLOCK_SIZE, grids[1][g-1]+Y_BLOCK_SIZE), anomaly_color_grid, 2)
       
                anomaly_mask = cv2.line(anomaly_mask, 
                                 (grids[0][g-1]+X_BLOCK_SIZE, grids[1][g-1]),
                                 (grids[0][g-1]+X_BLOCK_SIZE, grids[1][g-1]+Y_BLOCK_SIZE), anomaly_color_grid, 2)
        else:  
            anomaly_mask = np.zeros_like(frame)
        # Overlays the optical flow tracks on the original frame
        output = cv2.add(frame, grid_mask)
        output = cv2.add(output, anomaly_mask)
        resized_output = cv2.resize(output, (RESIZE_X,RESIZE_Y), interpolation = cv2.INTER_AREA)
        resized_rgb = cv2.resize(rgb, (RESIZE_X,RESIZE_Y), interpolation = cv2.INTER_AREA)
        
        # Opens a new window and displays the output frame
        cv2.imshow(name_of_videos, resized_output)
        cv2.imshow('mask '+name_of_videos, resized_rgb)
        # Frames are read by intervals of 10 milliseconds. The programs breaks out of the while loop when the user presses the 'q' key
        
        # for making video
        if if_make_video:
            frame_array_output.append(output)

            
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        
    if if_make_video:
        file_name = os.path.join(path_out, name_of_videos+'_'+UCSD_to_use+test_item+'.avi')
        out = cv2.VideoWriter(file_name,cv2.VideoWriter_fourcc(*'DIVX'), 20, size)
        for i in range(len(frame_array_output)):
            # writing to a image array
            out.write(frame_array_output[i])
        out.release()
    # The following frees up resources and closes all windows
    cap.release()
    cv2.destroyAllWindows()
    
    

def visualize_scaled_motion(path_in, name_of_sub_dir, grids,
                            X_BLOCK_SIZE , Y_BLOCK_SIZE, original_threshold=0):
    
    

    
    cap = cv2.VideoCapture(join(path_in, name_of_sub_dir + '.avi'))
    ret, frame = cap.read()
    
    #set the Farneback parameters
    fb_params = dict(pyr_scale  = 0.03, levels  = 1, winsize  = 7,
                 iterations = 3, poly_n = 7, poly_sigma  = 1.5, flags = 0) 
    
    # Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    # Creates an image filled with zero intensities with the same dimensions as the frame
    mask = np.zeros_like(frame)
    # Sets image saturation to maximum
    mask[..., 1] = 255
    
    # for visualising grid
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
        
    # calculate different thresholds for different rows of magnitude to discard
    numbers_y_grid = int(frame.shape[0]/Y_BLOCK_SIZE)
    decreasing_rate = 0.5
    
    motion_threshold = [original_threshold]
    for row in range(1, numbers_y_grid):
        motion_threshold.insert(0, motion_threshold[0]*decreasing_rate)
        if row%2 ==0:
            decreasing_rate -= 0.1

    
    while cap.isOpened():
        
        
        ret, frame = cap.read()
  

        if not ret:
            break
        
        # for dense optical visualization
        # Converts each frame to grayscale - we previously only converted the first frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Calculates dense optical flow by Farneback method
        # https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowfarneback
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, **fb_params)
        # Computes the magnitude and angle of the 2D vectors
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=0)
        #scale the magnitude
        for row in range(int(mask[..., 2].shape[0]/Y_BLOCK_SIZE)-1):
#            for row in range(int(mask[..., 2].shape[0]/Y_BLOCK_SIZE)):
           thr_idx = magnitude[Y_BLOCK_SIZE*row : Y_BLOCK_SIZE*(row+1), :] < motion_threshold[row]
           magnitude[Y_BLOCK_SIZE*row : Y_BLOCK_SIZE*(row+1), :][thr_idx] = 0  

        thr_idx = magnitude[Y_BLOCK_SIZE*(row+1) : , :] < motion_threshold[-1]
        magnitude[Y_BLOCK_SIZE*(row+1) : , :][thr_idx] = 0
        
        # Sets image hue according to the optical flow direction
        mask[..., 0] = angle * 180 / np.pi / 2

        # Sets image value according to the optical flow magnitude (normalized)
        mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        
    # Converts HSV to RGB (BGR) color representation
        rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
        
        # Updates previous frame
        prev_gray = gray
 ##############################              
               
      
        # Overlays the optical flow tracks on the original frame
        output = cv2.add(frame, grid_mask)
        resized_output = cv2.resize(output, (RESIZE_X,RESIZE_Y), interpolation = cv2.INTER_AREA)
        resized_rgb = cv2.resize(rgb, (RESIZE_X,RESIZE_Y), interpolation = cv2.INTER_AREA)

        # Opens a new window and displays the output frame
        cv2.imshow(name_of_sub_dir, resized_output)
        cv2.imshow('mask '+name_of_sub_dir, resized_rgb)
        # Frames are read by intervals of 10 milliseconds. The programs breaks out of the while loop when the user presses the 'q' key
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    # The following frees up resources and closes all windows
    cap.release()
    cv2.destroyAllWindows()


#%% for sparse-flow, not use in this case
def visualize_sparse_anomalous_gird(path_in, name_of_sub_dir, grids, y_test, y_pred , video_info_test, X_BLOCK_SIZE = 30, Y_BLOCK_SIZE = 30):
    
    cap = cv2.VideoCapture(join(path_in, name_of_sub_dir + '.avi'))
    ret, frame = cap.read()
    
    # Parameters for Shi-Tomasi corner detection
    feature_params = dict(maxCorners = 250, qualityLevel = 0.001, minDistance = 1, blockSize = 3)
    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(winSize = (60, 60), maxLevel = 3, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.05))

    # Variable for color to draw optical flow track
    color = (0, 255, 255)
    # Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, prev_binary = cv2.threshold(prev_gray, 127, 255, cv2.THRESH_BINARY)
    # Finds the strongest corners in the first frame by Shi-Tomasi method - we will track the optical flow for these corners
    # https://docs.opencv.org/3.0-beta/modules/imgproc/doc/feature_detection.html#goodfeaturestotrack
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask = None, **feature_params)
    # Creates an image filled with zero intensities with the same dimensions as the frame - for later drawing purposes
    mask = np.zeros_like(frame)
    
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
    
#    true_anomaly = video_info_test.iloc[np.where(y_test==1)[0]]
    pred_anomaly = video_info_test.iloc[np.where(y_pred==1)[0]]
    
    which_video = int(name_of_sub_dir[-3:])
    anomaly_this_video = pred_anomaly.loc[pred_anomaly['video']==which_video]
    frame_count = 1
    time_interval = 1
    
    
    fgbg2 = cv2.createBackgroundSubtractorMOG2(); 
    
    while cap.isOpened():
        
        
        ret, frame = cap.read()
        
        frame_count +=1
        if frame_count%5==0:
            time_interval +=1
            
        if not ret:
            break
        
        fgmask2 = fgbg2.apply(frame)
        edged=cv2.Canny(frame,30,200)
        # for sparse optical visualization
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
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            # Returns a contiguous flattened array as (x, y) coordinates for new point
            x_new, y_new = np.float32(np.clip(new.ravel(),0 ,[frame.shape[1], frame.shape[0]]))
            # Returns a contiguous flattened array as (x, y) coordinates for old point
            x_old, y_old = np.float32(np.clip(old.ravel(),0 ,[frame.shape[1], frame.shape[0]]))
            if cal_distance(x_new,y_new,x_old,y_old) > 0.4:
                # Draws line between new and old position with green color and 2 thickness
                mask = cv2.line(mask, (x_new, y_new), (x_old, y_old), color, 1)


        # Updates previous frame
        prev_binary = binary.copy()
        # Updates previous good feature points
        prev_pts = good_new.reshape(-1, 1, 2)
 ##############################       
        
        
        anomaly_mask = np.zeros_like(frame)

        anomaly_color_grid = (0 ,0 , 255)
        
        
        if time_interval in np.array(anomaly_this_video['time_interval']):
            grids_of_anomaly = np.array(anomaly_this_video.loc[anomaly_this_video['time_interval']==time_interval]['grid'])
            
            
            for g in grids_of_anomaly:
                anomaly_mask = cv2.line(anomaly_mask, 
                                 (grids[0][g-1]+X_BLOCK_SIZE, grids[1][g-1]),
                                 (grids[0][g-1], grids[1][g-1]), anomaly_color_grid, 2)
                anomaly_mask = cv2.line(anomaly_mask, 
                                 (grids[0][g-1], grids[1][g-1]+Y_BLOCK_SIZE),
                                 (grids[0][g-1], grids[1][g-1]), anomaly_color_grid, 2)
                anomaly_mask = cv2.line(anomaly_mask, 
                                 (grids[0][g-1], grids[1][g-1]+Y_BLOCK_SIZE),
                                 (grids[0][g-1]+X_BLOCK_SIZE, grids[1][g-1]+Y_BLOCK_SIZE), anomaly_color_grid, 2)
       
                anomaly_mask = cv2.line(anomaly_mask, 
                                 (grids[0][g-1]+X_BLOCK_SIZE, grids[1][g-1]),
                                 (grids[0][g-1]+X_BLOCK_SIZE, grids[1][g-1]+Y_BLOCK_SIZE), anomaly_color_grid, 2)
        else:  
            anomaly_mask = np.zeros_like(frame)
        # Overlays the optical flow tracks on the original frame
        output = cv2.add(frame, grid_mask)
#        output = cv2.add(output, anomaly_mask)
        output = cv2.add(output, mask)
        
        #resize
        resized_output = cv2.resize(output, (RESIZE_X,RESIZE_Y), interpolation = cv2.INTER_AREA)
        resized_binary = cv2.resize(binary, (RESIZE_X,RESIZE_Y), interpolation = cv2.INTER_AREA)
        resized_edged = cv2.resize(edged, (RESIZE_X,RESIZE_Y), interpolation = cv2.INTER_AREA)
        resized_fgmask2 = cv2.resize(fgmask2, (RESIZE_X,RESIZE_Y), interpolation = cv2.INTER_AREA)
        # Opens a new window and displays the output frame
        cv2.imshow(name_of_sub_dir, resized_output)
        cv2.imshow('binary', resized_binary)
        cv2.imshow('canny edges',resized_edged)
        cv2.imshow('MOG2', resized_fgmask2)

        # Frames are read by intervals of 10 milliseconds. The programs breaks out of the while loop when the user presses the 'q' key
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    # The following frees up resources and closes all windows
    cap.release()
    cv2.destroyAllWindows()