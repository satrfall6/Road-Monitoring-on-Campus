# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 12:01:28 2020

@author: satrf
"""

import cv2 
import os 
import numpy as np
from os.path import join, isfile
import matplotlib.pyplot as plt
import math
import fnmatch
    

def cal_distance(x2, y2, x1, y1):

    return  math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

#%%
def capture_dense_optical_flow(path_in, name_of_videos, grids, time_interval = 5,
                               X_BLOCK_SIZE = 30, Y_BLOCK_SIZE = 30,  
                               if_show = False,
                               if_normal_ori = True):
    '''
    Objective: capture optical flow points from video
    input:
        path_in: dir of training video data, e.g './data/train/videos'
        name_of_videos: read the videos in the directory iteratively
                        '/data/train/videos/name_of_videos'. 'train001' 
        grids: the corner for each grid created by "make_grid" function
        if_show: a boolin defines if to show the video with optical flow and grid
    Output:
        an n by 6 array [starting x, starting y, magnitude, orientation, 
                         grid of the flow, time interval of the flow]
    
    '''
    
    #set the Farneback parameters
    fb_params = dict(pyr_scale  = 0.03, levels  = 1, winsize  = 4,
                 iterations = 3, poly_n = 5, poly_sigma  = 1.2, flags = 0)    # The video feed is read in as a VideoCapture object
    cap = cv2.VideoCapture(join(path_in, name_of_videos + '.avi'))

    # ret = a boolean return value from getting the frame, first_frame = the first frame in the entire video sequence
    ret, frame = cap.read()
    # Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    # Creates an image filled with zero intensities with the same dimensions as the frame
    mask = np.zeros_like(frame)
    # Sets image saturation to maximum
    mask[..., 1] = 255


    # calculate different thresholds for different rows of magnitude to discard
    numbers_y_grid = int(frame.shape[0]/Y_BLOCK_SIZE)
    decreasing_rate = 0.88
    original_threshold = 1.2
    motion_threshold = [original_threshold]
    for row in range(1, numbers_y_grid):
        motion_threshold.insert(0, motion_threshold[0]*decreasing_rate)
        if row%2 ==0:
            decreasing_rate -= .07


    # if needs to show the video, portrait the grids
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
        

        # to discard the stationary objects, and use different threshold for different rows
        for row in range(numbers_y_grid-1):
#            for row in range(int(mask[..., 2].shape[0]/Y_BLOCK_SIZE)):
           thr_idx = magnitude[Y_BLOCK_SIZE*row : Y_BLOCK_SIZE*(row+1), :] < motion_threshold[row]
           magnitude[Y_BLOCK_SIZE*row : Y_BLOCK_SIZE*(row+1), :][thr_idx] = 0  
#           if row < 2:
#               magnitude[Y_BLOCK_SIZE*row : Y_BLOCK_SIZE*(row+1), :] = magnitude[Y_BLOCK_SIZE*row : Y_BLOCK_SIZE*(row+1), :] * median_ratio[row]        

        thr_idx = magnitude[Y_BLOCK_SIZE*(row+1) : , :] < motion_threshold[-1] # -1 here same as index row+1
        magnitude[Y_BLOCK_SIZE*(row+1) : , :][thr_idx] = 0
#        magnitude = np.array([m if m > motion_threshold else 0 for m in magnitude.reshape(-1)]).reshape(angle.shape[0], angle.shape[1])
       
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
                if magnitude[y][x] > motion_threshold[0]:
                    if if_normal_ori:
                        orientation = categorize_by_bin(mask[..., 0][y][x], 0, 180, 20)
                    else:
                        orientation = categorize_to_four(angle[y][x])
                    which_grid = categorize_flow_points(x, y, grids, X_BLOCK_SIZE, Y_BLOCK_SIZE)
    
                    flow_points.append([x, y, np.clip(magnitude[y][x],0,15), orientation, 
                                        which_grid, set_of_frames])
        
        frame_count += 1
        if (frame_count+1) % time_interval == 0:
            set_of_frames +=1

  
      
        if if_show == True:    
            # Opens a new window and displays the input frame
            cv2.imshow("input", frame)
   
            # Overlays the optical flow tracks on the original frame
            output = cv2.add(rgb, grid_mask)
            # Opens a new window and displays the output frame
            cv2.imshow("dense optical flow" + name_of_videos , output) 

            # Frames are read by intervals of 10 milliseconds. The programs breaks out of the while loop when the user presses the 'q' key
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    # The following frees up resources and closes all windows
    cap.release()
    cv2.destroyAllWindows()
    return np.array(flow_points)


def make_grid(first_frame, X_BLOCK_SIZE, Y_BLOCK_SIZE):
    
    '''
    Objective: create grids for the frames
    input:
        first_frame: the frame of a video
        X_BLOCK_SIZE: length of x edge
        Y_BLOCK_SIZE: length of y edge

    Output:
        an 3 by n (total number of the grids) array. 
        x, y are the coordinate of the point of the top left corner
        [[x coordinate: ...],
         [y coordinate: ...],
         [grid number: ...]]
    
    '''
    
    height, width = first_frame.shape[0], first_frame.shape[1]
    
    # for storing x, y, grid index;
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
    '''
    Objective: sort the flow points into the corresponding grids
    input:
        x_old: the starting x position of the flow points
        y_old: the starting y position of the flow points
        grids: the pre-calculated grids
    Output:
        which grid this point belongs to
    
    '''
    num_y_grids_per_col = len(set(grids[1]))
    max_grid_idx = grids.shape[1]-1
    x_grid = int(x_old/X_BLOCK_SIZE)
    y_grid = int(y_old/Y_BLOCK_SIZE)
    grid = x_grid * num_y_grids_per_col + y_grid
    which_grid = grids[2][np.clip(grid,0,max_grid_idx)] 

    
    return which_grid



def dist_of_grids(grids_to_plot, classified_points):
    
    
    fig, axs = plt.subplots(2, len(grids_to_plot), figsize = (14,8))

    for i, grid in enumerate(grids_to_plot):
        in_this_grid = np.where(classified_points[:,4]==grid)
        axs[0, i].hist(classified_points[in_this_grid][:,2])
        axs[0, i].set_title('Manitude of Grid'+ str(grid))
        axs[1, i].hist(classified_points[in_this_grid][:,3])
        axs[1, i].set_title('Direction of Grid'+ str(grid))




def categorize_by_bin(angle, low, high, interval, if_right = False):
    
    '''
    Objective: sort the orientation of the flow points to 9 direction (0, 180, 20)
    input:
        angle: the angle in degree
        low: the low boundary
        high: the high boundary
        interval: the interval of the range
        if_right: 
    Output:
        the category of the angle of this point
    
    '''
    
    
    bins = np.linspace(low, high, int(((high-low) / interval)+1))
    
    return np.clip(np.digitize(angle, bins, right=if_right), 1, 9)
    
def categorize_to_four(angle):
    
    '''
    Objective: a paper sort angle to 4 categories, this function follow the way they sort the direction
    input:
        angle: the angle in degree

    Output:
        the category of the angle of this point
    
    '''
    
    if 0 < np.rad2deg(angle) <=45 or 135 < np.rad2deg(angle) <= 180:
        return 3
    elif 45 < np.rad2deg(angle) <= 135:
        return 4
    elif 225 < np.rad2deg(angle) <= 315:
        return 1
    else:
        return 2


def frames_to_avi(path_in, path_out, name_of_sub_dir, fps):
    
    '''
    Objective: turn frames into videos
    input:
        path_in: the root path of the directories of the frames, e.g.: for the same scene, day1, day2,...
        path_out: the path to store the made videos
        name_of_sub_dir: day1, day2,... or train001, train002, ..., etc.
        fps: frame per second
    Output:
        videos made from the frames
    
    '''
    
    frame_array = []
    data_path = join(path_in, name_of_sub_dir)
    files = [f for f in os.listdir(data_path) if isfile(join(data_path, f)) and (fnmatch.fnmatch(f, '*.tif') or fnmatch.fnmatch(f, '*.jpg'))]
    #for sorting the file names properly
    files.sort(key = lambda x: x[5:-4])
    files.sort()

    for i in range(len(files)):
        filename = join(path_in, name_of_sub_dir) + '/' + files[i]
        #reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)       
        #inserting the frames into an image array
        frame_array.append(img)
    
   
    file_name = os.path.join(path_out, name_of_sub_dir + '.avi')
    out = cv2.VideoWriter(file_name,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()
 
def frames_to_avi_overlap(path_in, path_out, name_of_sub_dir, fps, 
                          frames_in_video = 200, 
                          frame_gap = 20):
    '''
    Objective: when the frames of a scene are not enough, make videos with overlapped frames
    input:
        path_in: the root path of the directories of the frames, e.g.: for the same scene, day1, day2,...
        path_out: the path to store the made videos
        name_of_sub_dir: day1, day2,... or train001, train002, ..., etc.
        fps: frame per second
        frames_in_video: max frames in a video
        frame_gap: ex:20, 1~200, 20~220, 40~240,..., etc.
    Output:
        videos made from the frames
    
    '''

    data_path = join(path_in, name_of_sub_dir)
    files = [f for f in os.listdir(data_path) if isfile(join(data_path, f)) and (fnmatch.fnmatch(f, '*.tif') or fnmatch.fnmatch(f, '*.jpg'))]
    #for sorting the file names properly
    files.sort(key = lambda x: x[5:-4])
    files.sort()
    os.makedirs(join(path_out,name_of_sub_dir),exist_ok=True)
    
    video_count = 0
    for i in range(0, len(files)-frames_in_video, frame_gap):
        
        frame_array = []
        for j in range(0, frames_in_video):
            filename = join(path_in, name_of_sub_dir) + '/' + files[i+j]
            #reading each files
            img = cv2.imread(filename)
            height, width, layers = img.shape
            size = (width,height)       
            #inserting the frames into an image array
            frame_array.append(img)
        
        file_name = os.path.join(path_out, name_of_sub_dir + '/'+str(video_count)+'.avi')
        out = cv2.VideoWriter(file_name,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
        for f in range(len(frame_array)):
            # writing to a image array
            out.write(frame_array[f])
        out.release()    
        
        video_count+=1


def video_to_frames(path_in, path_out, name_of_videos, frames_in_video = 400):
    
    '''
    Objective: when the video is very long, e.g. 90000 frames, slice the frames into smaller videos
    input:
        path_in: the path of the video
        path_out: the path to store the made sub-directories of frames, e.g. ./name_of_videos/(0001, 0002, ...)
        name_of_sub_dir: 0001, 0002, ..., etc.
        frames_in_video: max frames in a video

    Output:
        sub-directories with frames inside under the path ./path_out/name_of_videos/(0001, 0002, ...)  
    
    '''


    # create a folder for each video to store its frames
    
    # set a variable to store video data 
    
     
    cap = cv2.VideoCapture(join(path_in, name_of_videos + '.avi'))

    # this indicates that which frame are we looking at
    order_of_frames = 1
    
    video_count = 0    
    
    gap_between_frames = 2   
    # run the loop to all image direcotry(each video)
    while cap.isOpened(): 
        
        ret, frame = cap.read()
        if not ret:
            break
        if order_of_frames % frames_in_video ==1:
            video_count += 1
            order_of_frames = 1
            os.makedirs(join(path_out, name_of_videos+'/'+str(video_count).zfill(4)), exist_ok=True)
            pic_path = join(path_out, name_of_videos+'/'+str(video_count).zfill(4))
            
        if (order_of_frames % gap_between_frames == 0):  # save frame for every 2 frames
            # print(pic_path + str(c) + '.jpg')

            cv2.imwrite(pic_path + '/' +  
                        str(order_of_frames).zfill(3) + '.jpg', frame)  

        order_of_frames += 1
#        cv2.waitKey(0)
    # after running through each frame, close this video    
    cap.release()
    return