# -*- coding: utf-8 -*-
"""
Created on Fri May  1 13:46:40 2020

@author: satrf
"""
import numpy as np

import fnmatch

import cv2 
print(cv2.getBuildInformation())

import os 
#os.chdir(r'C:\Users\satrf\LB_stuff\Git\Road-Monitoring-on-Campus')
from os.path import join, isdir, isfile
from pathlib import PureWindowsPath
import pandas as pd
import tqdm

from sklearn.neighbors import KDTree
from data_pipeline import make_grid, capture_dense_optical_flow, grids_of_interest
from detect_anomaly import visualize_dense_anomalous_gird
import matplotlib.pyplot as plt
import time 

IF_TRAIN = False
IF_VISUALIZE = True

#setting test params
#test_item = '_magScaled'
#test_item = '_9ori'
test_item = '_multiThrMagScaled12_9ori'

IF_NORMAL_ORI = True

if IF_NORMAL_ORI:
    orientation_categories = 9
else:
    orientation_categories = 4


######################################

# setting path and directory
YOUR_PATH = os.getcwd()
YOUR_PATH = PureWindowsPath(YOUR_PATH)
print('working directory now: ', YOUR_PATH)


# hyper parameters
X_BLOCK_SIZE, Y_BLOCK_SIZE = 50,30
TIME_INTERVAL = 20

#path_in = train_video_path = (r'./data/QMUL/junction1/train/')
#test_video_path = (r'./data/QMUL/junction1/test/')
#
#sub_dir_train = [d[0:-4] for d in os.listdir(train_video_path) if isfile(join(train_video_path, d))]
#sub_dir_test = [d[0:-4] for d in os.listdir(test_video_path) if isfile(join(test_video_path, d)) and not fnmatch.fnmatch(d, '*_gt')]
#name_of_sub_dir = sub_dir_test[0]
#'''
#QMUL/test/028 is the u-turn video
#050 is the reverse traffic
#'''



detect_results = (r'./data/visualized_results')


## parameters for testing ucsd dataset
UCSD_to_use = str(2)    

# the input of the system is videos, set the path of the input videos 
ucsd_train = (r'./data/UCSD/UCSDped'+UCSD_to_use+'/Train/')
ucsd_test = (r'./data/UCSD/UCSDped'+UCSD_to_use+'/Test/')
path_in = train_video_path = (r'./data/Train'+UCSD_to_use+'/videos/')
test_video_path = (r'./data/Test'+UCSD_to_use+'/videos/')


sub_dir_train = [d for d in os.listdir(ucsd_train) if isdir(join(ucsd_train, d))]
sub_dir_test = [d for d in os.listdir(ucsd_test) if isdir(join(ucsd_test, d)) and not fnmatch.fnmatch(d, '*_gt')]
name_of_sub_dir = sub_dir_test[0]
    




# capture the first frame to make grids
cap = cv2.VideoCapture(join(test_video_path, name_of_sub_dir+'.avi'))

# ret = a boolean return value from getting the frame, first_frame = the first frame in the entire video sequence
ret, first_frame = cap.read()
# Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# make grid based on frame and block size 
GRIDS = make_grid(first_frame, X_BLOCK_SIZE, Y_BLOCK_SIZE)


IDX_MAGNITUDE = 2
IDX_ORIENTATION = 3
IDX_GRID = 4
IDX_TIME = 5
IDX_MAG_CAT = -1

TEST_VIDEOS = -1

#%% build maks of roi

'''
#%%
gb = cv2.GaussianBlur(prev_gray, (5,5), 2)
mb = cv2.medianBlur(prev_gray,5)
can = cv2.Canny(gb, 30, 250)

img = mask
prev_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

mb = cv2.medianBlur(prev_gray,5)
can = cv2.Canny(mb, 60, 220)
plt.imshow(can)


# should only use gray-scale img for thresholding

ret,thresh1 = cv2.threshold(prev_gray,127,255,cv2.THRESH_BINARY)
ret,thresh2 = cv2.threshold(prev_gray,127,255,cv2.THRESH_BINARY_INV)
ret,thresh3 = cv2.threshold(prev_gray,127,255,cv2.THRESH_TRUNC)
ret,thresh4 = cv2.threshold(prev_gray,127,255,cv2.THRESH_TOZERO)
ret,thresh5 = cv2.threshold(prev_gray,127,255,cv2.THRESH_TOZERO_INV)

th2 = cv2.adaptiveThreshold(mb,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
th3 = cv2.adaptiveThreshold(mb,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)

kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(can,kernel,iterations = 1)
dilation = cv2.dilate(can,kernel,iterations = 1)
opening = cv2.morphologyEx(can, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(can, cv2.MORPH_CLOSE, kernel)
gradient = cv2.morphologyEx(can, cv2.MORPH_GRADIENT, kernel)
tophat = cv2.morphologyEx(can, cv2.MORPH_TOPHAT, kernel)
blackhat = cv2.morphologyEx(can, cv2.MORPH_BLACKHAT, kernel)

titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV',
          'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding',
          'erosion','dilation','opening','closing',
          'gradient','tophat','blackhat']
images = [mb, thresh1, thresh2, thresh3, thresh4, thresh5, th2, th3,
          erosion, dilation, opening, closing, gradient, tophat, blackhat]



for i in range(len(images)):
    plt.figure(figsize=(40,28))
    plt.subplot(5,3,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])

plt.show()

1. ROI 抓不太到
2. 嘗試針對右轉車道追蹤2秒前，其他的就設(0, 0)
3. 太大的mag直接捨棄不要clip
4. 長一點的time interval


#%%

testing = cv2.morphologyEx(th3, cv2.MORPH_GRADIENT, kernel).copy()
plt.imshow(testing)
#%%
contours, hierarchy = cv2.findContours(testing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

testing = cv2.drawContours(testing, contours, -1, (0,255,0), 15)

plt.imshow(testing)
'''


#%%

# make flow for all the videos in a directory
def make_flow(path_of_videos, grids, ROI):
    
    '''
    Objective: capture optical flow points from video
    input:
        path_in: dir of training video data, e.g './data/train/videos'

        grids: the corner for each grid created by "make_grid" function

    Output:
        an n by 6 array [starting x, starting y, magnitude, orientation, 
                         grid of the flow, time interval of the flow]
    
    '''
    sub_files = [d[0:-4] for d in os.listdir(path_of_videos) 
                if isfile(join(path_of_videos, d)) 
                and fnmatch.fnmatch(d, '*.avi')]
    sub_files = sub_files[0:51]
    flow_points = np.array([]).reshape(0, 7)
    for d in tqdm.tqdm(sub_files):
    #    frames_to_avi(ucsd_test, test_video_path, d, 10)
        flow = capture_dense_optical_flow(path_of_videos, d, grids, 
                                          ROI, TIME_INTERVAL, 
                                          X_BLOCK_SIZE, Y_BLOCK_SIZE,
                                          if_normal_ori = IF_NORMAL_ORI,
                                          if_show = False)
        flow = np.c_[flow, np.ones(flow.shape[0])*int(d[-3:])]
        flow_points = np.r_[flow_points, flow]
    flow_points[:,IDX_ORIENTATION] = np.clip(flow_points[:,IDX_ORIENTATION], 0, 9)

    return flow_points 

# make spatial descriptor for a test or train flow dataframe
def make_spatial_descriptor(df_flow, magnitude_categories = 4):
    
    '''
    Objective: build the spatial descriptor for training the model
    input:
        df_flow: the output from 'make_flow'

        magnitude_categories: number of categories of magnitude

    Output:
        an n by (o*m)+3 dataframe 
    
    '''
    
    grid_with_moving_flow = tuple(set(df_flow['which grid']))
    set_of_videos = tuple(set(df_flow['which video']))
    set_of_time_intervals = tuple(set(df_flow['which time interval']))
    
    
    
    appeding_df = np.zeros((magnitude_categories*orientation_categories,df_flow.shape[1]))
    # for each row
    
    row_count = 0
    for m_cat in range(1, magnitude_categories+1): 
        ori_cat_count = 1
        while ori_cat_count <= orientation_categories: 
            # where mag_cat is 
            appeding_df[row_count][IDX_MAG_CAT] = int(m_cat)
            # where which grid is 
            appeding_df[row_count][IDX_ORIENTATION] = int(ori_cat_count)
           
            ori_cat_count+=1
            row_count += 1
    appeding_df = pd.DataFrame(appeding_df, columns = df_flow.columns).astype(df_flow.dtypes)

    spatial_descriptor = np.array([]).reshape(0, magnitude_categories*orientation_categories+3)
    for grid in tqdm.tqdm(grid_with_moving_flow):
        df_grid = df_flow.loc[(df_flow['which grid'] == grid)]
        for video in set_of_videos:
            df_video = df_grid.loc[(df_grid['which video'] == video)]
     
            for ti in set_of_time_intervals:
                
                df_time = df_video.loc[(df_video['which time interval'] == ti)]    
    
                if df_time.shape[0] == 0:
                    continue
                else:
                    spatial_table = df_time.append(appeding_df,ignore_index=True).groupby(['mag_cat', 'orientation']).size().unstack(fill_value=0)-1
    
                    spatial_table = spatial_table.to_numpy().reshape(1,-1)
                    spatial_table = np.append(spatial_table,[grid, ti, video]).reshape(1,-1)
                    spatial_descriptor = np.r_[spatial_descriptor, spatial_table]

    columns = []           
    for m_cat in range(1, magnitude_categories+1): 
        ori_cat_count = 1
        while ori_cat_count <= orientation_categories: 
            columns = columns + ['m'+str(m_cat)+'o'+str(ori_cat_count)]    
            ori_cat_count +=1
    
    # different from the training spatial
    return pd.DataFrame(spatial_descriptor, columns = columns+['grid', 'time_interval', 'video'])


def kdtree_model(X_train, X_test, num_neighbor = 7, leaf_size = 40, 
                 metric = 'l2',
                 distance_threshold = 3.6):
    
    
    '''
    Objective: build the KDTree model to calculate the distance between the input grids
                and the grids learned from normal behaviors
    
    Output: return the binary prediction (if a grid of a time interval in a video is anomalous)

    '''
    
    kdt = KDTree(X_train, leaf_size, metric)
     
    distance, _ = kdt.query(X_test, k=num_neighbor, return_distance=True)
    max_distance_idx = num_neighbor-1 
    tau = np.std(distance[:,max_distance_idx])*distance_threshold #3.6 for ped2, 5.25 for ped1
    
    
    y_pred = (distance[:,max_distance_idx]>tau)*1
    
    return y_pred

def make_train(path_of_videos, grids, ROI):

    '''
    Objective: call the 'make_flow' and 'make_spatial_descriptor' to make training set
    '''
    
    train_flow_points = make_flow(path_of_videos, grids, ROI)
    
    std_magnitude = np.std(train_flow_points[:,IDX_MAGNITUDE])
    mag_med = np.median(train_flow_points[:,IDX_MAGNITUDE])
    # set the magnitude into 4 categories
    magnitude_categories = 4
    mag_cat = np.array([(int(abs(m-mag_med)/std_magnitude)+1) 
                        if (abs(m-mag_med)<=std_magnitude*(magnitude_categories-1)) 
                        else magnitude_categories 
                        for m in train_flow_points[:,IDX_MAGNITUDE]])
    train_flow_points = np.c_[train_flow_points[:,0:7], mag_cat]
    
    df_train_flow = pd.DataFrame(train_flow_points, columns = ('x start', 'y start', 
                                                          'magnitude', 'orientation',
                                                          'which grid', 'which time interval',
                                                          'which video', 'mag_cat'))
    
    train_spatial = make_spatial_descriptor(df_train_flow)
    
    return train_spatial, std_magnitude, mag_med

def make_test(path_of_videos, grids, ROI, train_std_magnitude, train_mag_med):
    
    
    '''
    Objective: call the 'make_flow' and 'make_spatial_descriptor' to make testing set
    '''
    
    magnitude_categories = 4
    test_flow_points = make_flow(path_of_videos, grids, ROI)
    
    mag_cat = np.array([(int(abs(m-train_mag_med)/train_std_magnitude)+1) 
                        if (abs(m-train_mag_med)<=train_std_magnitude*(magnitude_categories-1)) 
                        else magnitude_categories 
                        for m in test_flow_points[:,IDX_MAGNITUDE]])
        
    test_flow_points = np.c_[test_flow_points[:,0:7], mag_cat]        
    df_test_flow = pd.DataFrame(test_flow_points, columns = ('x start', 'y start', 
                                                          'magnitude', 'orientation',
                                                          'which grid', 'which time interval',
                                                          'which video', 'mag_cat'))
    test_spatial = make_spatial_descriptor(df_test_flow)
    
    return test_spatial

#%%
def main():

    mask, ROI = grids_of_interest(train_video_path, GRIDS, X_BLOCK_SIZE, Y_BLOCK_SIZE, 
                                  TIME_INTERVAL, if_show = True)

    if IF_TRAIN:
        

        df_train_spatial, std_magnitude, mag_med =  make_train(train_video_path, GRIDS, 
                                                               ROI)
        df_test_spatial = make_test(test_video_path, GRIDS, ROI,
                  std_magnitude, mag_med)

        
        X_train = df_train_spatial.iloc[:,0:-3]
        X_test = df_test_spatial.iloc[:,0:-3]
        video_info_test = df_test_spatial.loc[:,'grid': 'video'].astype(int)
        y_pred = kdtree_model(X_train, X_test, num_neighbor = 7, leaf_size = 40, 
                     metric = 'l2', distance_threshold = 6)
        
    else:
        df_train_spatial = pd.read_csv(join((r'./data/Train'+UCSD_to_use+'/'), "df_train_dense_spatial"+UCSD_to_use+test_item+".csv"))
        df_train_spatial = df_train_spatial.iloc[:,1:]

        df_test_spatial = pd.read_csv(join((r'./data/Test'+UCSD_to_use+'/'), "df_test_dense_spatial"+UCSD_to_use+test_item+".csv"))
        df_test_spatial = df_test_spatial.iloc[:,1:]
        X_train = df_train_spatial.iloc[:,0:-1]
        X_test = df_test_spatial.iloc[:,0:-3]
        video_info_test = df_test_spatial.loc[:,'grid': 'video'].astype(int)
        y_pred = kdtree_model(X_train, X_test, num_neighbor = 7, leaf_size = 40, 
                     metric = 'l2', distance_threshold = 3)
        
    if IF_VISUALIZE:    
        # for visualize anomalies
        for d in sub_dir_test[0:TEST_VIDEOS]:
    
            visualize_dense_anomalous_gird(test_video_path, d, GRIDS, y_pred, 
                                       video_info_test, TIME_INTERVAL, detect_results,
                                       test_item, UCSD_to_use, 
                                       X_BLOCK_SIZE , Y_BLOCK_SIZE,
                                       if_make_video = False)
#%%
if __name__ == "__main__":     
    main()
