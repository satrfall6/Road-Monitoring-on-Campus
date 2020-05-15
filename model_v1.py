# -*- coding: utf-8 -*-
"""
Created on Fri May  1 13:46:40 2020

@author: satrf
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm
import fnmatch

import cv2 
import time
import os 
os.chdir(r'C:\Users\satrf\LB_stuff\Git\Road-Monitoring-on-Campus')
from os.path import join, isdir, isfile
from pathlib import PureWindowsPath
from scipy import sqrt, pi, arctan2
import pandas as pd


from video_preprocessing import frames_to_avi
from optcal_flow_v2 import capture_sparse_optical_flow, make_grid, categorize_flow_points, capture_dense_optical_flow


# setting path and directory
YOUR_PATH = os.getcwd()
YOUR_PATH = PureWindowsPath(YOUR_PATH)
print('working directory now: ', YOUR_PATH)

ucsd_train = (r'./data/UCSD/UCSDped1/Train/')
ucsd_test = (r'./data/UCSD/UCSDped1/Test/')
train_video_path = (r'./data/Train/videos/')
path_in = test_video_path = (r'./data/Test/videos/')


sub_dir_ucsd_train = [d for d in os.listdir(ucsd_train) if isdir(join(ucsd_train, d))]
sub_dir_ucsd_test = [d for d in os.listdir(ucsd_test) if isdir(join(ucsd_test, d)) and not fnmatch.fnmatch(d, '*_gt')]
name_of_sub_dir = sub_dir_ucsd_test[19]
# hyper parameters
X_BLOCK_SIZE, Y_BLOCK_SIZE = 30,30
TIME_INTERVAL = 5



cap = cv2.VideoCapture(join(test_video_path, name_of_sub_dir+'.avi'))

# ret = a boolean return value from getting the frame, first_frame = the first frame in the entire video sequence
ret, first_frame = cap.read()
# Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# make grid based on frame and block size 
grids = make_grid(first_frame, X_BLOCK_SIZE, Y_BLOCK_SIZE)


IDX_MAGNITUDE = 2
IDX_GRID = 4
IDX_TIME = 5
#%%
# make train
## make train flow
start = time.time()
train_flow_points = np.array([]).reshape(0, 8)
video_count = 1
for d in sub_dir_ucsd_train:
#    frames_to_avi(ucsd_train, train_video_path, d, 10)
    flow = capture_sparse_optical_flow(train_video_path, d, grids, 
                                       TIME_INTERVAL, X_BLOCK_SIZE, Y_BLOCK_SIZE)
    # for tracing which video it is 
    flow = np.c_[flow, np.ones(flow.shape[0])*video_count]
    # label for each flow, 0 is normal
    flow = np.c_[flow, np.zeros(flow.shape[0])]
    train_flow_points = np.vstack((train_flow_points, flow))
    video_count +=1


end = time.time()
print('total time spent: ', end - start , 'sec')

#plt.hist(train_flow_points[:,2], 30, range=[0.35, 2], facecolor='gray', align='mid')

std_magnitude = np.std(train_flow_points[:,IDX_MAGNITUDE])
#plt.hist(train_flow_points[:,3], 30, range=[0, 5], facecolor='gray', align='mid')
# set the magnitude into 4 categories
magnitude_categories = 4
train_flow_points[:,IDX_MAGNITUDE] = [(int(m/std_magnitude)+1) if (m<=std_magnitude*(magnitude_categories-1)) 
                                                   else magnitude_categories 
                                                   for m in train_flow_points[:,IDX_MAGNITUDE]]


#store to dataframe
df = pd.DataFrame(train_flow_points, columns = ('x start', 'y start', 'magnitude', 'orientation'
                                      , 'which grid', 'which time interval', 'which video', 'Label'))

df.to_csv(join((r'./data/Train/'), "train_flow.csv"))


## make train spatial descriptor


df_train = pd.read_csv(join((r'./data/Train/'), "train_flow.csv"))
df_train = df_train.iloc[:,1:]
df_train['magnitude'] =df_train['magnitude'].astype(pd.api.types.CategoricalDtype(categories=range(1,int(max(set(df_train['magnitude'] )))+2)))
df_train['orientation'] =df_train['orientation'].astype(pd.api.types.CategoricalDtype(categories=range(1,int(max(set(df_train['orientation'] )))+1)))
df_train['which grid'] =df_train['which grid'].astype(pd.api.types.CategoricalDtype(categories=range(1,int(max(set(df_train['which grid'] )))+1)))
df_train['which time interval'] =df_train['which time interval'].astype(pd.api.types.CategoricalDtype(categories=range(1,int(max(set(df_train['which time interval'] )))+1)))
df_train['which video'] =df_train['which video'].astype(pd.api.types.CategoricalDtype(categories=range(1,int(max(set(df_train['which video'] )))+1)))
df_train['Label'] =df_train['Label'].astype(pd.api.types.CategoricalDtype(categories=range(1)))
print(df_train.dtypes)
grid_with_moving_flow = tuple(set(df_train['which grid']))
set_of_videos = tuple(set(df_train['which video']))
set_of_time_intervals = tuple(set(df_train['which time interval']))

## make train

#when tabling, the non-extisting category will just disapear, append a table below to make sure each category exist
## not sure  what is the better way
appeding_df = np.zeros((magnitude_categories,df_train.shape[1]))
for r in range(magnitude_categories):
    appeding_df[r][2] = int(r+1)
    appeding_df[r][3] = int(r+1)
appeding_df = pd.DataFrame(appeding_df, columns = df_train.columns).astype(df_train.dtypes)

start = time.time()

# the spatial descriptor is 4*4
train = np.array([]).reshape(0, 18)
# for each 30*30 grid, calculate the stats
for grid in grid_with_moving_flow:
    df_grid = df_train.loc[(df_train['which grid'] == grid)]
    print('now it is grid', grid)
    for video in set_of_videos:
        df_video = df_grid.loc[(df_grid['which video'] == video)]
        
        # each row would be a time interval of a grid with a video
        for ti in set_of_time_intervals:
            
            df_time = df_video.loc[(df_video['which time interval'] == ti)]    

            if df_time.shape[0] == 0:
                continue
            else:
                # this would be the matrix shown in paper
                spatial_table = df_time.append(appeding_df,ignore_index=True).groupby(['magnitude', 'orientation']).size().unstack(fill_value=0)
                # just to remove the extra appended rows
                for i in range(spatial_table.shape[0]):
                    spatial_table.iloc[i, i] -= 1
                
                # flatten the table
                spatial_table = spatial_table.to_numpy().reshape(-1)
                # append which grid this cuboid belongs to 
                spatial_table = np.append(spatial_table,[grid,df_time['Label'].iloc[0]])
                train = np.vstack((train, spatial_table))
           
end = time.time()
print('total time spent: ', end - start , 'sec')

train_spatial = pd.DataFrame(train, columns = ['m1o1','m1o2','m1o3','m1o4',
                                               'm2o1','m2o2','m2o3','m2o4',
                                               'm3o1','m3o2','m3o3','m3o4',
                                               'm4o1','m4o2','m4o3','m4o4','grid','Label'])

train_spatial.to_csv(join((r'./data/Train/'), "train_spatial.csv"))


# make test 
## make test flow

# list the directory of anomalous video
dir_with_anomaly = [d[-10:-3] for d in os.listdir(ucsd_test) if fnmatch.fnmatch(d, '*_gt')]

start = time.time()

video_count = 1
test_flow_points = np.array([]).reshape(0, 8)
for d in sub_dir_ucsd_test:
#    frames_to_avi(ucsd_test, test_video_path, d, 10)
    flow = capture_sparse_optical_flow(test_video_path, d, grids, TIME_INTERVAL, X_BLOCK_SIZE, Y_BLOCK_SIZE)
    flow = np.c_[flow, np.ones(flow.shape[0])*video_count]
    flow = np.c_[flow, np.zeros(flow.shape[0])]

    # the video contains anomaly, locate the grid and the time interval it belongs to
    # !! within this "if" is the part that I guess something would go wrong, have no idea how to do it efficiently
    if d in dir_with_anomaly:
        anomaly_frames = [f for f in os.listdir(join(ucsd_test, d+'_gt')) if fnmatch.fnmatch(f, '*.bmp')]
        # set for anomalies label to add up        
        Label_for_anomaly = np.zeros((flow.shape[0],))
        # not all frames contain anomaly, trace which one has and label it 
        for i, bmp in enumerate(anomaly_frames, 1):
            im_bmp = cv2.imread(join(ucsd_test, d+'_gt/'+str(bmp)))
            im_bmp = cv2.cvtColor(im_bmp, cv2.COLOR_BGR2GRAY)
            if np.any(im_bmp!=0):
                # capture the 4 corners of the object 
                anomaly_boundary = [min(np.where(im_bmp!=0)[1]), max(np.where(im_bmp!=0)[1]),
                                 min(np.where(im_bmp!=0)[0]), max(np.where(im_bmp!=0)[0])]
                # find which grids that these corners belong to 
                anomaly_grids = (set([categorize_flow_points(anomaly_boundary[0], anomaly_boundary[2], grids, X_BLOCK_SIZE, Y_BLOCK_SIZE),
                                 categorize_flow_points(anomaly_boundary[0], anomaly_boundary[3], grids, X_BLOCK_SIZE, Y_BLOCK_SIZE),
                                 categorize_flow_points(anomaly_boundary[1], anomaly_boundary[2], grids, X_BLOCK_SIZE, Y_BLOCK_SIZE),
                                 categorize_flow_points(anomaly_boundary[1], anomaly_boundary[3], grids, X_BLOCK_SIZE, Y_BLOCK_SIZE),
                                 categorize_flow_points((anomaly_boundary[0]+anomaly_boundary[1])/2, (anomaly_boundary[2]+anomaly_boundary[3])/2, grids, X_BLOCK_SIZE, Y_BLOCK_SIZE)]))
            
                for a in anomaly_grids:                    
                     # add up these time intervals in these grids (with amomaly) 
                     Label_for_anomaly = Label_for_anomaly + ((flow[:,IDX_GRID]==a) & (flow[:,IDX_TIME]==int(i/TIME_INTERVAL)+1))*1
       
        # after interations, the non-zero part is the anomaly, set to 1 
        Label_for_anomaly = (Label_for_anomaly!=0)*1
        flow[:,7] = Label_for_anomaly
        
    video_count += 1   
    
    test_flow_points = np.vstack((test_flow_points, flow))

end = time.time()
print('total time spent: ', end - start , 'sec')

magnitude_categories = 4
test_flow_points[:,2] = [(int(m/std_magnitude)+1) if (m<=std_magnitude*(magnitude_categories-1)) else magnitude_categories for m in test_flow_points[:,2]]

df = pd.DataFrame(test_flow_points, columns = ('x start', 'y start', 'magnitude', 'orientation'
                                      , 'which grid', 'which time interval', 'which video', 'Label'))

df.to_csv(join((r'./data/Test/'), "df_test.csv"))

## make test spatial descriptor
df_test = pd.read_csv(join((r'./data/Test/'), "df_test.csv"))
df_test = df_test.iloc[:,1:]
df_test['magnitude'] =df_test['magnitude'].astype(pd.api.types.CategoricalDtype(categories=range(1,int(max(set(df_test['magnitude'] )))+2)))
df_test['orientation'] =df_test['orientation'].astype(pd.api.types.CategoricalDtype(categories=range(1,int(max(set(df_test['orientation'] )))+1)))
df_test['which grid'] =df_test['which grid'].astype(pd.api.types.CategoricalDtype(categories=range(1,int(max(set(df_test['which grid'] )))+1)))
df_test['which time interval'] =df_test['which time interval'].astype(pd.api.types.CategoricalDtype(categories=range(1,int(max(set(df_test['which time interval'] )))+1)))
df_test['which video'] =df_test['which video'].astype(pd.api.types.CategoricalDtype(categories=range(1,int(max(set(df_test['which video'] )))+1)))
df_test['Label'] =df_test['Label'].astype(pd.api.types.CategoricalDtype(categories=range(2)))
print(df_test.dtypes)
grid_with_moving_flow = tuple(set(df_test['which grid']))
set_of_videos = tuple(set(df_test['which video']))
set_of_time_intervals = tuple(set(df_test['which time interval']))


appeding_df = np.zeros((magnitude_categories,df_test.shape[1]))
for r in range(magnitude_categories):
    appeding_df[r][2] = int(r+1)
    appeding_df[r][3] = int(r+1)
appeding_df = pd.DataFrame(appeding_df, columns = df_test.columns).astype(df_test.dtypes)


start = time.time()

test = np.array([]).reshape(0, 20)
for grid in grid_with_moving_flow:
    df_grid = df_test.loc[(df_test['which grid'] == grid)]
    print('now it is grid', grid)
    for video in set_of_videos:
        df_video = df_grid.loc[(df_grid['which video'] == video)]
 
        for ti in set_of_time_intervals:
            
            df_time = df_video.loc[(df_video['which time interval'] == ti)]    

            if df_time.shape[0] == 0:
                continue
            else:
                spatial_table = df_time.append(appeding_df,ignore_index=True).groupby(['magnitude', 'orientation']).size().unstack(fill_value=0)
                for i in range(spatial_table.shape[0]):
                    spatial_table.iloc[i, i] -= 1
                spatial_table = spatial_table.to_numpy().reshape(-1)
                spatial_table = np.append(spatial_table,[grid, ti, video, df_time['Label'].iloc[0]])
                test = np.vstack((test, spatial_table))
           
end = time.time()
print('total time spent: ', end - start , 'sec')

test_spatial = pd.DataFrame(test, columns = ['m1o1','m1o2','m1o3','m1o4',
                                               'm2o1','m2o2','m2o3','m2o4',
                                               'm3o1','m3o2','m3o3','m3o4',
                                               'm4o1','m4o2','m4o3','m4o4',
                                               'grid','time_interval', 'video','Label'])

test_spatial.to_csv(join((r'./data/Test/'), "test_spatial.csv"))

# read spatial train and test
train_spatial = pd.read_csv(join((r'./data/Train/'), "train_spatial.csv"))
train_spatial = train_spatial.iloc[:,1:]
train_spatial['grid'] =train_spatial['grid'].astype(pd.api.types.CategoricalDtype(categories=range(1,int(max(set(train_spatial['grid'] )))+1)))
train_spatial['Label'] =train_spatial['Label'].astype(pd.api.types.CategoricalDtype(categories=range(1)))

test_spatial = pd.read_csv(join((r'./data/Test/'), "test_spatial.csv"))
test_spatial = test_spatial.iloc[:,1:]
test_spatial['grid'] =test_spatial['grid'].astype(pd.api.types.CategoricalDtype(categories=range(1,int(max(set(test_spatial['grid'] )))+1)))
test_spatial['Label'] =test_spatial['Label'].astype(pd.api.types.CategoricalDtype(categories=range(2)))


X_train = train_spatial.iloc[:,0:-2]


X_test = test_spatial.iloc[:,0:-4]
y_test= test_spatial.iloc[:,-1]
video_info_test = test_spatial.loc[:,'grid': 'video']

#%%

'''
#%% distance based clustering 
from sklearn.cluster import KMeans


wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=10,  random_state=0).fit(X_train)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
#%%
from scipy.spatial.distance import euclidean
from sklearn.metrics import accuracy_score

kmeans = KMeans(n_clusters=4, random_state=0).fit(X_train)
learned_pattern = kmeans.cluster_centers_
kmean_label = kmeans.labels_

#calculate the threshold
dist_stats = [[],[],[],[]]
for i in range(X_train.shape[0]):
    dist_stats[kmean_label[i]].append(euclidean(X_train.iloc[i].to_numpy(), learned_pattern[kmean_label[i]]))

dist_threshold = [np.median(th)+4*np.std(th) for th in dist_stats]

X_test = test_spatial.iloc[:,0:-4]
y_test= test_spatial.iloc[:,-1]
video_info_test = test_spatial.loc[:,'grid': 'video']

y_test_pred = np.zeros((X_test.shape[0],))
for test in range(X_test.shape[0]):
    for pa in range(learned_pattern.shape[0]):
        if euclidean(X_test.iloc[test].to_numpy(), learned_pattern[pa]) < dist_threshold[pa]:
            y_test_pred[test] = 0
        else:  
            y_test_pred[test] = 1


from sklearn.metrics import confusion_matrix


len(np.where(y_test==1)[0])
tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()

precision = tp/(tp+fp)
recall = tp/(tp+fn)
f1_score = 2*(precision*recall)/(precision+recall)
print('precesion is', precision)
print('recall is', recall)
print('f1 score is ',f1_score)

'''
#%%
from sklearn.neighbors import KDTree
from sklearn.metrics import confusion_matrix
#KDTree.valid_metrics

num_neighbor = 7
kdt = KDTree(X_train, leaf_size=40, metric='l2')

distance, _ = kdt.query(X_test, k=num_neighbor, return_distance=True)
max_distance_idx = num_neighbor-1 
tau = np.std(distance[:,max_distance_idx])*3
y_pred = (distance[:,max_distance_idx]>tau)*1
d = 'Test004'
#d = 'Test020'
#d = 'Test025'
#d = 'Test026'
visualize_dense_anomalous_gird(test_video_path, d, grids, y_test, y_pred , video_info_test, X_BLOCK_SIZE = 30, Y_BLOCK_SIZE = 30)

#tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
#
#precision = tp/(tp+fp)
#recall = tp/(tp+fn)
#f1_score = 2*(precision*recall)/(precision+recall)
#print('precesion is', precision)
#print('recall is', recall)
#print('f1 score is ',f1_score)
#plt.hist(distance[:,max_distance_idx], 20, range = [0, 500])
#%% one class svm
from sklearn.svm import OneClassSVM

clf = OneClassSVM(nu=0.035, kernel="linear", gamma='auto').fit(X_train)

y_pred = clf.predict(X_test)
y_pred = (y_pred==-1)*1
plt.hist(y_pred)


from sklearn.metrics import confusion_matrix

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

precision = tp/(tp+fp)
recall = tp/(tp+fn)
f1_score = 2*(precision*recall)/(precision+recall)
print('precesion is', precision)
print('recall is', recall)
print('f1 score is ',f1_score)
#%% fine tune LK params for video 20 
from optcal_flow_v2 import capture_optical_flow_points
path_in = test_video_path
for d in sub_dir_ucsd_test:
#d = "Test002"
    capture_optical_flow_points(test_video_path, d, grids, time_interval = 5, X_BLOCK_SIZE = 30, Y_BLOCK_SIZE = 30,  if_show = True)


#%%
# make train
## make train dense flow
start = time.time()
train_flow_points = np.array([]).reshape(0, 8)
video_count = 1
for d in sub_dir_ucsd_train:
#    frames_to_avi(ucsd_train, train_video_path, d, 10)
    flow = capture_dense_optical_flow(train_video_path, d, grids, 
                                       TIME_INTERVAL, X_BLOCK_SIZE, Y_BLOCK_SIZE,
                                       if_show = False)
    # for tracing which video it is 
    flow = np.c_[flow, np.ones(flow.shape[0])*video_count]
    # label for each flow, 0 is normal
    flow = np.c_[flow, np.zeros(flow.shape[0])]
    train_flow_points = np.vstack((train_flow_points, flow))
    video_count +=1


end = time.time()
print('total time spent: ', end - start , 'sec')

#plt.hist(train_flow_points[:,2], 30, range=[0.35, 2], facecolor='gray', align='mid')

std_magnitude = np.std(train_flow_points[:,IDX_MAGNITUDE])
#plt.hist(train_flow_points[:,3], 30, range=[0, 5], facecolor='gray', align='mid')
# set the magnitude into 4 categories
magnitude_categories = 4
train_flow_points[:,IDX_MAGNITUDE] = [(int(m/std_magnitude)+1) if (m<=std_magnitude*(magnitude_categories-1)) 
                                                   else 0 
                                                   for m in train_flow_points[:,IDX_MAGNITUDE]]


#store to dataframe
df = pd.DataFrame(train_flow_points, columns = ('x start', 'y start', 'magnitude', 'orientation'
                                      , 'which grid', 'which time interval', 'which video', 'Label'))

df.to_csv(join((r'./data/Train/'), "train_dense_flow.csv"))

## make train dense spatial descriptor


df_train = pd.read_csv(join((r'./data/Train/'), "train_dense_flow.csv"))
df_train = df_train.iloc[:,1:]
df_train['magnitude'] =df_train['magnitude'].astype(pd.api.types.CategoricalDtype(categories=range(1,int(max(set(df_train['magnitude'] )))+2)))
df_train['orientation'] =df_train['orientation'].astype(pd.api.types.CategoricalDtype(categories=range(1,int(max(set(df_train['orientation'] )))+1)))
df_train['which grid'] =df_train['which grid'].astype(pd.api.types.CategoricalDtype(categories=range(1,int(max(set(df_train['which grid'] )))+1)))
df_train['which time interval'] =df_train['which time interval'].astype(pd.api.types.CategoricalDtype(categories=range(1,int(max(set(df_train['which time interval'] )))+1)))
df_train['which video'] =df_train['which video'].astype(pd.api.types.CategoricalDtype(categories=range(1,int(max(set(df_train['which video'] )))+1)))
df_train['Label'] =df_train['Label'].astype(pd.api.types.CategoricalDtype(categories=range(1)))
print(df_train.dtypes)
grid_with_moving_flow = tuple(set(df_train['which grid']))
set_of_videos = tuple(set(df_train['which video']))
set_of_time_intervals = tuple(set(df_train['which time interval']))

## make train

#when tabling, the non-extisting category will just disapear, append a table below to make sure each category exist
## not sure  what is the better way
appeding_df = np.zeros((magnitude_categories,df_train.shape[1]))
for r in range(magnitude_categories):
    appeding_df[r][2] = int(r+1)
    appeding_df[r][3] = int(r+1)
appeding_df = pd.DataFrame(appeding_df, columns = df_train.columns).astype(df_train.dtypes)

start = time.time()

# the spatial descriptor is 4*4
train = np.array([]).reshape(0, 18)
# for each 30*30 grid, calculate the stats
for grid in grid_with_moving_flow:
    df_grid = df_train.loc[(df_train['which grid'] == grid)]
    print('now it is grid', grid)
    for video in set_of_videos:
        df_video = df_grid.loc[(df_grid['which video'] == video)]
        
        # each row would be a time interval of a grid with a video
        for ti in set_of_time_intervals:
            
            df_time = df_video.loc[(df_video['which time interval'] == ti)]    

            if df_time.shape[0] == 0:
                continue
            else:
                # this would be the matrix shown in paper
                spatial_table = df_time.append(appeding_df,ignore_index=True).groupby(['magnitude', 'orientation']).size().unstack(fill_value=0)
                # just to remove the extra appended rows
                for i in range(spatial_table.shape[0]):
                    spatial_table.iloc[i, i] -= 1
                
                # flatten the table
                spatial_table = spatial_table.to_numpy().reshape(-1)
                # append which grid this cuboid belongs to 
                spatial_table = np.append(spatial_table,[grid,df_time['Label'].iloc[0]])
                train = np.vstack((train, spatial_table))
           
end = time.time()
print('total time spent: ', end - start , 'sec')

train_spatial = pd.DataFrame(train, columns = ['m1o1','m1o2','m1o3','m1o4',
                                               'm2o1','m2o2','m2o3','m2o4',
                                               'm3o1','m3o2','m3o3','m3o4',
                                               'm4o1','m4o2','m4o3','m4o4','grid','Label'])

train_spatial.to_csv(join((r'./data/Train/'), "train_dense_spatial.csv"))

# make test 
## make test flow

# list the directory of anomalous video
dir_with_anomaly = [d[-10:-3] for d in os.listdir(ucsd_test) if fnmatch.fnmatch(d, '*_gt')]

start = time.time()

video_count = 1
test_flow_points = np.array([]).reshape(0, 8)
for d in sub_dir_ucsd_test:
#    frames_to_avi(ucsd_test, test_video_path, d, 10)
    flow = capture_dense_optical_flow(test_video_path, d, grids, TIME_INTERVAL, X_BLOCK_SIZE, Y_BLOCK_SIZE)
    flow = np.c_[flow, np.ones(flow.shape[0])*video_count]
    flow = np.c_[flow, np.zeros(flow.shape[0])]

    # the video contains anomaly, locate the grid and the time interval it belongs to
    # !! within this "if" is the part that I guess something would go wrong, have no idea how to do it efficiently
    if d in dir_with_anomaly:
        anomaly_frames = [f for f in os.listdir(join(ucsd_test, d+'_gt')) if fnmatch.fnmatch(f, '*.bmp')]
        # set for anomalies label to add up        
        Label_for_anomaly = np.zeros((flow.shape[0],))
        # not all frames contain anomaly, trace which one has and label it 
        for i, bmp in enumerate(anomaly_frames, 1):
            im_bmp = cv2.imread(join(ucsd_test, d+'_gt/'+str(bmp)))
            im_bmp = cv2.cvtColor(im_bmp, cv2.COLOR_BGR2GRAY)
            if np.any(im_bmp!=0):
                # capture the 4 corners of the object 
                anomaly_boundary = [min(np.where(im_bmp!=0)[1]), max(np.where(im_bmp!=0)[1]),
                                 min(np.where(im_bmp!=0)[0]), max(np.where(im_bmp!=0)[0])]
                # find which grids that these corners belong to 
                anomaly_grids = (set([categorize_flow_points(anomaly_boundary[0], anomaly_boundary[2], grids, X_BLOCK_SIZE, Y_BLOCK_SIZE),
                                 categorize_flow_points(anomaly_boundary[0], anomaly_boundary[3], grids, X_BLOCK_SIZE, Y_BLOCK_SIZE),
                                 categorize_flow_points(anomaly_boundary[1], anomaly_boundary[2], grids, X_BLOCK_SIZE, Y_BLOCK_SIZE),
                                 categorize_flow_points(anomaly_boundary[1], anomaly_boundary[3], grids, X_BLOCK_SIZE, Y_BLOCK_SIZE),
                                 categorize_flow_points((anomaly_boundary[0]+anomaly_boundary[1])/2, (anomaly_boundary[2]+anomaly_boundary[3])/2, grids, X_BLOCK_SIZE, Y_BLOCK_SIZE)]))
            
                for a in anomaly_grids:                    
                     # add up these time intervals in these grids (with amomaly) 
                     Label_for_anomaly = Label_for_anomaly + ((flow[:,IDX_GRID]==a) & (flow[:,IDX_TIME]==int(i/TIME_INTERVAL)+1))*1
       
        # after interations, the non-zero part is the anomaly, set to 1 
        Label_for_anomaly = (Label_for_anomaly!=0)*1
        flow[:,7] = Label_for_anomaly
        
    video_count += 1   
    
    test_flow_points = np.vstack((test_flow_points, flow))

end = time.time()
print('total time spent: ', end - start , 'sec')

magnitude_categories = 4
test_flow_points[:,2] = [(int(m/std_magnitude)+1) if (m<=std_magnitude*(magnitude_categories-1)) else magnitude_categories for m in test_flow_points[:,2]]

df = pd.DataFrame(test_flow_points, columns = ('x start', 'y start', 'magnitude', 'orientation'
                                      , 'which grid', 'which time interval', 'which video', 'Label'))

df.to_csv(join((r'./data/Test/'), "df_dense_test.csv"))

## make test spatial descriptor
df_test = pd.read_csv(join((r'./data/Test/'), "df_dense_test.csv"))
df_test = df_test.iloc[:,1:]
df_test['magnitude'] =df_test['magnitude'].astype(pd.api.types.CategoricalDtype(categories=range(1,int(max(set(df_test['magnitude'] )))+2)))
df_test['orientation'] =df_test['orientation'].astype(pd.api.types.CategoricalDtype(categories=range(1,int(max(set(df_test['orientation'] )))+1)))
df_test['which grid'] =df_test['which grid'].astype(pd.api.types.CategoricalDtype(categories=range(1,int(max(set(df_test['which grid'] )))+1)))
df_test['which time interval'] =df_test['which time interval'].astype(pd.api.types.CategoricalDtype(categories=range(1,int(max(set(df_test['which time interval'] )))+1)))
df_test['which video'] =df_test['which video'].astype(pd.api.types.CategoricalDtype(categories=range(1,int(max(set(df_test['which video'] )))+1)))
df_test['Label'] =df_test['Label'].astype(pd.api.types.CategoricalDtype(categories=range(2)))
print(df_test.dtypes)
grid_with_moving_flow = tuple(set(df_test['which grid']))
set_of_videos = tuple(set(df_test['which video']))
set_of_time_intervals = tuple(set(df_test['which time interval']))


appeding_df = np.zeros((magnitude_categories,df_test.shape[1]))
for r in range(magnitude_categories):
    appeding_df[r][2] = int(r+1)
    appeding_df[r][3] = int(r+1)
appeding_df = pd.DataFrame(appeding_df, columns = df_test.columns).astype(df_test.dtypes)


start = time.time()

test = np.array([]).reshape(0, 20)
for grid in grid_with_moving_flow:
    df_grid = df_test.loc[(df_test['which grid'] == grid)]
    print('now it is grid', grid)
    for video in set_of_videos:
        df_video = df_grid.loc[(df_grid['which video'] == video)]
 
        for ti in set_of_time_intervals:
            
            df_time = df_video.loc[(df_video['which time interval'] == ti)]    

            if df_time.shape[0] == 0:
                continue
            else:
                spatial_table = df_time.append(appeding_df,ignore_index=True).groupby(['magnitude', 'orientation']).size().unstack(fill_value=0)
                for i in range(spatial_table.shape[0]):
                    spatial_table.iloc[i, i] -= 1
                spatial_table = spatial_table.to_numpy().reshape(-1)
                spatial_table = np.append(spatial_table,[grid, ti, video, df_time['Label'].iloc[0]])
                test = np.vstack((test, spatial_table))
           
end = time.time()
print('total time spent: ', end - start , 'sec')

test_spatial = pd.DataFrame(test, columns = ['m1o1','m1o2','m1o3','m1o4',
                                               'm2o1','m2o2','m2o3','m2o4',
                                               'm3o1','m3o2','m3o3','m3o4',
                                               'm4o1','m4o2','m4o3','m4o4',
                                               'grid','time_interval', 'video','Label'])

test_spatial.to_csv(join((r'./data/Test/'), "test_dense_spatial.csv"))
#%%
# read spatial train and test
train_spatial = pd.read_csv(join((r'./data/Train/'), "train_dense_spatial.csv"))
train_spatial = train_spatial.iloc[:,1:]
train_spatial['grid'] =train_spatial['grid'].astype(pd.api.types.CategoricalDtype(categories=range(1,int(max(set(train_spatial['grid'] )))+1)))
train_spatial['Label'] =train_spatial['Label'].astype(pd.api.types.CategoricalDtype(categories=range(1)))

test_spatial = pd.read_csv(join((r'./data/Test/'), "test_dense_spatial.csv"))
test_spatial = test_spatial.iloc[:,1:]
test_spatial['grid'] =test_spatial['grid'].astype(pd.api.types.CategoricalDtype(categories=range(1,int(max(set(test_spatial['grid'] )))+1)))
test_spatial['Label'] =test_spatial['Label'].astype(pd.api.types.CategoricalDtype(categories=range(2)))


X_train = train_spatial.iloc[:,0:-2]


X_test = test_spatial.iloc[:,0:-4]
y_test= test_spatial.iloc[:,-1]
video_info_test = test_spatial.loc[:,'grid': 'video']


#%%bins example

x = np.array([0.2, 6.4, 3.0, 1.6])
bins = np.array([0.0, 1.0, 2.5, 4.0, 10.0])
inds = np.digitize(x, bins)
for n in range(x.size):
    print(bins[inds[n]-1], "<=", x[n], "<", bins[inds[n]])    
    
x = np.array([1.2, 10.0, 12.4, 15.5, 20.])
bins = np.array([0, 5, 10, 15, 20])
np.digitize(x,bins,right=True)

np.digitize(x,bins,right=False)
#%%example for auc, roc_curve

fpr, tpr, threshold = metrics.roc_curve(y_test, preds)

y_pred_proba = clf.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()