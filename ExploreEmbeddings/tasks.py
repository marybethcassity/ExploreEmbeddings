import numpy as np
import os 
import cv2
import shutil
import matplotlib.pyplot as plt 
import matplotlib

#from celery import Celery

from bsoid_utils import *

def return_plot(folder_path, fps, UMAP_PARAMS, cluster_range, HDBSCAN_PARAMS, training_fraction):
    if not os.path.isdir('uploads'):
        os.mkdir('uploads')
    if not os.path.isdir(os.path.join('uploads', 'csvs')):
        os.mkdir(os.path.join('uploads', 'csvs'))
    if not os.path.isdir(os.path.join('uploads', 'videos')):
        os.mkdir(os.path.join('uploads', 'videos'))

    for filename in os.listdir(folder_path):
        if filename.endswith('.mp4'):
            mp4filepath = os.path.join('uploads', 'videos', filename)
            shutil.copyfile(os.path.join(folder_path,filename), mp4filepath)

        elif filename.endswith('.csv'):
            csvfilepath = os.path.join('uploads', 'csvs', filename)
            csvfilename = filename
            shutil.copyfile(os.path.join(folder_path,filename), csvfilepath)
            file_j_df = pd.read_csv(csvfilepath, low_memory=False)          

    pose_chosen = []

    file_j_df_array = np.array(file_j_df)

    p = st.multiselect('Identified __pose__ to include:', [*file_j_df_array[0, 1:-1:3]], [*file_j_df_array[0, 1:-1:3]])
    for a in p:
        index = [i for i, s in enumerate(file_j_df_array[0, 1:]) if a in s]
        if not index in pose_chosen:
            pose_chosen += index
    pose_chosen.sort()

    file_j_processed, p_sub_threshold = adp_filt(file_j_df, pose_chosen)
    file_j_processed = file_j_processed.reshape((1, file_j_processed.shape[0], file_j_processed.shape[1]))

    scaled_features, features, frame_mapping, frame_number = compute(file_j_processed, file_j_df_array, fps)

    train_size = subsample(file_j_processed, fps, training_fraction)

    sampled_embeddings, sampled_frame_mapping, sampled_frame_number = learn_embeddings(scaled_features, features, UMAP_PARAMS, train_size, frame_mapping, frame_number)

    assignments = hierarchy(cluster_range, sampled_embeddings, HDBSCAN_PARAMS)

    sampled_embeddings_filtered = sampled_embeddings[assignments>=0]
    assignments_filtered = assignments[assignments>=0]    
    sampled_frame_mapping_filtered = sampled_frame_mapping[assignments>=0] 
    sampled_frame_number_filtered =  sampled_frame_number[assignments>=0] 

    plot = create_plotly(sampled_embeddings_filtered, assignments_filtered, csvfilename, sampled_frame_mapping_filtered, sampled_frame_number_filtered)
    
    return plot, sampled_frame_mapping_filtered, sampled_frame_number_filtered, assignments_filtered, mp4filepath, csvfilepath

def save_images(mp4filepath, csvfilepath, folder_path, sampled_frame_mapping_filtered, sampled_frame_number_filtered, assignments_filtered, keypoints):
    if not os.path.isdir(os.path.join(folder_path,'clusters')):
        os.mkdir(os.path.join(folder_path,'clusters'))
    if not os.path.isdir(os.path.join(folder_path,'plots')):
        os.mkdir(os.path.join(folder_path,'plots'))

    file_j_df = pd.read_csv(csvfilepath, low_memory=False)          
    file_j_df_array = np.array(file_j_df)
    mp4 = cv2.VideoCapture(mp4filepath)
        
    clusters = np.unique(assignments_filtered)
    for cluster in clusters:
        if not os.path.isdir(os.path.join(folder_path,'clusters',str(cluster))):
            os.mkdir(os.path.join(folder_path,'clusters',str(cluster)))
            
            indeces = np.where(assignments_filtered==cluster)[0]

            frames = sampled_frame_mapping_filtered[indeces]
            frames = frames.astype(int)
            sorted_frames = np.sort(frames)
            differences = np.diff(sorted_frames)
            min_diff = np.min(differences)
            max_diff = np.max(differences)
            bins = np.arange(min_diff, max_diff + 2)
            matplotlib.use('Agg')
            plt.hist(differences, bins=bins, edgecolor='black')  
            plt.title(f'Cluster {cluster}')
            plt.xlabel('Frame Difference')
            plt.ylabel('Frequency')
            plt.savefig(os.path.join(folder_path,'plots',str(cluster)+".png")) 
            plt.clf()

            for index in indeces:

                frame_number = sampled_frame_number_filtered[index]
                frame_mapping = sampled_frame_mapping_filtered[index]
                
                try:
                    xy = [(int(float(x)), int(float(y))) for x, y in xy]
                except:
                    print(f"frame_mapping_filtered: {sampled_frame_mapping_filtered}")
                    print(f"index: {index}")
                
                mp4.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = mp4.read()
                
                if keypoints: 
                    keypoint_data = file_j_df_array[np.where(file_j_df_array[:,0]==str(frame_mapping))][0]
                
                    x = keypoint_data[1::3]
                    y = keypoint_data[2::3]

                    xy = np.concatenate([x.reshape(-1, 1), y.reshape(-1, 1)], axis=1)
                    for point in xy: 
                        cv2.circle(frame, point, radius=5, color=(0, 0, 255), thickness = -1)
            
                cv2.imwrite(os.path.join(folder_path,'clusters',str(cluster),str(frame_mapping)+".png"),frame)
                
    mp4.release()

    