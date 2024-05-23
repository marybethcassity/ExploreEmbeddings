import numpy as np
import os 
import cv2
import shutil
import matplotlib.pyplot as plt 
import matplotlib
import json
import time
import re

#from celery import Celery

from bsoid_utils import *


def return_plot(folder_path, fps, UMAP_PARAMS, cluster_range, HDBSCAN_PARAMS, training_fraction, name):
    if not os.path.isdir('uploads'):
        os.mkdir('uploads')
    if not os.path.isdir(os.path.join('uploads', 'csvs')):
        os.mkdir(os.path.join('uploads', 'csvs'))
    if not os.path.isdir(os.path.join('uploads', 'videos')):
        os.mkdir(os.path.join('uploads', 'videos'))

    dict_data = {
    'fps': fps,
    'fraction': training_fraction,
    'UMAP_min': UMAP_PARAMS['min_dist'],
    'UMAP_seed': UMAP_PARAMS['random_state'],
    'HDBSCAN_samples': HDBSCAN_PARAMS['min_samples'],
    'HDBSCAN_min': cluster_range[0],
    'HDBSCAN_max': cluster_range[1],
    'files': {}
}
    def numerical_sort_key(s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    csv_files_sorted = sorted(csv_files, key=numerical_sort_key)

    for filename in csv_files_sorted:
        if filename.endswith('.csv'):
            csvfilepath = os.path.join('uploads', 'csvs', filename)
            #csvfilename = filename
            shutil.copyfile(os.path.join(folder_path,filename), csvfilepath)
            file_j_df = pd.read_csv(csvfilepath, low_memory=False)   
            basename = filename.split("DLC")[0] 
            mp4_name = basename+".mp4"
        
            if os.path.isfile(os.path.join(folder_path,mp4_name)):
                
                mp4filepath = os.path.join('uploads', 'videos', mp4_name)
                shutil.copyfile(os.path.join(folder_path,mp4_name), mp4filepath) 
                
                pose_chosen = []

                file_j_df_array = np.array(file_j_df)

                p = st.multiselect('Identified __pose__ to include:', [*file_j_df_array[0, 1:-1:3]], [*file_j_df_array[0, 1:-1:3]])
                for a in p:
                    index = [i for i, s in enumerate(file_j_df_array[0, 1:]) if a in s]
                    if not index in pose_chosen:
                        pose_chosen += index
                pose_chosen.sort()
                
                dict_data['files'][basename] = {
                'mp4_path': mp4filepath,
                'csv_path': csvfilepath,
                'data_frame': file_j_df,
                'data_array': file_j_df_array,
                'pose_chosen': pose_chosen
            }
            else:
                print("No matching mp4 exists for {filename}.") 
        
        # if filename.endswith('.mp4'):
        #     mp4_paths.append(os.path.join('uploads', 'videos', filename))
        #     mp4filepath = os.path.join('uploads', 'videos', filename)
        #     shutil.copyfile(os.path.join(folder_path,filename), mp4filepath)

    # time_start = time.time()
            
    for basename, content in dict_data['files'].items():
        print(basename)
        file_j_processed, p_sub_threshold = adp_filt(content['data_frame'], content['pose_chosen'])

        # time_adp_filt = time.time()-time_start

        file_j_processed = file_j_processed.reshape((1, file_j_processed.shape[0], file_j_processed.shape[1]))

        dict_data['files'][basename]['file_j_processed'] = file_j_processed

        # time_start_2 = time.time()

        scaled_features, features, frame_mapping, frame_number = compute(file_j_processed, dict_data['files'][basename]['data_array'], fps)

        dict_data['files'][basename]['scaled_features'] = scaled_features
        dict_data['files'][basename]['features'] = features
        dict_data['files'][basename]['frame_mapping'] = frame_mapping
        dict_data['files'][basename]['frame_number'] = frame_number

        # time_compute = time.time()- time_start_2
        # time_start_2 = time.time()

        sampled_input_feats, sampled_frame_mapping, sampled_frame_number = subsample(file_j_processed, fps, training_fraction, scaled_features, frame_mapping, frame_number)

        dict_data['files'][basename]['sampled_input_feats'] = sampled_input_feats
        dict_data['files'][basename]['sampled_frame_mapping'] = sampled_frame_mapping
        dict_data['files'][basename]['sampled_frame_number'] = sampled_frame_number

        # time_subsample = time.time()-time_start_2
        # time_start_2 = time.time()

    # concatenate all scaled features and sampled input feats --> learn_embeddings

    sampled_embeddings, data, basename_mappings, csv_mappings, frame_mappings, frame_numbers, features  = learn_embeddings(UMAP_PARAMS, dict_data)
    
    
    # time_learn_embeddings = time.time()-time_start_2
    # time_start_2 = time.time()

    assignments = hierarchy(cluster_range, sampled_embeddings, HDBSCAN_PARAMS)

    # time_hierarchy = time.time()-time_start_2
    # time_total = time.time()-time_start

    # sampled_embeddings_filtered = sampled_embeddings[assignments>=0]
    # assignments_filtered = assignments[assignments>=0]    
    # sampled_frame_mapping_filtered = sampled_frame_mapping[assignments>=0] 
    # sampled_frame_number_filtered =  sampled_frame_number[assignments>=0] 

    data = { 
        "mapping": frame_mappings, 
        "frame_number":  frame_numbers,
        "assignments":  assignments,
        "basenames": basename_mappings,
        "csvs": csv_mappings,
        "fps" : fps,
        "fraction" : training_fraction,
        "UMAP_min" : UMAP_PARAMS['min_dist'],
        "UMAP_seed" : UMAP_PARAMS['random_state'],
        "HDBSCAN_samples" : HDBSCAN_PARAMS['min_samples'],
        "HDBSCAN_min" : cluster_range[0],
        "HDBSCAN_max" : cluster_range[1]}
    
    df = pd.DataFrame(data)
    
    if not os.path.isdir(os.path.join(folder_path, name)):
        os.mkdir(os.path.join(folder_path, name))

    features_df = pd.DataFrame(features.T)
    features_df.to_csv(os.path.join(folder_path, name, 'features.csv'),header=False, float_format='%.5f')

    df.to_csv(os.path.join(folder_path, name, "data.csv"), index=False) 

    np.save(os.path.join(folder_path, name, "embedding.npy"), sampled_embeddings)

    plot = create_plotly(sampled_embeddings, assignments, frame_mappings, frame_numbers, basename_mappings, csv_mappings)
    
    plot.write_html(os.path.join(folder_path, name,'plot.html'))
    
    graphJSON = json.dumps(plot, cls=plotly.utils.PlotlyJSONEncoder)
    with open(os.path.join(folder_path, name,'plot.json'), 'w') as f:
        f.write(graphJSON)

    return graphJSON, frame_mappings, frame_numbers, assignments, basename_mappings, csv_mappings, sampled_embeddings

def save_images(folder_path, frame_mappings, frame_numbers, assignments, basename_mappings, csv_mappings, keypoints, name):
    
    if not os.path.isdir(os.path.join(folder_path, name, 'clusters')):
        os.mkdir(os.path.join(folder_path, name,'clusters'))
    if not os.path.isdir(os.path.join(folder_path, name,'histograms')):
        os.mkdir(os.path.join(folder_path, name,'histograms'))

    #print(os.path.join(folder_path,name))
    #file_j_df = pd.read_csv(csvfilepath, low_memory=False)          
    #file_j_df_array = np.array(file_j_df)
    #mp4 = cv2.VideoCapture(mp4filepath)
        
    clusters = np.unique(assignments)
    for cluster in clusters:
        if not os.path.isdir(os.path.join(folder_path, name,'clusters',str(cluster))):
            os.mkdir(os.path.join(folder_path, name,'clusters',str(cluster)))

            indeces = np.where(assignments==cluster)[0]

            frame_mappings = np.asarray(frame_mappings)
            frames = frame_mappings[indeces]
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
            plt.savefig(os.path.join(folder_path, name,'histograms',str(cluster)+".png")) 
            plt.clf()

            for index in indeces:

                frame_number = frame_numbers[index]
                frame_mapping = frame_mappings[index]
                mp4 = cv2.VideoCapture(os.path.join(folder_path,basename_mappings[index]+".mp4"))
                
                mp4.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = mp4.read()
                
                if keypoints: 
                    file_j_df = pd.read_csv(os.path.join(folder_path,csv_mappings[index]), low_memory=False)         
                    file_j_df_array = np.array(file_j_df)

                    keypoint_data = file_j_df_array[np.where(file_j_df_array[:,0]==str(frame_mapping))][0]
                
                    x = keypoint_data[1::3]
                    y = keypoint_data[2::3]

                    xy = np.concatenate([x.reshape(-1, 1), y.reshape(-1, 1)], axis=1)
                    xy = [(int(float(x)), int(float(y))) for x, y in xy]

                    for point in xy: 
                        cv2.circle(frame, point, radius=5, color=(0, 0, 255), thickness = -1)
            
                cv2.imwrite(os.path.join(folder_path, name, 'clusters',str(cluster),basename_mappings[index]+"_"+str(frame_mapping)+".png"),frame)
                
    mp4.release()

    
