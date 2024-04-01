import numpy as np
import os 
import cv2

#from celery import Celery

#from bsoid_utils import learn_embeddings, hierarchy, create_plotly

def save_images(mp4filepath, folder_path, frame_mapping_filtered, assignments_filtered):
    if not os.path.isdir(os.path.join(folder_path,'newdir')):
        os.mkdir(os.path.join(folder_path,'newdir'))
    #mp4 = cv2.VideoCapture(mp4filepath)
    
    clusters = np.unique(assignments_filtered)
    for cluster in clusters:
        if not os.path.isdir(os.path.join(folder_path,str(cluster))):
            os.mkdir(os.path.join(folder_path,str(cluster)))

            indeces = np.where(assignments_filtered==cluster)[0]
            for index in indeces:
                #mp4.set(cv2.CAP_PROP_POS_FRAMES, frame_mapping_filtered[index])
                #ret, frame = mp4.read()
                #cv2.imwrite(os.path.join(os.path.join(folder_path,str(cluster)),str(frame)+".png"),frame)
                pass
    #mp4.release()
