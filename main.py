from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_wtf import FlaskForm
from wtforms import  SubmitField, StringField, HiddenField
from werkzeug.utils import secure_filename
import webbrowser
import threading

# from celery import Celery

import plotly.graph_objs as go

import pandas as pd
import numpy as np
import cv2
import os

import base64

from tasks import return_plot, save_images

cluster_range = [0.5, 1]

fps = 30     

# bsoid_umap/config/GLOBAL_CONFIG.py # edited 
UMAP_PARAMS = {
    'min_dist': 0.0,  # small value
    'random_state': 42,
}

HDBSCAN_PARAMS = {
    'min_samples': 1  # small value
}

app = Flask(__name__)
SECRET_KEY = os.urandom(32)
app.config['SECRET_KEY'] = 'secretkey'
app.config['TEMPLATES_AUTO_RELOAD'] = True

# app.config['CELERY_BROKER_URL'] = 'http://127.0.0.1:5000/'
# app.config['CELERY_RESULT_BACKEND'] = 'http://127.0.0.1:5000/'

# celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
# celery.conf.update(app.config)

def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000/')
    
class UploadForm(FlaskForm):
    action = HiddenField(default='upload')
    folder = StringField('Provide the path to the folder containing the csv and mp4 file:')
    upload = SubmitField('Generate UMAP Embedding')

class ClusterForm(FlaskForm):
    action = HiddenField(default='cluster')
    cluster = SubmitField('Save images in clusters')

@app.route('/process_click_data', methods=['POST'])
def process_click_data():
    click_data = request.get_json()
    frame_number = click_data[0]['frame'] if click_data else None

    if frame_number is not None:
        file_j_df = session.get('file_j_df') 
        mp4filepath = session.get('mp4')  
        frame_mapping_filtered = session.get('frame_mapping_filtered')

        if mp4filepath is not None:
            mp4 = cv2.VideoCapture(mp4filepath)
            mp4.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = mp4.read()
            mp4.release()

            if ret:

                # keypoint_data = file_j_df.iloc[frame_mapping_filtered[frame_number]]

                # x = keypoint_data.iloc[1::3].reset_index(drop=True).to_numpy()
                # y = keypoint_data.iloc[2::3].reset_index(drop=True).to_numpy()

                # xy = np.concatenate([x.reshape(-1, 1), y.reshape(-1, 1)], axis=1)
                # xy = [(int(float(x)), int(float(y))) for x, y in xy]

                # for point in xy: 
                #     cv2.circle(frame, point, radius=10, color=(255, 0, 0), thickness = -1)
                
                _, buffer = cv2.imencode('.jpg', frame)
                frame_data = base64.b64encode(buffer).decode('utf-8')

                return jsonify({'frame_data': frame_data})
            else:
                return jsonify({'error': 'Failed to retrieve frame'}), 400
        else:
            return jsonify({'error': 'Video not loaded'}), 400
    else:
        return jsonify({'error': 'Missing frame number'}), 400


@app.route('/', methods = ["GET", "POST"])
@app.route('/home', methods = ["GET", "POST"])
def home():
    plot = None
    folder_path = None
    file_j_df = None
    uploadform = UploadForm()
    clusterform = ClusterForm()

    if uploadform.validate_on_submit() and uploadform.upload.data: 
    
        folder_path = uploadform.folder.data
        session['folder_path'] = folder_path

        plot, frame_mapping_filtered, assignments_filtered, mp4filepath, file_j_df = return_plot(folder_path, fps, UMAP_PARAMS, cluster_range, HDBSCAN_PARAMS)

        # session['file_j_df'] = file_j_df.to_json()
        session['mp4'] = mp4filepath
        session['assignments_filtered'] = assignments_filtered.tolist()
        session['frame_mapping_filtered'] = frame_mapping_filtered.tolist()
        session['plot'] = plot
        
    if clusterform.validate_on_submit() and clusterform.cluster.data:
        
        mp4filepath = session.get('mp4')
        folder_path = session.get('folder_path')
        frame_mapping_filtered = session.get('frame_mapping_filtered')
        assignments_filtered = session.get('assignments_filtered')

        save_images(mp4filepath, folder_path, frame_mapping_filtered, assignments_filtered)
    
    return render_template('index.html', uploadform=uploadform, clusterform=clusterform, graphJSON=session.get('plot', None))

if __name__ == '__main__':
    threading.Thread(target=open_browser).start()
    app.run(debug = True)