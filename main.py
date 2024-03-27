from flask import Flask, render_template, request, jsonify, session
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from wtforms.validators import InputRequired
from werkzeug.utils import secure_filename
import webbrowser
import threading

import plotly
import plotly.graph_objs as go

import pandas as pd
import numpy as np
import cv2
import json
import os
import io

import base64

from bsoid_utils import *

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
app.config['SECRET_KEY'] = 'secretkey'
app.config['TEMPLATES_AUTO_RELOAD'] = True

def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000/')

class UploadForm(FlaskForm):
    file = FileField("CSV File")
    video = FileField("Video File")
    submit = SubmitField("Generate UMAP Embedding")
# class UploadMP4Form(FlaskForm):
#     video = FileField("Video File")

@app.route('/process_click_data', methods=['POST'])
def process_click_data():
    click_data = request.get_json()
    frame_number = click_data[0]['frame'] if click_data else None

    if frame_number is not None:
        mp4filepath = session.get('mp4', None)  # Get the mp4 object from the application context
        if mp4filepath is not None:
            mp4 = cv2.VideoCapture(mp4filepath)
            mp4.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = mp4.read()
            mp4.release()

            if ret:
                # Convert the frame to a format that can be sent as a JSON response
                _, buffer = cv2.imencode('.jpg', frame)
                frame_data = base64.b64encode(buffer).decode('utf-8')

                return jsonify({'frame_data': frame_data})
            else:
                return jsonify({'error': 'Failed to retrieve frame'}), 400
        else:
            return jsonify({'error': 'Video not loaded'}), 400
    else:
        return jsonify({'error': 'Missing frame number'}), 400


@app.route('/', methods = ["GET","POST"])
@app.route('/home', methods = ["GET","POST"])
def home():
    plot = None
    form = UploadForm()
    # mp4form = UploadMP4Form()
   
    if form.validate_on_submit():
        csvfile = form.file.data
        mp4file = form.video.data
        if not os.path.isdir('uploads'):
            os.mkdir('uploads')
        if not os.path.isdir(os.path.join('uploads', 'csvs')):
            os.mkdir(os.path.join('uploads', 'csvs'))
        if not os.path.isdir(os.path.join('uploads', 'videos')):
            os.mkdir(os.path.join('uploads', 'videos'))
        
        csvfilepath = os.path.join('uploads', 'csvs', csvfile.filename)
        csvfile.save(csvfilepath)
        file_j_df = pd.read_csv(csvfilepath, low_memory=False)
        
        mp4filepath = os.path.join('uploads', 'videos', mp4file.filename)
        mp4file.save(mp4filepath)
        mp4 = cv2.VideoCapture(mp4filepath)
        session['mp4'] = mp4filepath

        pose_chosen = []

        #file_j_df = pd.read_csv(file, low_memory=False)
       
        file_j_df_array = np.array(file_j_df)
        p = st.multiselect('Identified __pose__ to include:', [*file_j_df_array[0, 1:-1:3]], [*file_j_df_array[0, 1:-1:3]])
        for a in p:
            index = [i for i, s in enumerate(file_j_df_array[0, 1:]) if a in s]
            if not index in pose_chosen:
                pose_chosen += index
        pose_chosen.sort()

        file_j_processed, p_sub_threshold = adp_filt(file_j_df, pose_chosen)
        file_j_processed = file_j_processed.reshape((1, file_j_processed.shape[0], file_j_processed.shape[1]))

        scaled_features, features, frame_mapping = compute(file_j_processed, fps)

        train_size = subsample(file_j_processed, fps)

        sampled_embeddings = learn_embeddings(scaled_features, features, UMAP_PARAMS, train_size)

        assignments = hierarchy(cluster_range, sampled_embeddings, HDBSCAN_PARAMS)
        
        plot = create_plotly(sampled_embeddings, assignments, csvfile, frame_mapping)
   
    return render_template('index.html', form=form, graphJSON=plot)

if __name__ == '__main__':
    threading.Thread(target=open_browser).start()
    app.run(debug = True)