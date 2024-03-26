from flask import Flask, render_template, request, jsonify
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

@app.route('/relayout', methods=['POST'])
def relayout():
    # Get the event data from the request
    event_data = json.loads(request.data)

    # Check if the event is a 'plotly_relayout' event
    if 'plotly_relayout' in event_data['event']:
        # Get the selected point's ID
        selected_id = event_data['event']['plotly_relayout']['selection']

        # You can perform any desired operation with the selected ID
        print(f'Selected point ID: {selected_id}')

        # You can return a JSON response if needed
        return f'You selected point with ID: {selected_id}'

    return jsonify({"selected_id": selected_id})

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

        scaled_features, features = compute(file_j_processed, fps)

        train_size = subsample(file_j_processed, fps)

        sampled_embeddings = learn_embeddings(scaled_features, features, UMAP_PARAMS, train_size)

        assignments = hierarchy(cluster_range, sampled_embeddings, HDBSCAN_PARAMS)
        
        plot = create_plotly(sampled_embeddings, assignments, csvfile)
   
    return render_template('index.html', form=form, graphJSON=plot)

if __name__ == '__main__':
    threading.Thread(target=open_browser).start()
    app.run(debug = True)