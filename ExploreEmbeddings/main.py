from flask import Flask, render_template, request, jsonify, session 
from flask_wtf import FlaskForm
from wtforms import FloatField, SubmitField, StringField, HiddenField, BooleanField, IntegerField, Field
from wtforms.widgets import Input
import webbrowser
import threading
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase

# from celery import Celery

import plotly.graph_objs as go

import pandas as pd
import numpy as np
import cv2
import os

import base64

from tasks import return_plot, save_images

fps = 30     

app = Flask(__name__)
SECRET_KEY = os.urandom(32)
app.config['SECRET_KEY'] = 'secretkey'
app.config['TEMPLATES_AUTO_RELOAD'] = True

# see https://flask-sqlalchemy.palletsprojects.com/en/3.1.x/ 
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'  
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False


# app.config['CELERY_BROKER_URL'] = 'http://127.0.0.1:5000/'
# app.config['CELERY_RESULT_BACKEND'] = 'http://127.0.0.1:5000/'

# celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
# celery.conf.update(app.config)

db = SQLAlchemy(app)

class SessionData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    folder_path = db.Column(db.String(500))
    csv_path = db.Column(db.String(500))
    mp4_path = db.Column(db.String(500))
    assignments_filtered = db.Column(db.PickleType)
    sampled_frame_number_filtered = db.Column(db.PickleType)
    sampled_frame_mapping_filtered = db.Column(db.PickleType)
    keypoints = db.Column(db.Boolean) 

with app.app_context():
    db.create_all()

class FractionWidget(Input):
    input_type = 'range'

    def __call__(self, field, **kwargs):
        kwargs.setdefault('id', field.id)
        kwargs.setdefault('type', self.input_type)
        kwargs.setdefault('min', 0.05)
        kwargs.setdefault('max', 1)
        kwargs.setdefault('step', 0.05)  # Adjust step as needed
        return super(FractionWidget, self).__call__(field, **kwargs)

def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000/')
    
class UploadForm(FlaskForm):
    action = HiddenField(default='upload')
    folder = StringField('Provide the path to the folder containing the csv and mp4 files:')
    upload = SubmitField('Step 1: Generate UMAP Embedding')

class FractionForm(FlaskForm):
    action = HiddenField(default='adjust')
    slider = FloatField('Set the training input fraction within the range of 0.05 to 1:', widget=FractionWidget())

class KeypointForm(FlaskForm):
    action = HiddenField(default='upload')
    keypoints = BooleanField('Generate DLC keypoints?', default=True)

class ParameterForm(FlaskForm):
    action = HiddenField(default='parameters')
    umap_min_dist = FloatField('Set the UMAP min distance:', default=0.0, description="The effective minimum distance between embedded points.")
    umap_random_state = IntegerField('Set the UMAP random state:', default=42, description="The random state to ensure reproducibility (default is 42).")
    hdbscan_min_samples = IntegerField('Set the HDBSCAN min samples:', default=1, description="The number of samples in a neighborhood for a point to be considered as a core point.")
    hdbscan_eps_min = FloatField('Set the HDBSCAN min epsilon:', default=0.5, description="The minimum radius in which neighboring points will be considered part of the cluster.")
    hdbscan_eps_max = FloatField('Set the HDBSCAN max epsilon:', default=1.0, description="The maximum radius in which neighboring points will be considered part of the cluster.")

class ClusterForm(FlaskForm):
    action = HiddenField(default='cluster')
    cluster = SubmitField('Step 2: Save images in clusters')

@app.route('/process_click_data', methods=['POST'])
def process_click_data():
    click_data = request.get_json()
    frame_mapping = click_data[0]['frame_mapping'] if click_data else None
    frame_number = click_data[0]['frame_number']

    if frame_mapping is not None:
        
        session_data_id = session.get('session_data_id')

        if session_data_id:
            session_data = SessionData.query.get(session_data_id)
            if session_data:
                csvfilepath = session_data.csv_path
                mp4filepath = session_data.mp4_path
                keypoints = session_data.keypoints

        file_j_df = pd.read_csv(csvfilepath, low_memory=False)          
        file_j_df_array = np.array(file_j_df)

        if mp4filepath is not None:
            mp4 = cv2.VideoCapture(mp4filepath)
            mp4.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = mp4.read()
            mp4.release()

            if ret: 
                if keypoints:

                    keypoint_data = file_j_df_array[np.where(file_j_df_array[:,0]==str(frame_mapping))][0]

                    x = keypoint_data[1::3] 
                    y = keypoint_data[2::3]

                    xy = np.concatenate([x.reshape(-1, 1), y.reshape(-1, 1)], axis=1)
                    xy = [(int(float(x)), int(float(y))) for x, y in xy]

                    for point in xy: 
                        cv2.circle(frame, point, radius=5, color=(0, 0, 255), thickness = -1)
                
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
    fractionform = FractionForm()
    keypointform = KeypointForm()
    parameterform = ParameterForm()

    if uploadform.validate_on_submit() and uploadform.upload.data: 
        db.session.query(SessionData).delete()
        db.session.commit()

        folder_path = uploadform.folder.data
        training_fraction = fractionform.slider.data
        keypoints = keypointform.keypoints.data
        mindist = parameterform.umap_min_dist.data
        randomstate = parameterform.umap_random_state.data
        minsamples=parameterform.hdbscan_min_samples.data
        min_eps = parameterform.hdbscan_eps_min.data
        max_eps = parameterform.hdbscan_eps_max.data

        session['folder_path'] = folder_path

        UMAP_PARAMS = {
            'min_dist': mindist,  
            'random_state': randomstate,
        }

        HDBSCAN_PARAMS = {
            'min_samples': minsamples,
        }

        cluster_range = [min_eps, max_eps]

        plot, sampled_frame_mapping_filtered, sampled_frame_number_filtered, assignments_filtered, mp4filepath, csvfilepath = return_plot(folder_path, fps, UMAP_PARAMS, cluster_range, HDBSCAN_PARAMS, training_fraction)

        session_data = SessionData(
            folder_path=folder_path,
            csv_path=csvfilepath,
            mp4_path=mp4filepath,
            assignments_filtered=assignments_filtered,
            sampled_frame_number_filtered=sampled_frame_number_filtered,
            sampled_frame_mapping_filtered=sampled_frame_mapping_filtered,
            keypoints=keypoints
        )

        db.session.add(session_data)
        db.session.commit()

        session['session_data_id'] = session_data.id

        
    if clusterform.validate_on_submit() and clusterform.cluster.data:

        session_data_id = session.get('session_data_id')

        if session_data_id:
            session_data = SessionData.query.get(session_data_id)
            if session_data:
                csvfilepath = session_data.csv_path
                mp4filepath = session_data.mp4_path
                folder_path = session_data.folder_path
                assignments_filtered = session_data.assignments_filtered
                sampled_frame_number_filtered = session_data.sampled_frame_number_filtered
                sampled_frame_mapping_filtered = session_data.sampled_frame_mapping_filtered
                keypoints=keypoints
        
        save_images(mp4filepath, csvfilepath, folder_path, sampled_frame_mapping_filtered, sampled_frame_number_filtered, assignments_filtered, keypoints)
    
    return render_template('index.html', uploadform=uploadform, clusterform=clusterform, fractionform=fractionform, keypointform=keypointform, parameterform=parameterform, graphJSON = plot)

if __name__ == '__main__':
    threading.Thread(target=open_browser).start()
    app.run(debug = False)