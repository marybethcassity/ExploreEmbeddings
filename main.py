from flask import Flask, render_template, request, jsonify, session 
from flask_wtf import FlaskForm
from wtforms import FloatField, SubmitField, StringField, HiddenField, Field
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
    
# class ClusterSizeWidget:
#     def __call__(self, field, **kwargs):
#         kwargs.setdefault('type', 'range')
#         kwargs.setdefault('min', field.min)
#         kwargs.setdefault('max', field.max)
#         kwargs.setdefault('step', field.step)
#         # The value will not be directly set here because HTML does not support ranges as values directly
#         return HTMLString(f'<input {html_params(name=field.name, **kwargs)}>' 
#                           f'<input {html_params(name=field.name, **kwargs)}>')
    
# class RangeSliderField(Field):
#     widget = ClusterSizeWidget()

#     def __init__(self, label=None, validators=None, min=0, max=100, step=1, **kwargs):
#         super(RangeSliderField, self).__init__(label, validators, **kwargs)
#         self.min = min
#         self.max = max
#         self.step = step

def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000/')

# class UMAPForm(FlaskForm):
#     action = HiddenField(default='upload')
#     folder = FloatField('Provide the path to the folder containing the csv and mp4 files:')
#     upload = SubmitField('Step 1: Generate UMAP Embedding')    

class HDBSCANForm(FlaskForm):
    action = HiddenField(default='upload')
    folder = FloatField('Provide the path to the folder containing the csv and mp4 files:')
    upload = SubmitField('Step 1: Generate UMAP Embedding')
    
class UploadForm(FlaskForm):
    action = HiddenField(default='upload')
    folder = StringField('Provide the path to the folder containing the csv and mp4 files:')
    upload = SubmitField('Step 1: Generate UMAP Embedding')

class FractionForm(FlaskForm):
    action = HiddenField(default='adjust')
    slider = FloatField('Set the training input fraction within the range of 0.05 to 1:', widget=FractionWidget())

# class ClusterSizeForm(FlaskForm):
#     range_slider = RangeSliderField('Set the range of the minimum cluster size within the range of 0.02 to 1:', min=0.02, max=5.00, step=0.02)

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

        #csvfilepath = session.get('csv')
        file_j_df = pd.read_csv(csvfilepath, low_memory=False)          
        file_j_df_array = np.array(file_j_df)
        #mp4filepath = session.get('mp4')
        if mp4filepath is not None:
            mp4 = cv2.VideoCapture(mp4filepath)
            mp4.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = mp4.read()
            mp4.release()

            if ret:

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
    #clustersizeform = ClusterSizeForm()

    if uploadform.validate_on_submit() and uploadform.upload.data: 
        db.session.query(SessionData).delete()
        db.session.commit()
    
        folder_path = uploadform.folder.data
        training_fraction = fractionform.slider.data
        #range_values = clustersizeform.form.getlist('range_slider')
        #session['folder_path'] = folder_path

        plot, sampled_frame_mapping_filtered, sampled_frame_number_filtered, assignments_filtered, mp4filepath, csvfilepath = return_plot(folder_path, fps, UMAP_PARAMS, cluster_range, HDBSCAN_PARAMS, training_fraction)

        #session['csv'] = csvfilepath
        #session['mp4'] = mp4filepath
        #session['assignments_filtered'] = assignments_filtered.tolist()
        #session['sampled_frame_number_filtered'] = sampled_frame_number_filtered.tolist()
        #session['sampled_frame_mapping_filtered'] = sampled_frame_mapping_filtered.tolist()
        ##session['plot'] = plot

        session_data = SessionData(
            folder_path=folder_path,
            csv_path=csvfilepath,
            mp4_path=mp4filepath,
            assignments_filtered=assignments_filtered,
            sampled_frame_number_filtered=sampled_frame_number_filtered,
            sampled_frame_mapping_filtered=sampled_frame_mapping_filtered,
        )

        db.session.add(session_data)
        db.session.commit()

        session['session_data_id'] = session_data.id

        
    if clusterform.validate_on_submit() and clusterform.cluster.data:
        
        #csvfilepath = session.get('csv')
        #mp4filepath = session.get('mp4')
        #folder_path = session.get('folder_path')
        #assignments_filtered = session.get('assignments_filtered')
        #sampled_frame_number_filtered = session.get('sampled_frame_number_filtered')
        #sampled_frame_mapping_filtered = session.get('sampled_frame_mapping_filtered')

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
        
        save_images(mp4filepath, csvfilepath, folder_path, sampled_frame_mapping_filtered, sampled_frame_number_filtered, assignments_filtered)
    
    return render_template('index.html', uploadform=uploadform, clusterform=clusterform, fractionform=fractionform, graphJSON = plot)

if __name__ == '__main__':
    threading.Thread(target=open_browser).start()
    app.run(debug = False)