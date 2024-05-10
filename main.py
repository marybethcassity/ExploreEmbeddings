from flask import Flask, render_template, request, jsonify, session, redirect, url_for 
from flask_wtf import FlaskForm
from wtforms import FloatField, SubmitField, StringField, HiddenField, BooleanField, IntegerField, Field, SelectField
from wtforms.widgets import Input, NumberInput
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
import shutil
from sklearn.neighbors import NearestNeighbors as NN

import base64

from tasks import return_plot, save_images  

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

#  session_data = SessionData(
#             folder_path=folder_path,
#             assignments=assignments,
#             frame_numbers=frame_numbers,
#             frame_mappings=frame_mappings,
#             basename_mappings = basename_mappings,
#             csv_mappings = csv_mappings,
#             keypoints=keypoints,
#             name = name
#         )

class SessionData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    folder_path = db.Column(db.String(500))
    csv_path = db.Column(db.String(500))
    mp4_path = db.Column(db.String(500))
    assignments = db.Column(db.PickleType)
    frame_numbers = db.Column(db.PickleType)
    frame_mappings = db.Column(db.PickleType)
    keypoints = db.Column(db.Boolean)
    name = db.Column(db.String(500))
    basename_mappings = db.Column(db.PickleType)
    csv_mappings = db.Column(db.PickleType)
    embeddings = db.Column(db.PickleType)

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

class PlotlyForm(FlaskForm):
    action = HiddenField(default='upload')
    load_plot = BooleanField('Load plotly embedding if previously generated?', default=True)

class LoadNameForm(FlaskForm):
    action = HiddenField(default='upload')
    loadname = SelectField('Which plot do you want to open?')

class NameForm(FlaskForm):
    action = HiddenField(default='upload')
    name = StringField('What do you want to name this plot?')

class FractionForm(FlaskForm):
    action = HiddenField(default='adjust')
    slider = FloatField('Set the training input fraction within the range of 0.05 to 1 (default: 1):', default=1, widget=FractionWidget())

class KeypointForm(FlaskForm):
    action = HiddenField(default='upload')
    keypoints = BooleanField('Generate DLC keypoints?', default=True)

class ParameterForm(FlaskForm):
    action = HiddenField(default='parameters')
    fps = IntegerField('Set the fps (default: 30):', default=30, widget=NumberInput(step=1))
    umap_min_dist = FloatField('Set the UMAP min distance (default: 0.0):', default=0.0, widget=NumberInput(step=0.1, min = 0))
    umap_random_state = IntegerField('Set the UMAP random seed (default: 42):', default=42, widget=NumberInput(step=1))
    hdbscan_min_samples = IntegerField('Set the HDBSCAN min samples (default: 1):', default=1, widget=NumberInput(step=1, min = 0))
    hdbscan_cluster_min = FloatField('Set the percent of dataset for HDBSCAN min cluster size (default: 0.5):', default=0.5, widget=NumberInput(step=0.1, min = 0))
    hdbscan_cluster_max = FloatField('Set the percent of dataset for HDBSCAN max cluster size (default: 1.0):', default=1.0, widget=NumberInput(step=0.1, min = 0))

class ClusterForm(FlaskForm):
    action = HiddenField(default='cluster')
    cluster = SubmitField('Step 2: Save images in clusters')

@app.route('/get_folders', methods=['POST'])
def get_folders():
    data = request.get_json()
    directory = data.get('path')

    if os.path.exists(directory) and os.path.isdir(directory):
        folders = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
        return jsonify(folders)
    else:
        pass

@app.route('/process_click_data', methods=['POST'])
def process_click_data():
    data = request.get_json()
    click_data = data['clickData']
    radio_button_value = data['radioButtonValue']
    
    session['last_click_info'] = click_data

    if click_data: 
        frame_mapping = int(click_data[0]['frame_mapping']) if click_data else None
        frame_number = int(click_data[0]['frame_number'])
        frame_assignment = int(click_data[0]['assignment'])
        basename = str(click_data[0]['basename'])
        csv_name = str(click_data[0]['csv'])

        if frame_mapping is not None:
            
            session_data_id = session.get('session_data_id')

            if session_data_id:
                session_data = SessionData.query.get(session_data_id)
                if session_data:
                    folder_path = session_data.folder_path 
                    keypoints = session_data.keypoints
                    mappings = np.array(session_data.frame_mappings).astype(int)
                    numbers = np.array(session_data.frame_numbers).astype(int)
                    assignments = np.array(session_data.assignments).astype(int)
                    basenames = session_data.basename_mappings 
                    csvs = session_data.csv_mappings
                    embeddings = np.array(session_data.embeddings).astype(float)

    
            frame_images = []
            frames = []
            frame_assignments = []

            frame_numbers = []
            frame_mappings = []
            frame_basenames = []
            frame_csvs = []
            
            window = 5

            #start_mapping = np.where(frame_mapping==mapping)
            if radio_button_value == 'single':
                window = 0
                frame_numbers.append(frame_number)
                frame_mappings.append(frame_mapping)
                frame_assignments.append(frame_assignment)
                frame_basenames.append(basename)
                frame_csvs.append(csv_name)

                start_index = window
            
            elif radio_button_value == 'sequential_mp4':
                csvfilepath = os.path.join(folder_path,csv_name)

                file_j_df = pd.read_csv(csvfilepath, low_memory=False)          
                file_j_df_array = np.array(file_j_df) 

                for i in range(max(0, frame_number - window), min(len(file_j_df_array), frame_number + window + 1)):
                    frame_numbers.append(i)   
        
                for j in range(max(0, frame_mapping - window), min(len(file_j_df_array)+int(file_j_df_array[2, 0]), frame_mapping + window + 1)):
                    frame_mappings.append(j)
                    if j in mappings:
                        index = np.where(mappings==j)[0][0]
                        frame_assignments.append(int(assignments[index]))
                    else: 
                        frame_assignments.append('')

                frame_basenames = [basename]*len(frame_numbers)
                frame_csvs = [csv_name]*len(frame_numbers)

                start_index = window

            elif radio_button_value == 'sequential_cluster':
                indeces = np.where(assignments==frame_assignment)
                
                frame_numbers_unsorted = numbers[indeces]
                frame_mappings_unsorted = mappings[indeces]
                frame_assignments_unsorted = assignments[indeces]
                sort_indices = np.argsort(frame_mappings_unsorted)
                frame_numbers_sorted = frame_numbers_unsorted[sort_indices]
                frame_mappings_sorted = frame_mappings_unsorted[sort_indices]
                frame_assignments_sorted = frame_assignments_unsorted[sort_indices]

                index = np.where(frame_mappings_sorted==frame_mapping)[0][0]

                start_index = max(0, index - window)
                end_index = min(len(frame_numbers_sorted), index + window + 1)
            
                frame_numbers = frame_numbers_sorted[start_index:end_index]
                frame_mappings = frame_mappings_sorted[start_index:end_index]
                frame_assignments = frame_assignments_sorted[start_index:end_index]

                frame_numbers = frame_numbers.tolist()
                frame_mappings = frame_mappings.tolist()
                frame_assignments = frame_assignments.tolist()

                frame_basenames = [basename]*len(frame_numbers)
                frame_csvs = [csv_name]*len(frame_numbers)
                
                start_index = window 

            elif radio_button_value == 'embedded_space':
                nn_model = NN(n_neighbors=10, algorithm='auto')
                nn_model.fit(embeddings)

                index = np.where(mappings==frame_mapping)[0][0]

                distances, indices = nn_model.kneighbors([embeddings[index]])
                
                nearest_frames_numbers = [numbers[i] for i in indices[0]]
                nearest_frames_mappings = [mappings[i] for i in indices[0]]
                nearest_frames_assignments = [assignments[i] for i in indices[0]]
                nearest_frame_basenames = [basenames[i] for i in indices[0]]
                nearest_frame_csvs = [csvs[i] for i in indices[0]]

                frame_numbers = [int(num) for num in nearest_frames_numbers]
                frame_mappings = [int(mapping) for mapping in nearest_frames_mappings]
                frame_assignments = [int(assign) for assign in nearest_frames_assignments]
                frame_basenames = nearest_frame_basenames
                frame_csvs = nearest_frame_csvs

                start_index = 0

            for k in range(len(frame_numbers)):
                
                mp4filepath = os.path.join(folder_path,frame_basenames[k]+".mp4")
                mp4 = cv2.VideoCapture(mp4filepath)
                mp4.set(cv2.CAP_PROP_POS_FRAMES, frame_numbers[k])
                ret, frame = mp4.read()

                if ret: 
                    if keypoints:
                        csvfilepath = os.path.join(folder_path,frame_csvs[k])

                        file_j_df = pd.read_csv(csvfilepath, low_memory=False)          
                        file_j_df_array = np.array(file_j_df) 

                        keypoint_data = file_j_df_array[np.where(file_j_df_array[:,0]==str(frame_mappings[k]))][0]

                        x = keypoint_data[1::3] 
                        y = keypoint_data[2::3]

                        xy = np.concatenate([x.reshape(-1, 1), y.reshape(-1, 1)], axis=1)
                        xy = [(int(float(x)), int(float(y))) for x, y in xy]

                        for point in xy: 
                            cv2.circle(frame, point, radius=5, color=(0, 0, 255), thickness = -1)
                    
                    _, buffer = cv2.imencode('.jpg', frame)
                    frame_data = base64.b64encode(buffer).decode('utf-8')
                    frame_images.append(frame_data)
                    frames.append(frame_mappings[k])
                
            mp4.release()
            return jsonify({'frame_data': frame_images, 'frames': frames, 'assignments': frame_assignments, 'start_index': start_index, 'basenames': frame_basenames})

@app.route('/', methods = ["GET", "POST"])
@app.route('/home', methods = ["GET", "POST"])
def home():
    graphJSON = None
    folder_path = None
    file_j_df = None
    uploadform = UploadForm()
    clusterform = ClusterForm()
    fractionform = FractionForm()
    keypointform = KeypointForm()
    plotlyform = PlotlyForm()
    parameterform = ParameterForm()
    nameform = NameForm()
    loadnameform = LoadNameForm()
    form_submitted = False

    if uploadform.validate_on_submit() and uploadform.upload.data: 
        
        db.session.query(SessionData).delete()
        db.session.commit()

        folder_path = uploadform.folder.data
        keypoints = keypointform.keypoints.data
        load_plot = plotlyform.load_plot.data
    
        if load_plot: 
            name = loadnameform.loadname.data
            
            parameterform = ParameterForm(formdata=None)
            uploadform = UploadForm(formdata=None)
            nameform = NameForm(formdata=None)
    
            for filename in os.listdir(folder_path):
                if filename.endswith('.mp4'):
                    mp4filepath = os.path.join('uploads', 'videos', filename)
                    shutil.copyfile(os.path.join(folder_path,filename), mp4filepath)

                elif filename.endswith('.csv'):
                    csvfilepath = os.path.join('uploads', 'csvs', filename)
                    csvfilename = filename
                    shutil.copyfile(os.path.join(folder_path,filename), csvfilepath)
                    file_j_df = pd.read_csv(csvfilepath, low_memory=False)          
            
            for filename in os.listdir(os.path.join(folder_path, name)):
                if filename.endswith('.json'):
                    with open(os.path.join(folder_path,name,filename), 'r', encoding='utf-8') as f:
                        graphJSON = f.read()

                elif filename == 'data.csv':

                    data = pd.read_csv(os.path.join(folder_path,name,filename))
                    frame_mappings = data["mapping"]
                    frame_numbers = data["frame_number"]
                    assignments = data["assignments"]
                    basename_mappings = data["basenames"]
                    csv_mappings = data["csvs"]
                    fps = data["fps"][0]
                    UMAP_min = data["UMAP_min"][0]
                    UMAP_seed = data["UMAP_seed"][0]
                    HDBSCAN_samples = data["HDBSCAN_samples"][0]
                    HDBSCAN_min = data["HDBSCAN_min"][0]
                    HDBSCAN_max = data["HDBSCAN_max"][0]

                    parameterform.fps.data = fps
                    parameterform.umap_min_dist.data = UMAP_min
                    parameterform.umap_random_state.data = UMAP_seed
                    parameterform.hdbscan_min_samples.data = HDBSCAN_samples
                    parameterform.hdbscan_cluster_min.data = HDBSCAN_min
                    parameterform.hdbscan_cluster_max.data = HDBSCAN_max
                    uploadform.folder.data = folder_path
                    nameform.name.data = name+"_copy"
                
                elif filename == 'embedding.npy':
                    embeddings = np.load(os.path.join(folder_path,name,filename))

        else: 
            
            fps = parameterform.fps.data
            training_fraction = fractionform.slider.data
            mindist = parameterform.umap_min_dist.data
            randomstate = parameterform.umap_random_state.data
            minsamples=parameterform.hdbscan_min_samples.data
            min_cluster = parameterform.hdbscan_cluster_min.data
            max_cluster = parameterform.hdbscan_cluster_max.data
            name = nameform.name.data

            UMAP_PARAMS = {
                'min_dist': mindist,  
                'random_state': randomstate,
            }

            HDBSCAN_PARAMS = {
                'min_samples': minsamples,
            }

            cluster_range = [min_cluster, max_cluster]
            
            nameform.name.data = name+"_copy"
            
            graphJSON, frame_mappings, frame_numbers, assignments, basename_mappings, csv_mappings, embeddings  = return_plot(folder_path, fps, UMAP_PARAMS, cluster_range, HDBSCAN_PARAMS, training_fraction, name)

        session_data = SessionData(
            folder_path=folder_path,
            assignments=assignments,
            frame_numbers=frame_numbers,
            frame_mappings=frame_mappings,
            keypoints=keypoints,
            name = name, 
            basename_mappings = basename_mappings, 
            csv_mappings = csv_mappings,
            embeddings = embeddings
        )

        db.session.add(session_data)
        db.session.commit()

        session['session_data_id'] = session_data.id
        
    if clusterform.validate_on_submit() and clusterform.cluster.data:

        session_data_id = session.get('session_data_id')

        if session_data_id:
            session_data = SessionData.query.get(session_data_id)
            if session_data:
                folder_path = session_data.folder_path
                assignments = session_data.assignments
                frame_numbers = session_data.frame_numbers
                frame_mappings = session_data.frame_mappings
                keypoints = session_data.keypoints
                name = session_data.name
                basename_mappings = session_data.basename_mappings
                csv_mappings = session_data.csv_mappings
        
        save_images(folder_path, frame_mappings, frame_numbers, assignments, basename_mappings, csv_mappings, keypoints, name)
    
    return render_template('index.html', uploadform=uploadform, plotlyform=plotlyform, clusterform=clusterform, fractionform=fractionform, keypointform=keypointform, parameterform=parameterform, graphJSON = graphJSON, nameform=nameform, loadnameform=loadnameform)

if __name__ == '__main__':
    threading.Thread(target=open_browser).start()
    app.run(debug = False)