from flask import Flask, render_template
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
import json
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secretkey'
app.config['TEMPLATES_AUTO_RELOAD'] = True

def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000/')

class UploadFileForm(FlaskForm):
    file = FileField("File")
    submit = SubmitField("Generate UMAP Embedding")

@app.route('/', methods = ["GET","POST"])
@app.route('/home', methods = ["GET","POST"])
def home():
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data # First grab the file
        #file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename(file.filename))) # Then save the file
        return "File has been uploaded."
    return render_template('index.html', form=form)


# @app.route('/')
# def index():
#     feature = 'Bar'
#     bar = create_plot(feature)
#     return render_template('index.html', plot=bar)

# def create_plot(feature):
#     if feature == 'Bar':
#         N = 40
#         x = np.linspace(0, 1, N)
#         y = np.random.randn(N)
#         df = pd.DataFrame({'x': x, 'y': y}) # creating a sample dataframe
#         data = [
#             go.Bar(
#                 x=df['x'], # assign x as the dataframe column 'x'
#                 y=df['y']
#             )
#         ]
#     else:
#         N = 1000
#         random_x = np.random.randn(N)
#         random_y = np.random.randn(N)

#         # Create a trace
#         data = [go.Scatter(
#             x = random_x,
#             y = random_y,
#             mode = 'markers'
#         )]


#     graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)

#     return graphJSON


# @app.route('/bar', methods=['GET', 'POST'])
# def change_features():

#     feature = request.args['selected']
#     graphJSON= create_plot(feature)

#     return graphJSON


if __name__ == '__main__':
    threading.Thread(target=open_browser).start()
    app.run(debug = True)