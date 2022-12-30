from mailbox import Message
import os
from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename
import subprocess
import json
import shutil
import tempfile
from flask import jsonify, send_file
from flask_cors import CORS

import sys
import json
import os
import argparse
import pandas as pd
import shutil
from blast_ct.trainer.inference import ModelInference, ModelInferenceEnsemble
from blast_ct.train import set_device
from blast_ct.read_config import get_model, get_test_loader
from blast_ct.nifti.savers import NiftiPatchSaver

app=Flask(__name__)
CORS(app)
device = "0"
job_dir = '/tmp/blast_ct'
os.makedirs(job_dir, exist_ok=True)
install_dir = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(install_dir, 'blast_ct/data/config.json'), 'r') as f:
        config = json.load(f)
model = get_model(config)
device = set_device("0")
app.secret_key = "secret key"
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 * 1024 * 1024

# Get current path
path = os.getcwd()
# file Upload
UPLOAD_FOLDER = "/home/input"

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = set(['dcm', 'nii.gz', 'dicom', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/ich/test', methods=['GET'])
def upload_form():
    return render_template('upload.html')

@app.route('/ich/predict', methods=['POST'])
def upload():
    print("inside ---")
    # os.mkdir(app.config['UPLOAD_FOLDER'])
    if request.method == 'POST':

        
        print("inside ---")
        files = request.files.getlist('files[]')
        inputDir = tempfile.mkdtemp(dir="input")
        os.environ['inputDir'] = inputDir
        outDir = tempfile.mkdtemp(dir="output")
        os.environ['outDir'] = outDir
        inputFile = ""
        for file in files:
            filename = secure_filename(file.filename)
            print(filename)
            inputFile = inputDir +"/" +filename
            # file.save(inputDir +"/" +filename)
            file.save(inputFile)
            
    
        print(app.config['UPLOAD_FOLDER'])
        outputfile = outDir +"/infile.nii.gz"
        test_csv_path = os.path.join(job_dir, 'test.csv')
        pd.DataFrame(data=[['im_0', inputFile]], columns=['id', 'image']).to_csv(test_csv_path, index=False)
        test_loader = get_test_loader(config, model, test_csv_path, use_cuda=not device.type == 'cpu')
        saver = NiftiPatchSaver(job_dir, test_loader, write_prob_maps=False)
        model_paths = [os.path.join(install_dir, f'blast_ct/data/saved_models/model_{i:d}.pt') for i in range(1, 13)]
        ModelInferenceEnsemble(job_dir, device, model, saver, model_paths, task='segmentation')(None)
        output_dataframe = pd.read_csv(os.path.join(job_dir, 'predictions/prediction.csv'))

        shutil.copyfile(output_dataframe.loc[0, 'prediction'], outputfile)
        shutil.rmtree(job_dir)
        return send_file(outputfile, mimetype="application/zip, application/octet-stream, application/x-zip-compressed, multipart/x-zip")
        # return send_file("/home/output/infile.nii.gz", mimetype="application/zip, application/octet-stream, application/x-zip-compressed, multipart/x-zip")

 
if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5000,debug=False,threaded=True)
