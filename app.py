import os
import pandas as pd
import hashlib
import shutil
import zipfile

from flask import Flask, render_template, request, send_file, session
from werkzeug.utils import secure_filename
from datetime import datetime
from image_processing import process_image
from models_prediction import get_system_prediction
from rembg import new_session

#CONFIGURATIONS
app = Flask(__name__)
app.config['STATIC_FOLDER'] = 'static'
app.config['REQUESTS'] = 'static/requests'

#HELPER FUNCTIONS
ALLOWED_EXTENSIONS = {'png', 'jpeg', "tif", "jpg"}
app.secret_key = 'Apfelkuchen' 

def allowed_file(filename):
    #checks if filename is allowed 
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_identifier():
    #encodes current time as unique code
    def encode_string(input_string):
        #todo: uuid
       encoded = hashlib.md5(input_string.encode()).hexdigest()
       return encoded[:5]
    
    return encode_string(datetime.now().strftime("%d%m%y%H%M%S"))

#SERVER
@app.route('/')
def start():
    return render_template('upload.html')


@app.route('/upload_folder', methods=['POST'])
def upload_folder():
    # CREATES NEW DIRECTORY FOR REQUEST 
    session['identifier'] = get_identifier()
    session['request_path'] = os.path.join(app.config['REQUESTS'], "request_{}".format(session['identifier']))
    session['request_path_raw'] = os.path.join(session['request_path'], "raw")
    session['request_path_processed'] = os.path.join(session['request_path'], "processed")

    os.mkdir(session['request_path'])
    os.mkdir(session['request_path_raw'])
    os.mkdir(session['request_path_processed'])
    
    # CREATES NEW SESSION FOR REMBG
    bgremove_session = new_session()
    
    # CHECK IF FILES ARE VALID
    files = request.files.getlist("file") 
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            try:  
                # files are saved as we want to store both the raw and processed version
                file.save(os.path.join(session['request_path_raw'], filename))
            except Exception as e:
                return f'Error saving file: {str(e)}', 500  
        else:
            return "{} was not uploaded, operation stopped".format(file.filename)

    for file in files:
        filename = secure_filename(file.filename)
        # PROCESS IMAGES BY SECONDARY SCRIPT
        image_path = os.path.join(session['request_path_raw'], filename)
        processed_img = process_image(image_path, bgremove_session)

        # SAVE FILE AS PNG TO FOLDER
        processed_img_path = os.path.join(session['request_path_processed'], filename.split(".")[0]+".png")
        processed_img.save(processed_img_path)
        
    # CNN PREDICT SPECIES
    prediction_df = get_system_prediction(os.path.join(session['request_path_processed']))
    predictiondf_path = os.path.join(session['request_path'], "predictions_{}.csv".format(session['identifier']))
    prediction_df.to_csv(predictiondf_path, sep=';')
        
    return render_template('predictions.html', predictions=prediction_df.to_dict(orient='records'))

@app.route('/download_csv', methods=['GET'])
def download_csv():
    csv_file_path = os.path.join(session['request_path'], "predictions_{}.csv".format(session['identifier']))
    return send_file(csv_file_path, as_attachment=True)

@app.route('/download_folder', methods=['GET'])
def download_folder():
    zip_filename = 'request_{}.zip'.format(session['identifier'])

    # Create a ZipFile Object in write mode
    with zipfile.ZipFile(zip_filename, 'w') as zipObj:
        # List files in the directory
        for filename in os.listdir(session['request_path']):
            # Create complete filepath of file in directory
            filePath = os.path.join(session['request_path'], filename)
            # Add file to zip without any folder structure
            zipObj.write(filePath, arcname=filename)

    # Send the zip file
    return send_file(zip_filename, as_attachment=True)
    
if __name__ == '__main__':
    app.run()
