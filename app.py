import os
import pandas as pd
import uuid
import shutil
import zipfile
import tempfile

from flask import Flask, render_template, request, session, send_file
from werkzeug.utils import secure_filename
from datetime import datetime
from image_processing import process_image
from models_prediction import get_system_prediction
from rembg import new_session
from zipfile import ZipFile

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
    # get radom identifier for each request
    return str(uuid.uuid4())[:8]

#SERVER
@app.route('/')
def start():
    return render_template('upload.html')


@app.route('/upload_folder', methods=['POST'])
def upload_folder():
    # CREATES NEW DIRECTORY FOR THE REQUEST
    session['identifier'] = get_identifier()
    session['request_path'] = os.path.join(app.config['REQUESTS'], "request_{}".format(session['identifier']))
    session['request_path_processed'] = os.path.join(session['request_path'], "processed")

    os.mkdir(session['request_path'])
    os.mkdir(session['request_path_processed'])
    
    # CREATES NEW SESSION FOR REMBG
    bgremove_session = new_session()
    
    # CHECK IF FILES ARE VALID
    files = request.files.getlist("file") 
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # PROCESS IMAGES DIRECTLY FROM MEMORY, SKIP SAVING RAW FILES
            try:
                # Assume `process_image` can take a file stream directly:
                processed_img = process_image(file.stream, bgremove_session)
                processed_img_path = os.path.join(session['request_path_processed'], filename.split(".")[0]+".png")
                processed_img.save(processed_img_path)
            except Exception as e:
                return f'Error processing file: {str(e)}', 500  
        else:
            return "{} was not uploaded, operation stopped".format(file.filename)
    
    # CONTINUE AS PREVIOUSLY FOR PREDICTION AND RESPONSE HANDLING
    prediction_df = get_system_prediction(os.path.join(session['request_path_processed']))
    predictiondf_path = os.path.join(session['request_path'], "predictions_{}.csv".format(session['identifier']))
    
    prediction_df[["image_name",
                   "highest_species_prediction",
                   "highest_species_confidence",
                   "second_highest_species_prediction",
                   "second_highest_species_confidence"]].to_csv(predictiondf_path, sep=';')

    prediction_dict = prediction_df.to_dict(orient='records')
    title = str(session['identifier'])
    
    return render_template('predictions.html', predictions=prediction_dict, request=title)

@app.route('/display_pdf')
def display_pdf():
    pdf_path = os.path.join(app.config['STATIC_FOLDER'], "guide", "ConVector_MosquitoWingRemovalGuide.pdf")
    return send_file(pdf_path, as_attachment=False)

@app.route('/display_iden_pdf')
def display_iden_pdf():
    pdf_path = os.path.join(app.config['STATIC_FOLDER'], "guide", "Reverse-identification-key.pdf")
    return send_file(pdf_path, as_attachment=False)

@app.route('/download_csv', methods=['GET'])
def download_csv():
    csv_file_path = os.path.join(session['request_path'], "predictions_{}.csv".format(session['identifier']))
    return send_file(csv_file_path, as_attachment=True)

@app.route('/download_folder', methods=['GET'])
def download_folder():
    folder_path = session['request_path']
    zip_file_name = 'request_{}.zip'.format(session['identifier'])
    zip_path = os.path.join(tempfile.gettempdir(), zip_file_name)
            
    try:
        with ZipFile(zip_path, 'w') as zipObj:
            # Get the root directory name to exclude from the zip archive
            root_dir = os.path.basename(os.path.normpath(folder_path))
            
            # Iterate over all the files in directory
            for folderName, subfolders, filenames in os.walk(folder_path):
                for filename in filenames:
                    # Create complete filepath of file in directory
                    filePath = os.path.join(folderName, filename)
                    
                    # Get the relative path to the file, excluding the first two parent directories
                    arcname = os.path.relpath(filePath, folder_path).replace(root_dir + os.sep, '',  1)
                    
                    # Add file to zip with the new relative path
                    zipObj.write(filePath, arcname=arcname)
        
        # Send the zipped file
        return send_file(zip_path, as_attachment=True)
    finally:
        # Remove the temporary zip file after sending
        if os.path.exists(zip_path):
            os.remove(zip_path)

    
if __name__ == '__main__':
    app.run()
