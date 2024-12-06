import os
import uuid
import tempfile
import pandas as pd

from flask import Flask, render_template, request, session, send_file
from image_processing import ImageGenerator
from models_prediction import get_system_prediction
from rembg import new_session
from zipfile import ZipFile

# Create a new session for REMBG
bgremove_session = new_session()

# Configurations
app = Flask(__name__)
app.config["STATIC_FOLDER"] = "static"
app.config["REQUESTS"] = "static/requests"
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "Apfelkuchen")

N_augmentations = 4

def get_identifier():
    """Generate a random identifier for each request."""
    return str(uuid.uuid4())[:8]

@app.route("/")
def start():
    """Render the upload page."""
    return render_template("upload.html")


@app.route("/upload_folder", methods=["POST"])
def upload_folder():
    """Handle the folder upload, process images, and save results."""

    # Create a new directory for the request
    session["identifier"] = get_identifier()
    session["request_path"] = os.path.join(
        app.config["REQUESTS"], "request_{}".format(session["identifier"])
    )
    session["request_path_processed"] = os.path.join(
        session["request_path"], "processed"
    )

    os.mkdir(session["request_path"])
    os.mkdir(session["request_path_processed"])

    # Check and process each file
    files = request.files.getlist("file")
    file_name_list = [file.filename.split(".")[0]+".png" for file in files]
    processed_file_name_list = [os.path.join(session["request_path_processed"], filename.split(".")[0] + ".png") for filename in file_name_list]

    # Get image dataset from the uploaded files
    augmented_datasets = ImageGenerator(file_list=files,
                                N_augmentations=N_augmentations,
                                processed_file_name_list=processed_file_name_list,
                                bg_session=bgremove_session)

    # Get system predictions
    prediction_df = get_system_prediction(session["request_path_processed"], augmented_datasets, file_name_list)
    
    # Save the predictions to a CSV file
    predictiondf_path = os.path.join(
        session["request_path"], "predictions_{}.csv".format(session["identifier"])
    )

    prediction_df[
        [   "image_path",
            "image_name",
            "highest_species_prediction",
            "highest_species_confidence",
            "second_highest_species_prediction",
            "second_highest_species_confidence",
        ]
    ].to_csv(predictiondf_path, sep=";")

    prediction_dict = prediction_df.to_dict(orient="records")
    title = str(session["identifier"])
    
    return render_template(
        "predictions.html", predictions=prediction_dict, request=title
    )

@app.route("/get_example")
def get_example():
   """Display some example predictions."""

   session["identifier"] = "example"

   example_path = os.path.join(
        app.config["STATIC_FOLDER"], "example", "predictions_example.csv"
    )
   
   prediction_dict = pd.read_csv(example_path, sep=";").to_dict(orient="records")
   
   return render_template(
        "predictions.html", predictions=prediction_dict, request="Example"
    )

@app.route("/display_pdf")
def display_pdf():
    """Display the mosquito wing removal guide PDF."""
    pdf_path = os.path.join(
        app.config["STATIC_FOLDER"], "guide", "ConVector_MosquitoWingRemovalGuide.pdf"
    )
    return send_file(pdf_path, as_attachment=False)


@app.route("/display_other_pdf")
def display_other_pdf():
    """Display the other labels as PDF."""
    pdf_path = os.path.join(
        app.config["STATIC_FOLDER"], "guide", "Other_Species.pdf"
    )
    return send_file(pdf_path, as_attachment=False)


@app.route("/download_csv", methods=["GET"])
def download_csv():
    """Download the CSV file containing predictions."""
    if session["identifier"] == "example":
        csv_file_path = os.path.join(
            app.config["STATIC_FOLDER"], "example", "predictions_example.csv"
        )
    else:
        csv_file_path = os.path.join(
            session["request_path"], "predictions_{}.csv".format(session["identifier"])
        )
    return send_file(csv_file_path, as_attachment=True)


@app.route("/download_folder", methods=["GET"])
def download_folder():
    """Download the entire request folder as a zip file."""
    if session["identifier"] == "example":
        folder_path = os.path.join(app.config["STATIC_FOLDER"], "example")
    else:
        folder_path = session["request_path"]

    zip_file_name = "request_{}.zip".format(session["identifier"])
    zip_path = os.path.join(tempfile.gettempdir(), zip_file_name)

    try:
        with ZipFile(zip_path, "w") as zipObj:
            # Get the root directory name to exclude from the zip archive
            root_dir = os.path.basename(os.path.normpath(folder_path))

            # Iterate over all the files in directory
            for folderName, subfolders, filenames in os.walk(folder_path):
                for filename in filenames:
                    # Create complete filepath of file in directory
                    filePath = os.path.join(folderName, filename)

                    # Get the relative path to the file, excluding the first two parent directories
                    arcname = os.path.relpath(filePath, folder_path).replace(
                        root_dir + os.sep, "", 1
                    )

                    # Add file to zip with the new relative path
                    zipObj.write(filePath, arcname=arcname)

        # Send the zipped file
        return send_file(zip_path, as_attachment=True)
    finally:
        # Remove the temporary zip file after sending
        if os.path.exists(zip_path):
            os.remove(zip_path)


if __name__ == "__main__":
    app.run(debug=False)