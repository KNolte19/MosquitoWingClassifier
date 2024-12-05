import os
import uuid
import tempfile

from flask import Flask, render_template, request, session, send_file
from werkzeug.utils import secure_filename
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

# Allowed file extensions for uploads
ALLOWED_EXTENSIONS = {"png", "jpeg", "tif", "jpg", "tiff"}
NUM_AUG = 2

def allowed_file(filename):
    """Check if the filename has an allowed extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


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

    augmented_datasets = []
    for i in range(NUM_AUG):
        # Dont augment the first image
        if i == 0:
            augment_bool = False
        else:
            augment_bool = True

        # Get image dataset from the uploaded files
        dataset = ImageGenerator(file_list=files,
                                augment_bool=augment_bool,
                                processed_file_name_list=processed_file_name_list,
                                bg_session=bgremove_session)
        
        augmented_datasets.append(dataset)

    # Get system predictions
    prediction_df = get_system_prediction(session["request_path_processed"], augmented_datasets, file_name_list)

    # Save the predictions to a CSV file
    predictiondf_path = os.path.join(
        session["request_path"], "predictions_{}.csv".format(session["identifier"])
    )

    prediction_df[
        [
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


@app.route("/display_pdf")
def display_pdf():
    """Display the mosquito wing removal guide PDF."""
    pdf_path = os.path.join(
        app.config["STATIC_FOLDER"], "guide", "ConVector_MosquitoWingRemovalGuide.pdf"
    )
    return send_file(pdf_path, as_attachment=False)


@app.route("/display_iden_pdf")
def display_iden_pdf():
    """Display the reverse identification key PDF."""
    pdf_path = os.path.join(
        app.config["STATIC_FOLDER"], "guide", "Reverse-identification-key.pdf"
    )
    return send_file(pdf_path, as_attachment=False)


@app.route("/download_csv", methods=["GET"])
def download_csv():
    """Download the CSV file containing predictions."""
    csv_file_path = os.path.join(
        session["request_path"], "predictions_{}.csv".format(session["identifier"])
    )
    return send_file(csv_file_path, as_attachment=True)


@app.route("/download_folder", methods=["GET"])
def download_folder():
    """Download the entire request folder as a zip file."""
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
    app.run(debug=False, port=1919)
