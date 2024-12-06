# BALROG - Mosquito Wing Species Classification

## Overview
BALROG is a web-based app designed to assist in identifying mosquito wing species through image analysis. Users can upload images of mosquito wings and receive classifications for the species present. The service is built using Flask and Gunicorn, providing a user-friendly interface accessible via a web browser.

Please note that this version is intended for testing and validation purposes. Although the classification model is reliable, there may be occasional misclassifications. For decisions requiring high accuracy, it is recommended to consult with  experts or utilize additional resources. Each prediction includes a confidence level metric to gauge the reliability of the classification. Upon submission, you will receive an identifier for accessing the prediction results or processed images.

## Getting Started
To deploy BALROG on your local machine, you need to have Python and Docker installed. Follow the steps below to set up the application:

1. **Clone the Repository:**\
`git clone https://github.com/KNolte19/MosquitoWingClassifier.git`

2. **Download remBG weights to avoid unnecessary download**\
BALROG utilises the RemBG library to remove the background, the model is relatively large and to avoid downloading it every time the application is restarted we advice downloading the model weights beforehand.\ 
`https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx`

3. **Build the Docker Image:**\
`docker build -t app .`

4. **Run Docker** \
You have to change the path to the copy of this repo on your machine\
`docker run -p 5050:5050 -v /path/to/repo/app/static/requests:/app/static/requests app`\
For instance\
`docker run -p 5050:5050 -v /HansPeter/Documents/app/static/requests:/app/static/requests app`

5. **Access the Application:**\
Open your web browser and navigate to `http://localhost:5050`.


## Features
- **Image Upload:** Users can upload images of mosquito wings directly through the web interface.
- **Species Classification:** The system processes uploaded images and returns the identified species.
- **Confidence Metrics:** Each result includes a confidence score indicating the model's certainty in its classification.
- **Download Results:** Users can download the classification report or processed images using the provided identifier.

---

**Maintained by:** Kristopher Nolte

**Support:** Contact us at kristopher.nolte@bnitm.de for any inquiries or assistance.

**Version:** v0.2

**Last Updated:** December  6,  2024
