# LandRoval - Mosquito Wing Species Classification

## Overview
LandRoval is a web-based app designed to assist in identifying mosquito wing species through image analysis. Users can upload images of mosquito wings and receive classifications for the species present. The service is built using Flask and Gunicorn, providing a user-friendly interface accessible via a web browser.

Please note that this version is intended for testing and validation purposes. Although the classification model is reliable, there may be occasional misclassifications. For decisions requiring high accuracy, it is recommended to consult with  experts or utilize additional resources. Each prediction includes a confidence level metric to gauge the reliability of the classification. Upon submission, you will receive an identifier for accessing the prediction results or processed images.

## Getting Started
To deploy LandRoval on your local machine, you need to have Python and Docker installed. Follow the steps below to set up the application:

1. **Clone the Repository:**
bash git clone https://github.com/KNolte19/MosquitoWingClassifier.git

2. **Build the Docker Image:**
bash docker build -t app .

4. **Access the Application:**
   Open your web browser and navigate to `http://localhost:8000`.

## Features
- **Image Upload:** Users can upload images of mosquito wings directly through the web interface.
- **Species Classification:** The system processes uploaded images and returns the identified species.
- **Confidence Metrics:** Each result includes a confidence score indicating the model's certainty in its classification.
- **Download Results:** Users can download the classification report or processed images using the provided identifier.

---

**Maintained by:** Kristopher Nolte

**Support:** Contact us at kristopher.nolte@bnitm.de for any inquiries or assistance.

**Version:** v0.0.2

**Last Updated:** February  9,  2024
