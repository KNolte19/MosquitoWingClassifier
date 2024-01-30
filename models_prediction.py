import tensorflow as tf 
import numpy as np 
import pandas as pd
import os
from joblib import dump, load

#LOAD MODEL
IMG_SIZE = 300
CNN_MODEL_PATH = "models/cenedril_wing_model.h5"
SPECIES_NAMES = ['Aedes aegypti', 'Aedes albopictus', 'Aedes cantans', 'Aedes cataphylla',
                 'Aedes cinereus', 'Aedes communis','Aedes japonicus', 'Aedes koreicus',
                 'Anopheles maculipennis','Culex pipiens / torrentium', 'Aedes punctor',
                 'Coquillettidia richiardii','Aedes rusticus', 'Aedes sticticus', 'Aedes vexans']

#ToDo initiliaze model that it returns feature map
cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)
# Create a new model with the same input but different output
cnn_model = tf.keras.Model(inputs=cnn_model.input,
                           outputs=[cnn_model.output, cnn_model.get_layer('global_average_pooling2d').output])

calibration_model = load("models/calibrator_model.joblib")
novelty_model = load("models/novelty_model.joblib")

def getImage(file_path):
    # Load the raw data from the file as a string
    image = tf.io.read_file(file_path)
    # Convert the compressed string to a 3D uint8 tensor
    image = tf.io.decode_png(image, channels=3)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    return image

def get_cnn_prediction(dataset):
    # Let model predict
    prediction_list, feature_map_list = cnn_model.predict(dataset, verbose=0)

    def parse_prediction(prediction_list, rank):
        highest_score_list = [np.sort(prediction)[-rank] for prediction in prediction_list]
        highest_score_index_list = [np.where(prediction_list[i] == highest_score)[0][0] for i, highest_score in enumerate(highest_score_list)]
        species_name_list = [SPECIES_NAMES[highest_score_index] for highest_score_index in highest_score_index_list]
        
        return np.asarray(highest_score_list), np.asarray(species_name_list)
    
    highest_score_list, species_name_list = parse_prediction(prediction_list, 1)    
    sec_highest_score_list, sec_species_name_list = parse_prediction(prediction_list, 2)
    
    return highest_score_list, species_name_list, sec_highest_score_list, sec_species_name_list, feature_map_list

def get_calibrated_prediction(score_list):
    # Calibrate cnn outputs
    probability_prediction = calibration_model.predict_proba(score_list.reshape(-1,1))
    return ["%.3f" % probability[1] for probability in probability_prediction]

def get_novelty_prediction(feature_map_list):
    # Calibrate c outputs
    novelty_prediction_list = novelty_model.predict_proba(feature_map_list)
    return ["%.3f" % novelty[1] for novelty in novelty_prediction_list]
    
def get_system_prediction(folder_path):
    # Load Images
    folder_path = folder_path + "/*.png"
    file_list = tf.data.Dataset.list_files(folder_path, shuffle=False)
    dataset = file_list.map(getImage).batch(1)

    # Get Predictions
    highest_score_list, species_name_list, sec_highest_score_list, sec_species_name_list, feature_map_list = get_cnn_prediction(dataset)
    calibrated_highest_score_list = get_calibrated_prediction(highest_score_list)
    calibrated_sec_highest_score_list = get_calibrated_prediction(sec_highest_score_list)

    novelty_score_list = get_novelty_prediction(feature_map_list)

    # Write Dataframe
    df = pd.DataFrame({"image_path": [f.numpy().decode('utf-8') for f in file_list],
                       "knownclass_confidence": novelty_score_list,
                       "highest_species_prediction": species_name_list,
                       "highest_species_confidence": calibrated_highest_score_list,
                       "second_highest_species_prediction": sec_species_name_list,
                       "second_highest_species_confidence": calibrated_sec_highest_score_list})

    df["image_name"] = [name.split(os.sep)[-1] for name in df["image_path"]]
    
    df = df.astype({'image_path': 'object',
                    "image_name": "object",
                    'knownclass_confidence': 'float64',
                    "highest_species_prediction": 'object',
                    "highest_species_confidence": 'float64',
                    "second_highest_species_prediction": 'object',
                    "second_highest_species_confidence": 'float64'})


    return df