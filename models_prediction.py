import tensorflow as tf 
import numpy as np 
import pandas as pd
import os
from joblib import dump, load

# SET PARAMETERS
IMG_SIZE = 300
SPECIES_NAMES = ['modestus', 'cinereus', 'communis-punctor', 'rusticus',
       'sticticus', 'vexans', 'claviger', 'other', 'richiardii',
       'aegypti', 'albopictus', 'japonicus', 'koreicus', 'maculipennis',
       'pipiens-torrentium', 'stephensi', 'morsitans', 'cantans',
       'caspius', 'cataphylla', 'vishnui-group']

# LOAD MODEL
cnn_model = tf.keras.models.load_model("static/models/TUNING_9_2024-06-13-10-07", compile=False)

def getImage(file_path):
    # Load the raw data from the file as a string
    image = tf.io.read_file(file_path)
    # Convert the compressed string to a 3D uint8 tensor
    image = tf.io.decode_png(image, channels=3)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    return image

def get_cnn_prediction(dataset):
    # Let model predict
    prediction_list = cnn_model.predict(dataset, verbose=0)
    
    def parse_prediction(prediction_list, rank):
        highest_score_list = [np.sort(prediction)[-rank] for prediction in prediction_list]
        highest_score_index_list = [np.where(prediction_list[i] == highest_score)[0][0] for i, highest_score in enumerate(highest_score_list)]
        species_name_list = [SPECIES_NAMES[highest_score_index] for highest_score_index in highest_score_index_list]
        
        return np.asarray(highest_score_list), np.asarray(species_name_list)
    
    highest_score_list, species_name_list = parse_prediction(prediction_list, 1)    
    sec_highest_score_list, sec_species_name_list = parse_prediction(prediction_list, 2)
    
    return highest_score_list, species_name_list, sec_highest_score_list, sec_species_name_list
    
def get_system_prediction(folder_path):
    # Load Images
    folder_path = folder_path + "/*.png"
    file_list = tf.data.Dataset.list_files(folder_path, shuffle=False)
    dataset = file_list.map(getImage).batch(1)

    # Get CNN Predictions
    highest_score_list, species_name_list, sec_highest_score_list, sec_species_name_list = get_cnn_prediction(dataset)
    
    # Write Dataframe
    df = pd.DataFrame({"image_path": [f.numpy().decode('utf-8') for f in file_list],
                       "highest_species_prediction": species_name_list,
                       "highest_species_confidence": highest_score_list,
                       "second_highest_species_prediction": sec_species_name_list,
                       "second_highest_species_confidence": sec_highest_score_list})

    df["image_name"] = [name.split(os.sep)[-1] for name in df["image_path"]]
    
    df = df.astype({'image_path': 'object',
                    "image_name": "object",
                    "highest_species_prediction": 'object',
                    "highest_species_confidence": 'float64',
                    "second_highest_species_prediction": 'object',
                    "second_highest_species_confidence": 'float64'})
    
    df = df.round({"highest_species_confidence": 2, "second_highest_species_confidence": 2})

    return df