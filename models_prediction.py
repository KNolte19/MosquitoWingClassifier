import tensorflow as tf
import numpy as np
import pandas as pd
import os

# Set image size and species names
IMG_SIZE = 300
SPECIES_NAMES = [
    "Cx. modestus",
    "Ae. cinereus-geminus",
    "Ae. communis-punctor",
    "Ae. rusticus",
    "Ae. sticticus",
    "Ae. vexans",
    "An. claviger",
    "other",
    "Cq. richiardii",
    "Ae. aegypti",
    "Ae. albopictus",
    "Ae. japonicus",
    "Ae. koreicus",
    "An. maculipennis",
    "Cx. pipiens-torrentium",
    "An. stephensi",
    "Cs. morsitans-fumipennis",
    "Ae. annulipes-group",
    "Ae. caspius",
    "Ae. cataphylla",
    "Cx. vishnui-group",
]

# Load pre-trained model
cnn_model = tf.keras.models.load_model("static/models/cnn_appmodel.h5", compile=False)


def logistic_function(x):
    """
    Logistic function for converting scores to probabilities.

    Args:
        x (float): Input score.

    Returns:
        float: Probability value.
    """
    coefficients = 2.278
    intercept = -7.722
    z = np.dot(x, coefficients) + intercept
    return 1 / (1 + np.exp(-z))


def load_image(file_path):
    """
    Load and preprocess an image from the given file path.

    Args:
        file_path (str): Path to the image file.

    Returns:
        tensor: Preprocessed image tensor.
    """
    image = tf.io.read_file(file_path)
    image = tf.io.decode_png(image, channels=3)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    return image


def get_cnn_prediction(dataset):
    """
    Get CNN model predictions for the given dataset.

    Args:
        dataset (tf.data.Dataset): Dataset containing images.

    Returns:
        tuple: Predictions for highest and second highest scores and species names.
    """
    prediction_list = cnn_model.predict(dataset, verbose=0)

    def parse_prediction(predictions, rank):
        """
        Parse the prediction list to get species names and scores.

        Args:
            predictions (list): List of predictions.
            rank (int): Rank of the prediction to parse (1 for highest, 2 for second highest).

        Returns:
            tuple: Arrays of scores and species names.
        """
        highest_scores = [np.sort(prediction)[-rank] for prediction in predictions]
        highest_indices = [
            np.where(prediction == score)[0][0]
            for prediction, score in zip(predictions, highest_scores)
        ]
        species_names = [SPECIES_NAMES[idx] for idx in highest_indices]

        return np.asarray(logistic_function(highest_scores)), np.asarray(species_names)

    highest_scores, highest_species = parse_prediction(prediction_list, 1)
    second_highest_scores, second_highest_species = parse_prediction(prediction_list, 2)

    return (
        highest_scores,
        highest_species,
        second_highest_scores,
        second_highest_species,
    )


def get_system_prediction(folder_path):
    """
    Get system predictions for all images in the specified folder.

    Args:
        folder_path (str): Path to the folder containing images.

    Returns:
        DataFrame: DataFrame containing predictions and confidence scores.
    """
    folder_path = os.path.join(folder_path, "*.png")
    file_list = tf.data.Dataset.list_files(folder_path, shuffle=False)
    dataset = file_list.map(load_image).batch(1)

    highest_scores, highest_species, second_highest_scores, second_highest_species = (
        get_cnn_prediction(dataset)
    )

    df = pd.DataFrame(
        {
            "image_path": [f.numpy().decode("utf-8") for f in file_list],
            "highest_species_prediction": highest_species,
            "highest_species_confidence": highest_scores,
            "second_highest_species_prediction": second_highest_species,
            "second_highest_species_confidence": second_highest_scores,
        }
    )

    df["image_name"] = df["image_path"].apply(lambda x: os.path.basename(x))

    df = df.astype(
        {
            "image_path": "object",
            "image_name": "object",
            "highest_species_prediction": "object",
            "highest_species_confidence": "float64",
            "second_highest_species_prediction": "object",
            "second_highest_species_confidence": "float64",
        }
    )

    df = df.round(
        {
            "highest_species_confidence": 2,
            "second_highest_species_confidence": 2,
        }
    )

    return df
