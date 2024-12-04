import skimage as ski
import numpy as np
import pandas as pd
import os
import torch

# Set image size and species names
SPECIES_NAMES = [
    'Ae. aegypti',
    'Ae. albopictus',
    'Ae. annulipes-group',
    'Ae. caspius',
    'Ae. cataphylla',
    'Ae. cinereus-geminus pair',
    'An. claviger-petragani group s.l.',
    'Ae. communis-punctor pair',
    'Ae. japonicus',
    'Ae. koreicus',
    'An. maculipennis s.l.',
    'Cx. modestus',
    'Cs. morsitans-fumipennis pair',
    'other',
    'Cx. torrentium-pipiens s.l. pair',
    'Cq. richiardii',
    'Ae. rusticus',
    'An. stephensi',
    'Ae. sticticus',
    'Ae. vexans',
    'Cx. vishnui-group',
]

# Load pre-trained model
cnn_model = torch.load(os.path.join("static", "models", "model_1_flowing-music-18.pt"), map_location=torch.device('cpu'))

def logistic_function(x):
    """
    Logistic function for converting scores to probabilities.

    Args:
        x (float): Input score.

    Returns:
        float: Probability value.
    """
    coefficients = 2.52
    intercept = -7.85
    z = np.dot(x, coefficients) + intercept
    return 1 / (1 + np.exp(-z))

def prediction_loop(dataloader):
    predictions = []
    with torch.no_grad():
        for batch, (X) in enumerate(dataloader):
                
                # Compute prediction
                pred = cnn_model(torch.tensor(X[0]).unsqueeze(0))

                predictions.append(pred.cpu().detach().numpy())

    return np.concatenate(predictions)

def get_cnn_prediction(file_list):
    """
    Get CNN model predictions for the given dataset.

    Args:
        dataset (tf.data.Dataset): Dataset containing images.

    Returns:
        tuple: Predictions for highest and second highest scores and species names.
    """

    # Generate pytorch dataset
    dataset = torch.utils.data.TensorDataset(torch.stack(file_list)) 
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, shuffle=False) #, num_workers=8

    # get prediction from model
    prediction_list = prediction_loop(dataloader)

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

#ef get_system_prediction(folder_path):
def get_system_prediction(folder_path, file_list, file_name_list):
    """
    Get system predictions for all images in the specified folder.

    Args:
        folder_path (str): Path to the folder containing images.

    Returns:
        DataFrame: DataFrame containing predictions and confidence scores.
    """

    file_name_list = [os.path.join(folder_path, x) for x in file_name_list]
    highest_scores, highest_species, second_highest_scores, second_highest_species = get_cnn_prediction(file_list)

    df = pd.DataFrame(
        {
            "image_path": file_name_list,
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
