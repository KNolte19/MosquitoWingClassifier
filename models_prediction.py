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
                image = torch.tensor(X).unsqueeze(0)
                print(image.shape)
                # Check if image is wrongly processed (based on the proportion of non-zero pixels)
                if torch.sum((image > 0)/ 73344) < 0.275:
                    pred = torch.tensor([[-99] * len(SPECIES_NAMES)])
                else:
                    pred = cnn_model(image)

                predictions.append(pred.cpu().detach().numpy())

    return np.concatenate(predictions)

def get_cnn_prediction(datasets):
    """
    Get CNN model predictions for the given dataset.

    Args:
        dataset (tf.data.Dataset): Dataset containing images.

    Returns:
        tuple: Predictions for highest and second highest scores and species names.
    """

    # Generate pytorch dataloader
    dataloader = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset(datasets), batch_size=1, shuffle=False)

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

        species_names = np.asarray([SPECIES_NAMES[idx] for idx in highest_indices])
        calibrated_highest_scores = np.asarray(logistic_function(highest_scores))

        return calibrated_highest_scores, species_names

    highest_scores, highest_species = parse_prediction(prediction_list, rank=1)
    second_highest_scores, second_highest_species = parse_prediction(prediction_list, rank=2)

    return (
        highest_scores,
        highest_species,
        second_highest_scores,
        second_highest_species,
    )

#ef get_system_prediction(folder_path):
def get_system_prediction(folder_path, datasets, file_name_list):
    """
    Get system predictions for all images in the specified folder.

    Args:
        folder_path (str): Path to the folder containing images.

    Returns:
        DataFrame: DataFrame containing predictions and confidence scores.
    """

    file_name_list = [os.path.join(folder_path, x) for x in file_name_list]

    # Initialize lists to store all predictions
    all_highest_scores = []
    all_highest_species = []
    all_second_highest_scores = []
    all_second_highest_species = []

    # Get predictions for every augmentation run
    for dataset in datasets:
        highest_scores, highest_species, second_highest_scores, second_highest_species = get_cnn_prediction(dataset)
        all_highest_scores.append(highest_scores)
        all_highest_species.append(highest_species)
        all_second_highest_scores.append(second_highest_scores)
        all_second_highest_species.append(second_highest_species)

    # Average the results
    highest_scores = np.mean(all_highest_scores, axis=0)
    second_highest_scores = np.mean(all_second_highest_scores, axis=0)

    # Use the most frequent species prediction
    highest_species = [max(set(species), key=species.count) for species in zip(*all_highest_species)]
    second_highest_species = [max(set(species), key=species.count) for species in zip(*all_second_highest_species)]

    # Check for low confidence predictions
    for i, score in enumerate(highest_scores):
        # Do not return prediction if score is too low
        if (float(score) < 0.5):
            highest_species[i] = "Low Confidence Prediction"
        # Do not return prediction if score is close to zero as it indicates a processing error
        if (float(score) < 0.00001):
            highest_species[i] = "Processing Error Detected"
    for i, score in enumerate(second_highest_scores):
        # Do not return prediction if score is too low
        if (float(score) < 0.1) or highest_species[i] == "Low Confidence Prediction":
            second_highest_species[i] = "Low Confidence Prediction"
        # Do not return prediction if score is close to zero as it indicates a processing error
        if (float(score) < 0.00001):
            highest_species[i] = "Processing Error Detected"

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
