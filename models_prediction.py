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
cnn_model = torch.load(os.path.join("static", "models", "model_2_deft-bird-19.pt"), map_location=torch.device('cpu'), weights_only=False)

def logistic_function(x):
    coefficients = 2.52
    intercept = -7.85
    z = np.dot(x, coefficients) + intercept
    return 1 / (1 + np.exp(-z))

def prediction_loop(dataloader):
    predictions = []
    with torch.no_grad():
        for batch, (X) in enumerate(dataloader):
                # Check if image is wrongly processed (based on the proportion of non-zero pixels)
                if torch.sum((X > 0)/ 73344) < 0.275:
                    pred = torch.tensor([[-99] * len(SPECIES_NAMES)])
                else:
                    pred = cnn_model(X.unsqueeze(0))

                predictions.append(pred.cpu().clone().detach().numpy())

    return np.concatenate(predictions)

def get_cnn_prediction(datasets):
    prediction_list = []
    # Generate pytorch dataloader
    for dataset in datasets:
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)
        prediction_list.extend(prediction_loop(dataloader))
    
    return prediction_list

def parse_prediction(predictions, rank):
    
    highest_scores = [np.sort(prediction)[-rank] for prediction in predictions]
    highest_indices = [
        np.where(prediction == score)[0][0]
        for prediction, score in zip(predictions, highest_scores)
        ]

    species_names = np.asarray([SPECIES_NAMES[idx] for idx in highest_indices])
    calibrated_highest_scores = np.asarray(logistic_function(highest_scores))

    return list(calibrated_highest_scores), list(species_names)

#ef get_system_prediction(folder_path):
def get_system_prediction(folder_path, datasets, file_name_list):
    file_name_list = [os.path.join(folder_path, x) for x in file_name_list]

    # Get predictions for every augmentation run
    avg_prediction_list = []
    for dataset in datasets:
        prediction_list = get_cnn_prediction(dataset)
        avg_prediction= np.mean(prediction_list, axis=0)
        avg_prediction_list.append(avg_prediction)

    # Get highest and second highest predictions
    highest_scores, highest_species = parse_prediction(avg_prediction_list, 1)
    second_highest_scores, second_highest_species = parse_prediction(avg_prediction_list, 2)

    # Check for low confidence predictions
    for i, score in enumerate(highest_scores):
        # Do not return prediction if score is too low
        if (float(score) < 0.00001):
            highest_species[i] = "Processing Error Detected"
        if (float(score) < 0.5):
            highest_species[i] = "Low Confidence Prediction"
        # Do not return prediction if score is close to zero as it indicates a processing error

    for i, score in enumerate(second_highest_scores):
        # Do not return prediction if score is close to zero as it indicates a processing error
        if (float(score) < 0.00001):
            second_highest_species[i] = "Processing Error Detected"
        # Do not return prediction if score is too low
        if (float(score) < 0.1) or highest_species[i] == "Low Confidence Prediction":
            second_highest_species[i] = "Low Confidence Prediction"

    pd.set_option('display.max_colwidth', None) 
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
