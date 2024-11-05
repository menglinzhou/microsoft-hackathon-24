# evaluate_helpers.py

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
# import helper functions from data_helpers.py
from processed_data_module import ProcessedData
from predicted_results_module import PredictionResults

__all__ = [ 'get_predicted_labels', 'evaluate_model']


def get_predicted_labels(predictions, threshold=0.5):
    """Process the model predictions to get the final labels given a threshold

    Args:
        predictions (float): PredictionResults instance
        threshold (float, optional): the lowest probability of predicting an entry as machine-generated texts. Defaults to 0.5.

    Returns:
        _type_: _description_
    """
    # Check if the input is an instance of PredictionResults
    if not isinstance(predictions, PredictionResults):
        raise TypeError("Input must be a `PredictionResults` instance.")
    # Extract predictions from the PredictionResults object
    predictions = predictions.predictions

    # Convert the list of predictions to a numpy array
    predictions = np.array(predictions)
    # Apply thresholding to get the predicted labels
    predicted_labels = (predictions[:, 1] > threshold).astype(int)
    return predicted_labels


def evaluate_model(predicted_label, data):
    """Compute metrics using the true label and the predicted label"""
    true_labels = data['labels']

    accuracy = accuracy_score(true_labels, predicted_label)
    f1 = f1_score(true_labels, predicted_label, average="weighted")
    fpr = 1 - accuracy

    metrics = {
        'Accuracy': round(accuracy, 3),
        'F1 score': round(f1, 3),
        'FPR': round(fpr, 3)
    }

    return metrics
