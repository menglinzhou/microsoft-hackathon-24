# evaluate_helpers.py
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score
# import helper functions from data_helpers.py
from data_helpers import is_processed_data
from predicted_results_module import PredictionResults

__all__ = ["get_predicted_labels", "evaluate_model"]


def get_predicted_labels(predictions, threshold=0.5):
    """Process the model predictions to get the final labels given a threshold

    Args:
        predictions (float): PredictionResults instance
        threshold (float, optional): the lowest probability of predicting an entry as machine-generated texts. Defaults to 0.5.

    Returns:
        _type_: _description_
    """
    # Check if the input is an instance of PredictionResults
    if isinstance(predictions, PredictionResults):
        # Extract predictions from the PredictionResults object
        predictions = predictions.predictions['Predicted_Probs(1)']
    elif isinstance(predictions, list):
        pass
    else:
        raise TypeError("Input must be a `PredictionResults` instance or a list.")
    

    # Apply thresholding to get the predicted labels
    predicted_labels = [1 if prob >= threshold else 0 for prob in predictions]
    return predicted_labels


def evaluate_model(predicted_prob, true_label, threshold = 0.5, print_metrics = True):
    """Compute metrics using the true label and the predicted label

    Args:
        predicted_prob: A `PredictionResults` object or a list containing predicted probabilities
        true_label: A `ProcessedData` object or a list containing true labels
        threshold (float, optional): minimum probability to predict to label `1`. Defaults to 0.5.
        roc_curve (bool, optional): if print ROC curve. Defaults to True.
        print_metrics (bool, optional): if print metrics. Defaults to True.

    Returns:
        dictionary: evaluation matrics
    """
    
    # Check the input types
    if isinstance(predicted_prob, PredictionResults):
        predicted_prob = predicted_prob.predictions['Predicted_Probs(1)']
    elif isinstance(predicted_prob, list):
        pass
    else:
        raise TypeError("`predicted_prob` must be a `PredictionResults` instance or a list.")
    
    if is_processed_data(true_label):
        true_label = true_label.data['labels']
    elif isinstance(true_label, list):
        pass
    else:
        raise TypeError("`true_label` must be a `ProcessedData` instance or a list.")
    

    # Compute AUC
    auc_score = roc_auc_score(true_label, predicted_prob)

    # Get the predicted labels
    predicted_label = get_predicted_labels(predicted_prob, threshold = 0.85)

    accuracy = accuracy_score(true_label, predicted_label)
    precision = precision_score(true_label, predicted_label)
    recall = recall_score(true_label, predicted_label)
    f1 = f1_score(true_label, predicted_label, average="weighted")
    specificity = recall_score(true_label, predicted_label, pos_label=0)

    # Compute confusion matrix to derive FPR
    tn, fp, fn, tp = confusion_matrix(true_label, predicted_label).ravel()
    fprate = fp / (fp + tn)  # False Positive Rate
    
    metrics = {
        'AUC': round(auc_score, 3),
        'Accuracy': round(accuracy, 3),
        'Precision': round(precision, 3),
        'Recall': round(recall, 3),
        'Weighted F1 score': round(f1, 3),
        'Specificity': round(specificity, 3),
        'Confusion Matrix': [[tn, fp], [fn, tp]],
        'FPR': round(fprate, 3)
    }
    
    if print_metrics:
        print("Overall Metrics given threshold of {:.2f}".format(threshold))
        print("===============")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"False Positive Rate: {fprate:.3f}")
        print(f"Weighted F1 score: {f1:.3f}")
        print(f"Specificity: {specificity:.3f}")
        print("Confusion Matrix:")
        print(pd.DataFrame([[tn, fp], [fn, tp]], index=['True 0', 'True 1'], columns=['Predicted 0', 'Predicted 1']))
    
    return metrics
