# model_helpers.py

import torch
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
from transformers import AutoTokenizer
import pandas as pd
from datasets import Dataset
# import helper functions from data_helpers.py
from processed_data_module import ProcessedData
from predicted_results_module import PredictionResults


__all__ = ["tokenize_data", "custom_collate_fn", "inference_model", "save_inference_to_csv", "read_inference_as_DataFrame", "read_inference_as_Dataset"] 


# Initialize the tokenizer and data collator outside the function to avoid re-initialization
tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-small")


def tokenize_data(data):
    """Tokenize the input data for training or evaluation

    Args:
        data (_type_): dataset object

    Raises:
        e: Exception raised during tokenization

    Returns:
        _type_: tokenized data
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-small")
        return tokenizer(data["text"], max_length=512, truncation=True, padding=True, return_tensors="pt")
    except Exception as e:
        print("Error during tokenization:", e)
        print("Offending examples:", data["text"])
        raise e


def custom_collate_fn(features):
    # Extract the 'text' field from each example if they are not already tokenized
    if 'input_ids' not in features[0]:
        texts = [feature['text'] for feature in features]
        # Tokenize the batch of texts using `__call__`
        return tokenizer(
            texts,
            padding=True,               # Pad sequences to the max length in the batch
            truncation=True,            # Truncate sequences that exceed max length
            max_length=512,             # Set a maximum length if necessary
            return_tensors="pt"         # Return as PyTorch tensors
        )
    
    # If features are already tokenized, filter only necessary keys and return as tensors
    else:
        batch = {
            'input_ids': [],
            'attention_mask': [],
            'token_type_ids': []
        }
        for feature in features:
            for key in batch.keys():
                if key in feature:
                    batch[key].append(feature[key])
        
        # Convert lists to tensors
        return {k: torch.tensor(v) for k, v in batch.items()}


def inference_model(model, data, nthreads = 4):
    """Inference function to get predicted probabilites for each label in the dataset

    Args:
        model: Pretrained or fined-tuned model
        data: ProcessedData instance
        nthreads (int): Number of threads to use for PyTorch

    Raises:
        TypeError: Input must be a `ProcessedData` instance

    Returns:
        _type_: predicted probabilities for labels `0` and `1` respectively
    """
    if isinstance(data, ProcessedData): 
        data = data.data
    elif isinstance(data, list):
        data = Dataset.from_dict({"text": data})
    elif isinstance(data, pd.DataFrame):
        data = Dataset.from_pandas(data)
    else:
        raise TypeError("Input must be a `ProcessedData` instance or a list of strings or a pandas DataFrame.")
    
    # Set the number of threads for PyTorch
    torch.set_num_threads(nthreads)

    # tokenize input data
    data = data.map(lambda x: tokenize_data(x), batched=True, desc="Tokenizing data")
    # load batches of tokenized data
    dataloader = DataLoader(data, batch_size=1, collate_fn=custom_collate_fn, shuffle=False)
    # create a list for the output prediction
    predictions = []
    # Counter for entries processed
    entry_count = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            # Apply softmax to get probabilities
            probs = softmax(logits, dim=-1).cpu().numpy()
            
            # Extract the probability of the positive class 
            predictions.extend(probs[:, 1])  # Assuming index 1 is the positive class
            
            # Update counter and show progress every 100 entries
            entry_count += len(probs)
            if entry_count % 100 == 0:
                print(f"Processed {entry_count} entries...")

    # Create an instance of PredictionResults with predictions
    predicted_ds = Dataset.from_dict({'Predicted_Probs(1)': [prob for prob in predictions]})
    results = PredictionResults(predicted_ds)

    return results


def save_inference_to_csv(prediction_results, data, csv_file_path):
    """
    Transform the predictions from a PredictionResults object into a Pandas DataFrame 
    and save it as a CSV file.

    Args:
        data: origianl dataset with all columns including true labels and machine classes
        prediction_results (PredictionResults): An instance of PredictionResults containing predictions.
        csv_file_path (str): The file path where the CSV will be saved.
    """
    # Check if the input is an instance of PredictionResults
    if isinstance(prediction_results, PredictionResults):
        # Extract predictions from the PredictionResults object
        predictions = prediction_results.predictions
    else:
        raise TypeError("Input must be a `PredictionResults` instance.")
    if isinstance(data, ProcessedData): 
        # Extract data from the ProcessedData object
        data = data.data
    else:
        raise TypeError("Input must be a `ProcessedData` instance.")

    # Check if the predictions can be combined with the original data
    if len(predictions) != len(data):
        raise ValueError("Length of predictions must match the length of original data.")
    
    # Convert list to DataFrame
    predictions_df = predictions.to_pandas()
    # Convert Dataset to DataFrame
    data_df = data.to_pandas()
    
    # Combine original data with predictions
    combined_data = pd.concat([data_df.reset_index(drop=True), predictions_df.reset_index(drop=True)], axis=1)

    # Save the combined DataFrame as a CSV file
    combined_data.to_csv(csv_file_path, index=False)
    print(f"Combined DataFrame saved successfully.")

def read_inference_as_DataFrame(csv_file_path):
    # Load the CSV file into a Pandas DataFrame
    df = pd.read_csv(csv_file_path)
    
    return df

def read_inference_as_Dataset(csv_file_path):
    # Load the CSV file into a Pandas DataFrame
    df = pd.read_csv(csv_file_path)
    
    # Convert the DataFrame to a datasets.Dataset
    dataset = Dataset.from_pandas(df)
    
    return dataset
