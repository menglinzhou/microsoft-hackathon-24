# data_helpers.py

import datasets
from datasets import Dataset, concatenate_datasets
import pandas as pd
from collections import Counter
from processed_data_module import ProcessedData
from typing import Union

__all__ = ["process_data", "is_processed_data", "train_test_split_unequal_class", "train_test_split_equal_class", "label_counter",
           "check_column_type", "check_column_values", "check_column_string"]


def process_data(data, text_col = "generation", label_col = "model", reduced = False):
    """Format the input data for training or evaluation

    Args:
        data: a pandas DataFrame or a datasets.Dataset object
        text_col (string): text column name
        label_col (string): label column name

    Returns:
        _type_: formalized data
    """
    
    # Ensure the input data is in a proper format
    def ensure_dataset(df):
        # Check if the object is already a datasets.Dataset
        if isinstance(df, Dataset):
            return df  # It's already a Dataset, so return it as-is
        # Check if the object is a pandas DataFrame
        elif isinstance(df, pd.DataFrame):
            return Dataset.from_pandas(df)  # Convert DataFrame to Dataset
        elif is_processed_data(data): 
            pass
        else:
            raise TypeError("Input must be a pandas DataFrame or a datasets.Dataset object.")
    ensure_dataset(data)

    if is_processed_data(data):
        data = data.data
        if reduced:
            data = data.remove_columns([col for col in data.column_names if col not in ["text", "labels"]])
        return ProcessedData(data)

    # Add the text and labels columns for training
    if text_col != "text":
        data = data.rename_column(text_col, "text") # rename the text column directly
    if label_col != "labels":
        def mutate_column(temp):
            if label_col != "labels":
                temp['labels'] = temp['model']  # Create 'labels' column with 'model' values
            return temp
        data = data.map(mutate_column, desc="Creating labels column")
    
    # Drop NAs
    data = data.filter(lambda x: x["text"] is not None and x["labels"] is not None, desc="Dropping NAs")

    # Transform `labels` to binary 0/1 if needed
    def transform_labels(input):
        if input["labels"] == 0 or input["labels"] == 1: # check if the label is already 0 or 1
            return {"labels": input["labels"]}
        elif "human" in str(input["labels"]).lower(): # check if the label contains "human" in a case-insensitive manner
            return {"labels": 0}
        else: # if the label is not 0, 1, or "human", assume it's machine-generated
            return {"labels": 1}
    # Get the unique labels
    unique_labels = set(data["labels"])
    # Apply the label transformation if needed
    if not (unique_labels == {0, 1}):
        data = data.map(transform_labels, desc="Transforming labels to binary 0/1")
    
    # Transform `labels` to integer if needed
    if not check_column_values(data, "labels"):
        data = data.map(lambda x: {"labels": int(x["labels"])}, desc="Converting labels to numeric")

    # Reduce the data columns if needed
    if reduced:
        data = data.remove_columns([col for col in data.column_names if col not in ["text", "labels"]])

    return ProcessedData(data)


def is_processed_data(data) -> bool:
    """Check if data is an instance of ProcessedData"""
    return isinstance(data, ProcessedData)


def train_test_split_unequal_class(data, test_size=0.2, seed=42):
    """Split the dataset into train and test sets.

    Args:
        test_size (float): Proportion of the dataset to include in the test split.
        seed (int): Random seed for reproducibility.

    Returns:
        (ProcessedData, ProcessedData): Train and test sets as ProcessedData instances.
    """
    # Ensure `data` is a Dataset or ProcessedData instance with Dataset data
    if is_processed_data(data):
        data = data.data
    elif isinstance(data, pd.DataFrame):
        data = Dataset.from_pandas(data)
    elif not isinstance(data, Dataset):
        raise TypeError("Input must be a `Dataset` or `ProcessedData` instance.")

    # Perform the train-test split
    split_data = data.train_test_split(test_size=test_size, shuffle=True, seed=seed)

    # Verify that split_data is a dictionary and contains 'train' and 'test'
    if isinstance(split_data, dict) and 'train' in split_data and 'test' in split_data:
        # Return train and test sets as new ProcessedData instances
        train_set = ProcessedData(split_data['train'])
        test_set = ProcessedData(split_data['test'])
        # Wrap the output in ProcessedData if the input was ProcessedData
        if is_processed_data:
            return ProcessedData(train_set), ProcessedData(test_set)
        else:
            return train_set, test_set
    else:
        raise ValueError("Unexpected format for split data. Expected a dictionary with keys 'train' and 'test'.")


def train_test_split_equal_class(data, test_size=0.5, seed=42):
    """Split the dataset into train and test sets with equal counts of labels 0 and 1.

    Args:
        data (ProcessedData or Dataset): The dataset to split, expected to have a `labels` column.
        test_size (float): Proportion of each label group to assign to the test set (e.g., 0.5 for 50%).
        seed (int): Random seed for reproducibility.

    Returns:
        (Dataset, Dataset): Train and test sets as new Dataset instances with equal numbers of labels 0 and 1.
    """
    # Ensure `data` is a Dataset or ProcessedData instance with Dataset data
    if isinstance(data, ProcessedData):
        data = data.data
    elif isinstance(data, pd.DataFrame):
        data = Dataset.from_pandas(data)
    elif not isinstance(data, Dataset):
        raise TypeError("Input must be a `Dataset` or `ProcessedData` instance.")

    # Separate data into two subsets based on label values
    label_0 = data.filter(lambda x: x["labels"] == 0, desc="Filtering label 0")
    label_1 = data.filter(lambda x: x["labels"] == 1, desc="Filtering label 1")
    
    # Get the smaller size of the two label groups for balanced splitting
    min_count = min(len(label_0), len(label_1))

    # Shuffle and truncate each label subset to the minimum count
    label_0 = label_0.shuffle(seed=seed).select(range(min_count))
    label_1 = label_1.shuffle(seed=seed).select(range(min_count))
    
    # Calculate split index based on test_size
    test_count = int(min_count * test_size)
    train_count = min_count - test_count

    # Split each label subset into train and test sets
    label_0_train = label_0.select(range(train_count))
    label_0_test = label_0.select(range(train_count, min_count))
    label_1_train = label_1.select(range(train_count))
    label_1_test = label_1.select(range(train_count, min_count))

    # Concatenate the stratified splits to form train and test sets
    train_set = concatenate_datasets([label_0_train, label_1_train])
    test_set = concatenate_datasets([label_0_test, label_1_test])

    # Shuffle the final train and test sets to mix the labels
    train_set = train_set.shuffle(seed=seed)
    test_set = test_set.shuffle(seed=seed)

    # Wrap the output in ProcessedData if the input was ProcessedData
    if is_processed_data:
        return ProcessedData(train_set), ProcessedData(test_set)
    else:
        return train_set, test_set


def label_counter(data):
    """Count the occurrences of each label"""
    label_counts = Counter(data["labels"])
    # Print the counts of 0s and 1s
    print(f"Count of human-written entries: {label_counts[0]}")
    print(f"Count of machine-generated entries: {label_counts[1]}")
    return

def check_column_type(data, column_name: str) -> Union[str, None]:
    if isinstance(data, ProcessedData):
        data = data.data

    """Check the data type of the specified column."""
    # Access the features of the dataset (assuming it's a single split)
    column_feature = data.features.get(column_name)
    if column_feature is None:
        return f"Column '{column_name}' does not exist in the dataset."
    
    # Determine the type of the column
    if isinstance(column_feature, datasets.ClassLabel):
        return "categorical (ClassLabel)"
    elif isinstance(column_feature, datasets.Value):
        if column_feature.dtype in ['int32', 'int64', 'float32', 'float64']:
            return "numeric"
        else:
            return "Value type but not numeric"
    else:
        return "Unknown type"

def check_column_values(data, column_name: str) -> bool:
    if isinstance(data, ProcessedData):
        data = data.data

    """Check if the column is numeric"""
    column_feature = data.features.get(column_name)
    if column_feature is None:
        return f"Column '{column_name}' does not exist in the dataset."
    return isinstance(column_feature, datasets.Value)

def check_column_string(data, column_name: str) -> bool:
    if isinstance(data, ProcessedData):
        data = data.data

    """Check if the column is a list of strings"""
    column_feature = data.features.get(column_name)
    if column_feature is None:
        return f"Column '{column_name}' does not exist in the dataset."
    return isinstance(column_feature, datasets.ClassLabel)
