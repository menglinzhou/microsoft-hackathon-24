# processed_data_module.py

# Define the ProcessedData class
class ProcessedData:
    def __init__(self, data):
        """Initialize the class with processed data."""
        self.data = data

    def __getitem__(self, key):
        """Allow accessing columns directly like a dictionary."""
        return self.data[key]
    
    def __repr__(self):
        return f"ProcessedData with {len(self.data)} rows"
    