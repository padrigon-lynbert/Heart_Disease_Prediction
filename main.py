import openml

# Load the dataset with ID 37
dataset = openml.datasets.get_dataset(37)

# Get basic information about the dataset
print(f"Name: {dataset.name}")
print(f"Description: {dataset.description[:500]}")  # Print the first 500 characters of the description
print(f"Features: {dataset.features}")
print(f"Number of instances: {dataset.number_of_instances}")
print(f"Number of features: {dataset.number_of_features}")
