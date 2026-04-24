import os
from preprocessing.constants import *
from preprocessing.dataset import MAESTRO, MAPS

def loadDataset(dataset_name="maestro", dataset_root=None):
    if dataset_name.lower() == "maestro":
        path = dataset_root if dataset_root else DATA_PATH
        print(f"Initializing MAESTRO dataset from {path}...")
        train_dataset = MAESTRO(path=path, groups=['train'], sequence_length=SEQUENCE_LENGTH)
        validation_dataset = MAESTRO(path=path, groups=['validation'], sequence_length=SEQUENCE_LENGTH)
        test_dataset = MAESTRO(path=path, groups=['test'], sequence_length=SEQUENCE_LENGTH)
    elif dataset_name.lower() == "maps":
        path = dataset_root if dataset_root else MAPS_DATA_PATH
        print(f"Initializing MAPS dataset from {path}...")
        train_dataset = MAPS(path=path, groups=['train'], sequence_length=SEQUENCE_LENGTH)
        validation_dataset = MAPS(path=path, groups=['validation'], sequence_length=SEQUENCE_LENGTH)
        test_dataset = MAPS(path=path, groups=['test'], sequence_length=SEQUENCE_LENGTH)
    else:
        print(f"Error: Unsupported dataset '{dataset_name}'")
        return 

    if len(train_dataset) == 0:
        print("Error: Dataset is empty. Check your path and metadata")
        return 
    if len(validation_dataset) == 0:
        print("Error: Dataset is empty. Check your path and metadata")
        return 
    if len(test_dataset) == 0:
        print("Error: Dataset is empty. Check your path and metadata")
        return 
        
    return train_dataset, validation_dataset, test_dataset

if __name__ == "__main__":
    print(f"Using device: {DEVICE}")

    dataset_choice = "maestro"  # Change to "maps" to load MAPS dataset
    path_to_check = DATA_PATH if dataset_choice == "maestro" else MAPS_DATA_PATH

    if os.path.exists(path_to_check):
        loadDataset(dataset_choice, path_to_check)
    else:
        print(f"Path not found: {path_to_check}")
        print(f"Please edit constants.py to fix {dataset_choice.upper()} path")
