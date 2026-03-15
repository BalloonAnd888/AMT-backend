import os
from preprocessing.constants import *
from preprocessing.dataset import MAESTRO

def loadDataset(dataset_root):
    print(f"Initializing MAESTRO dataset from {dataset_root}...")
    train_dataset = MAESTRO(path=DATA_PATH, groups=['train'], sequence_length=SEQUENCE_LENGTH)
    validation_dataset = MAESTRO(path=DATA_PATH, groups=['validation'], sequence_length=SEQUENCE_LENGTH)
    test_dataset = MAESTRO(path=DATA_PATH, groups=['test'], sequence_length=SEQUENCE_LENGTH)

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

    if os.path.exists(DATA_PATH):
        loadDataset(DATA_PATH)
    else:
        print(f"Path not found: {DATA_PATH}")
        print("Please edit DATA_PATH in loadDataset.py")
