from pathlib import Path
import pandas as pd
import numpy as np

from config import *

def load_dataset(user_id: int, data_type: str):

    valid_types = ["Raw", "Filtered", "ZScore"]

    if data_type not in valid_types:
        raise ValueError(f"Invalid data_type '{data_type}'. Valid options are: {valid_types}")

    user_folder = DATASET_FOLDER / f"User_{user_id}" / data_type
    
    if not user_folder.exists(): 
        raise ValueError(f"Dataset folder not found: {user_folder}")

    csv_files = list(user_folder.glob("*.csv"))

    X = []
    y = []

    for csv_file in csv_files:
        filename = csv_file.stem
        label = None

        for key, value in LABEL_MAP.items():
            if f"Task{key}" in filename:
                label = value
                break

        if label is None:
            continue

        df = pd.read_csv(csv_file)
        data = df[["C1", "C2", "C3", "C4"]].values.astype(np.float32)

        if data.shape != (1000, 4):
            raise ValueError( f"Invalid shape in {csv_file.name}: {data.shape}" )

        #(1000,4) -> (4,1000)
        data = data.T

        X.append(data)
        y.append(label)

    if len(X) == 0:
        raise ValueError( f"No valid EEG windows found in {user_folder}" )
        
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    print(f"Loaded dataset: {X.shape}")
    print(f"Labels: {y.shape}")

    return X, y