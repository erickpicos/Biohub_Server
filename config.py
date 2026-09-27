from pathlib import Path

#Folders
DATASET_FOLDER = Path("dataset")
ONNX_FOLDER = Path("onnx")

#Training
TEST_SIZE = 0.2
BATCH_SIZE = 4
EPOCHS = 100
LEARNING_RATE = 0.001

DATASET_TYPE = "Filtered" # ["Raw", "Filtered", "ZScore"]

#EEG
N_CHANNELS = 4
N_CLASSES = 3
N_SAMPLES = 1000
SAMPLING_RATE = 250
WINDOW_SECONDS = 4

#Labels
LABEL_MAP = { "Left": 0, "Rest": 1, "Right": 2 }