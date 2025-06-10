import torch
import os

# --- Device Configuration (CPU Only) ---
device = torch.device('cpu') 

# --- Paths ---
MODEL_PATH = "trained_model.pth"
BEST_MODEL_PATH = "best_trained_model.pth" # Path to save the best model
IMAGE_INDEX_FEATURES_PATH = "image_features.pt"
IMAGE_INDEX_METADATA_PATH = "image_metadata.pkl"
DATA_DIR = "dataset"
CSV_NAME = "data2.csv" # Make sure this is your larger, good quality dataset
CSV_PATH = os.path.join(DATA_DIR, CSV_NAME)

# --- Model & Training Parameters (Can be tuned) ---
EMBED_SIZE = 384 # This might change if your new TextEncoder has a fixed output size
# VOCAB_SIZE = 50000 
# MAX_TEXT_LEN = 32  

TRAIN_EPOCHS = 10 
TRAIN_LR = 0.001


# NEW Training Parameters
TRAIN_BATCH_SIZE = 32 # Define a batch size for training
VALIDATION_BATCH_SIZE = 32
VALIDATION_SPLIT_SIZE = 0.2 # 20% of data for validation
EARLY_STOPPING_PATIENCE = 5 # Number of epochs to wait for improvement before stopping

# --- Indexing Parameters ---
INDEX_SAMPLE_SIZE_MIN_ITEMS = 0 # This can be the full dataset after training
INDEX_BATCH_SIZE = 16

# --- UI Defaults ---
DEFAULT_TOP_K_RESULTS = 5
DEFAULT_RESULTS_COLS = 3