import torch
import os

# --- Device Configuration (CPU Only) ---
device = torch.device('cpu')

# --- Paths ---
MODEL_PATH = "trained_model.pth"
IMAGE_INDEX_FEATURES_PATH = "image_features.pt"
IMAGE_INDEX_METADATA_PATH = "image_metadata.pkl"
DATA_DIR = "dataset"
CSV_NAME = "amazon_products.csv"
CSV_PATH = os.path.join(DATA_DIR, CSV_NAME)

# --- Model & Training Parameters (Can be tuned) ---
EMBED_SIZE = 256
VOCAB_SIZE = 50000 # Increased from original for potentially better hashing
MAX_TEXT_LEN = 32
TRAIN_EPOCHS = 10
TRAIN_LR = 0.0001
TRAIN_BATCH_SIZE_MIN_ITEMS = 8 # Min items for batch_size in training
TRAIN_SAMPLE_SIZE_MIN_ITEMS = 50 # Min items for training sample

# --- Indexing Parameters ---
INDEX_SAMPLE_SIZE_MIN_ITEMS = 100
INDEX_BATCH_SIZE = 16

# --- UI Defaults ---
DEFAULT_TOP_K_RESULTS = 5
DEFAULT_RESULTS_COLS = 3