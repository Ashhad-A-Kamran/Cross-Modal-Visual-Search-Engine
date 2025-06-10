import streamlit as st
import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import pandas as pd
from sklearn.model_selection import train_test_split 

from config import (
    device, MODEL_PATH, BEST_MODEL_PATH, CSV_PATH, DATA_DIR,
    IMAGE_INDEX_FEATURES_PATH, IMAGE_INDEX_METADATA_PATH,
    # TRAIN_SAMPLE_SIZE_MIN_ITEMS, TRAIN_BATCH_SIZE_MIN_ITEMS, # Old params
    TRAIN_EPOCHS, TRAIN_LR, TRAIN_BATCH_SIZE, VALIDATION_BATCH_SIZE, VALIDATION_SPLIT_SIZE, # New params
    INDEX_SAMPLE_SIZE_MIN_ITEMS
)
from encoders import ImageEncoder, TextEncoder # Make sure these are the updated encoders later
from system import LostFoundSystem, train_model 
from datasets import LostFoundDataset
from app_logic import LostFoundApp

@st.cache_resource(show_spinner="Initializing system... Please wait.")
def initialize_system():
    print("DEBUG: Entered initialize_system()")
    os.makedirs(DATA_DIR, exist_ok=True)

    # Ensure CSV_PATH exists 
    if not os.path.exists(CSV_PATH):
        print(f"DEBUG: {CSV_PATH} not found, creating dummy for basic run.")


    # --- Initialize Model Components ---
    # Ensure EMBED_SIZE in config matches TextEncoder output if it's fixed
    image_enc = ImageEncoder().to(device) # ImageEncoder's embed_size should match TextEncoder's
    text_enc = TextEncoder().to(device)   # TextEncoder might now have a fixed embed_size
    model = LostFoundSystem(image_enc, text_enc).to(device)
    print("DEBUG: Core model initialized.")

    model_loaded_from_best = False
    if os.path.exists(BEST_MODEL_PATH): # Try loading the best model first
        try:
            model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
            print(f"DEBUG: Pre-trained best model loaded from {BEST_MODEL_PATH}")
            model_loaded_from_best = True
        except Exception as e:
            print(f"DEBUG: Error loading {BEST_MODEL_PATH}: {e}. Will check for {MODEL_PATH} or retrain.")

    if not model_loaded_from_best and os.path.exists(MODEL_PATH): # Fallback to regular MODEL_PATH
         try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            print(f"DEBUG: Pre-trained model loaded from {MODEL_PATH}")
            model_loaded_from_best = True # Treat as loaded
         except Exception as e:
            print(f"DEBUG: Error loading {MODEL_PATH}: {e}. Will attempt training.")


    if not model_loaded_from_best:
        print(f"DEBUG: No pre-trained model found or load failed. Starting full training.")
        try:
            full_df = pd.read_csv(CSV_PATH)
            if full_df.empty or len(full_df) < (TRAIN_BATCH_SIZE * 2): # Ensure enough data
                raise ValueError(f"{CSV_PATH} is empty or too small for meaningful training.")

            # --- Data Splitting ---
            train_df, val_df = train_test_split(full_df, test_size=VALIDATION_SPLIT_SIZE, random_state=42)
            print(f"DEBUG: Data split: {len(train_df)} training samples, {len(val_df)} validation samples.")

            transform = transforms.Compose([
                transforms.Resize(256), transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
            ])

            train_dataset = LostFoundDataset(train_df, transform)
            val_dataset = LostFoundDataset(val_df, transform)

            if len(train_dataset) > 0 and len(val_dataset) > 0:
                train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE,
                                          collate_fn=train_dataset.collate_fn, shuffle=True, num_workers=0) # num_workers > 0 can speed up if I/O is bottleneck
                val_loader = DataLoader(val_dataset, batch_size=VALIDATION_BATCH_SIZE,
                                        collate_fn=val_dataset.collate_fn, shuffle=False, num_workers=0)

                # Call the enhanced training function
                model = train_model(model, train_loader, val_loader,
                                    num_epochs=TRAIN_EPOCHS, model_save_path=MODEL_PATH, best_model_save_path=BEST_MODEL_PATH,
                                    learning_rate=TRAIN_LR) # Pass other params like device, early_stopping_patience from config
                model_loaded_from_best = True # Mark as trained
            else:
                print("DEBUG: Training or validation dataset was empty after processing. No training performed.")
        except Exception as e_train:
            print(f"DEBUG: Full training failed: {e_train}. Model remains un-trained or uses previously loaded state if any.")

    # --- Index Building (can use full dataset or a sample) ---
    initial_df_for_index_build = pd.DataFrame()
    # Logic to decide if index needs rebuilding or can be loaded by LostFoundApp
    try:
        df_for_app_index = pd.read_csv(CSV_PATH)
        initial_df_for_index_build = df_for_app_index
    except Exception as e_df:
        print(f"DEBUG: Error reading/sampling {CSV_PATH} for index: {e_df}")


    app_instance = LostFoundApp(model, initial_dataset_df=initial_df_for_index_build, load_index_from_disk=True)

    if app_instance.image_features.numel() == 0:
        print("WARNING: Index is empty after initialization. App may not function correctly.")
    if not model_loaded_from_best:
        print("WARNING: Model was not loaded from disk and training failed or was skipped. Search quality will be poor.")

    print("DEBUG: initialize_system() finished.")
    return app_instance