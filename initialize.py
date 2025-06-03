import streamlit as st # For @st.cache_resource
import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import pandas as pd

from config import (
    device, MODEL_PATH, CSV_PATH, DATA_DIR,
    IMAGE_INDEX_FEATURES_PATH, IMAGE_INDEX_METADATA_PATH,
    TRAIN_SAMPLE_SIZE_MIN_ITEMS, TRAIN_BATCH_SIZE_MIN_ITEMS,
    INDEX_SAMPLE_SIZE_MIN_ITEMS
)
from encoders import ImageEncoder, TextEncoder
from system import LostFoundSystem, minimal_train_model
from datasets import LostFoundDataset # SimpleIndexDataset is used within LostFoundApp
from app_logic import LostFoundApp

@st.cache_resource(show_spinner="Initializing system... Please wait.")
def initialize_system():
    print("DEBUG: Entered initialize_system()")

    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(CSV_PATH):
        print(f"DEBUG: {CSV_PATH} not found, creating dummy.")
        pd.DataFrame({
            'title': [f'Dummy Product {idx}' for idx in range(20)],
            'imgUrl': [f'https://via.placeholder.com/150?text=Dummy{idx+1}' for idx in range(20)]
        }).to_csv(CSV_PATH, index=False)
        print(f"DEBUG: Dummy data created at {CSV_PATH}")

    image_enc = ImageEncoder().to(device)
    text_enc = TextEncoder().to(device)
    model = LostFoundSystem(image_enc, text_enc).to(device)
    print("DEBUG: Core model initialized on CPU.")

    model_loaded_or_trained = False
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            print(f"DEBUG: Pre-trained model loaded from {MODEL_PATH}")
            model_loaded_or_trained = True
        except Exception as e:
            print(f"DEBUG: Error loading {MODEL_PATH}: {e}. Will attempt minimal training if needed.")

    if not model_loaded_or_trained:
        print(f"DEBUG: {MODEL_PATH} not found or load failed, starting minimal training.")
        try:
            df_check = pd.read_csv(CSV_PATH)
            if df_check.empty:
                raise ValueError(f"{CSV_PATH} is empty. Cannot train.")

            train_sample_size = min(TRAIN_SAMPLE_SIZE_MIN_ITEMS, len(df_check))
            df_minimal = df_check.sample(n=train_sample_size, random_state=42)
            print(f"DEBUG: Using {len(df_minimal)} samples for minimal training.")

            transform = transforms.Compose([
                transforms.Resize(256), transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
            ])
            min_dataset = LostFoundDataset(df_minimal, transform)
            if len(min_dataset) > 0:
                train_batch_size = min(TRAIN_BATCH_SIZE_MIN_ITEMS, len(min_dataset))
                min_loader = DataLoader(min_dataset, batch_size=train_batch_size, collate_fn=min_dataset.collate_fn, shuffle=True)
                model = minimal_train_model(model, min_loader, model_save_path=MODEL_PATH) # num_epochs from config default
                model_loaded_or_trained = True
            else:
                print("DEBUG: Minimal training dataset was empty after processing. No training performed.")
        except Exception as e_train:
            print(f"DEBUG: Minimal training failed: {e_train}. Model remains un-trained or uses previously loaded state if any.")

    initial_df_for_index_build = pd.DataFrame()
    if not (os.path.exists(IMAGE_INDEX_FEATURES_PATH) and os.path.exists(IMAGE_INDEX_METADATA_PATH)):
        try:
            full_df = pd.read_csv(CSV_PATH)
            if not full_df.empty:
                index_sample_size = min(INDEX_SAMPLE_SIZE_MIN_ITEMS, len(full_df))
                initial_df_for_index_build = full_df.sample(n=index_sample_size, random_state=123)
                print(f"DEBUG: Prepared DataFrame with {len(initial_df_for_index_build)} items for potential index build.")
        except Exception as e_df:
            print(f"DEBUG: Error reading/sampling {CSV_PATH} for index: {e_df}")

    app_instance = LostFoundApp(model, initial_dataset_df=initial_df_for_index_build, load_index_from_disk=True)

    if app_instance.image_features.numel() == 0:
        print("WARNING: Index is empty after initialization. App may not function correctly.")
    if not model_loaded_or_trained:
        print("WARNING: Model was not loaded from disk and training failed or was skipped. Search quality will be poor.")

    print("DEBUG: initialize_system() finished.")
    return app_instance