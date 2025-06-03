# Cross-Modal Visual Search Engine

This project implements a visual search engine that allows users to find images from an indexed collection based on either textual descriptions or query images. It leverages deep learning models (ResNet18 for images, GRU for text) to create a joint embedding space where similar images and text descriptions are mapped closely together. The application is built with Python, PyTorch, and Streamlit.

## Features

*   **Text-to-Image Search:** Input a textual description to find matching images.
*   **Image-to-Image Search:** Upload an image or provide an image URL to find visually similar images.
*   **Dynamic Indexing:** Add new items (images with optional titles) to the search index at runtime.
*   **Minimal Model Training:** If a pre-trained model is not found, the system performs a minimal training routine on a sample of the dataset to get started.
*   **Persistent Index:** Image embeddings and metadata are saved to disk, so the index is reloaded on subsequent runs.
*   **CPU-Optimized:** Designed to run efficiently on CPUs.
*   **Interactive Web UI:** Built with Streamlit for an easy-to-use interface.
*   **Dummy Data Generation:** Includes functionality to create dummy data if no dataset is provided, allowing the app to run out-of-the-box.

## Project Structure

The project is organized into several Python modules for better maintainability:
```
Cross-Modal-Visual-Search-Engine/
├── app.py
├── config.py 
├── datasets.py 
├── encoders.py
├── system.py
├── app_logic.py
├── initialize.py
├── requirements.txt
└── dataset/ 
└── amazon_products.csv
```
## Technical Overview

1.  **Embedding Generation:**
    *   **Image Encoder (`encoders.ImageEncoder`):** Uses a pre-trained ResNet18 model (with a modified final layer) to convert images into fixed-size dense vector embeddings.
    *   **Text Encoder (`encoders.TextEncoder`):** Uses an Embedding layer followed by a GRU (Gated Recurrent Unit) to convert text descriptions into vector embeddings. A simple hashing-based tokenizer is currently used for simplicity.
2.  **Joint Embedding Space (`system.LostFoundSystem`):**
    *   The `LostFoundSystem` combines the image and text encoders.
    *   During a minimal training phase (`system.minimal_train_model`), it learns to map related images and texts to nearby points in the embedding space using a contrastive loss function (similar to CLIP). This encourages embeddings of corresponding (image, text) pairs to have high cosine similarity, while non-corresponding pairs have low similarity.
3.  **Indexing (`app_logic.LostFoundApp`):**
    *   A collection of images (from `dataset/amazon_products.csv` or added dynamically) is processed.
    *   For each image, the `ImageEncoder` generates its embedding.
    *   These image embeddings, along with their URLs and original titles, are stored in memory and persisted to disk (`image_features.pt`, `image_metadata.pkl`). This forms the searchable "image space."
4.  **Searching (`app_logic.LostFoundApp`):**
    *   **Text Query:** The input text is encoded by the `TextEncoder`. The resulting text embedding is compared (using cosine similarity, achieved via dot product of normalized vectors) against all image embeddings in the index.
    *   **Image Query:** The input image is encoded by the `ImageEncoder`. The resulting image embedding is compared against all other image embeddings in the index.
    *   The top-K most similar items are returned as search results.
5.  **Initialization (`initialize.initialize_system`):**
    *   Handles the entire setup process:
        *   Checks for/creates the dataset CSV.
        *   Initializes model components.
        *   Attempts to load a pre-trained model (`trained_model.pth`).
        *   If loading fails or the model doesn't exist, performs minimal training.
        *   Initializes the `LostFoundApp`, which tries to load the image index from disk or builds it from the dataset.
    *   This function is cached by Streamlit (`@st.cache_resource`) for efficiency.

## Setup and Installation

1.  **Clone the Repository (Optional - if you have it in a git repo):**
    ```bash
    git clone <repository_url>
    cd Cross-Modal-Visual-Search-Engine
    ```
    If you just have the files, create a project directory and place all the `.py` files inside it.

2.  **Create a Python Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    Ensure you have the `requirements.txt` file in your project directory with the following content:
    ```
    streamlit
    torch
    torchvision
    pandas
    requests
    Pillow
    ```
    Then run:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Application

1.  **Navigate to the Project Directory:**
    Open your terminal or command prompt and change to the directory where `app.py` and the other project files are located.
    ```bash
    cd path/to/your/Cross-Modal-Visual-Search-Engine
    ```

2.  **Run the Streamlit App:**
    ```bash
    streamlit run app.py
    ```

3.  **Access the Application:**
    Streamlit will typically open the application in your default web browser automatically. If not, open your browser and navigate to the URL displayed in the terminal (usually `http://localhost:8501`).

**First Run Notes:**
*   The first time you run the application, if `trained_model.pth`, `image_features.pt`, or `dataset/amazon_products.csv` do not exist, the system will:
    *   Create a dummy `dataset/amazon_products.csv`.
    *   Perform a minimal training routine for the model and save `trained_model.pth`.
    *   Build an initial image index from the dataset and save `image_features.pt` and `image_metadata.pkl`.
*   This initial setup might take a few minutes depending on your system. Subsequent runs will be faster as the model and index will be loaded from disk.

## Code Explanation

*   **`config.py`**: Centralizes all constants, file paths, and hyperparameters. This makes it easy to tune the system without digging through multiple files.
*   **`datasets.py`**:
    *   `LostFoundDataset`: Prepares image-text pairs from a Pandas DataFrame for training. It handles image downloading, basic text cleaning, and transformations.
    *   `SimpleIndexDataset`: A simpler dataset used specifically for preparing images when building the initial search index.
*   **`encoders.py`**:
    *   `ImageEncoder`: A PyTorch `nn.Module` that uses a pre-trained ResNet18 (with its classifier head replaced) to generate image embeddings.
    *   `TextEncoder`: An `nn.Module` that uses an `nn.Embedding` layer and an `nn.GRU` to generate text embeddings from tokenized input.
*   **`system.py`**:
    *   `LostFoundSystem`: An `nn.Module` that encapsulates the `ImageEncoder` and `TextEncoder`. It also holds a learnable `temperature` parameter used in the contrastive loss.
    *   `minimal_train_model`: Implements the training loop. It calculates a symmetric contrastive loss to pull embeddings of matching image-text pairs closer and push non-matching pairs apart.
*   **`app_logic.py`**:
    *   `LostFoundApp`: The heart of the application's backend logic.
        *   `__init__`: Initializes the model, image transformations, and loads/builds the image index.
        *   `_load_index_from_disk`, `_save_index_to_disk`: Handle persistence of the image features and metadata.
        *   `_build_index_from_dataset`: Populates the search index by encoding images from a dataset.
        *   `add_new_item`: Downloads, encodes, and adds a new image to the index.
        *   `search_by_text`, `search_by_image`: Implement the respective search functionalities by encoding the query and calling `_find_similar`.
        *   `_find_similar`: Computes cosine similarities between the query embedding and all indexed image embeddings and returns the top-k matches.
*   **`initialize.py`**:
    *   `initialize_system`: Orchestrates the entire setup. It's decorated with `@st.cache_resource` to ensure that this potentially heavy initialization is done only once per Streamlit session or when its code changes.
*   **`app.py`**:
    *   The main Streamlit script that defines the user interface.
    *   It calls `initialize_system()` to get the backend `app_instance`.
    *   It uses Streamlit widgets (radio buttons, text inputs, file uploaders, sliders, buttons) to get user input.
    *   It calls methods on the `app_instance` (e.g., `search_by_text`, `add_new_item`) based on user interaction.
    *   It displays search results, messages, and system information.

## Group Members

Created Ashhad, Xintong, Maaz ACSAI Batch 2023
