import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import pandas as pd
import requests
from PIL import Image, UnidentifiedImageError
from io import BytesIO
import os # For os.path.exists

from config import device, IMAGE_INDEX_FEATURES_PATH, IMAGE_INDEX_METADATA_PATH, INDEX_BATCH_SIZE # Import necessary configs
from datasets import SimpleIndexDataset # Import SimpleIndexDataset

class LostFoundApp:
    def __init__(self, model, initial_dataset_df=None, load_index_from_disk=True):
        self.model = model.to(device); self.device = device
        # Assuming ImageEncoder is part of the model and has a cnn.fc attribute
        encoder_output_size = model.image_encoder.cnn.fc.out_features
        self.image_features = torch.empty((0, encoder_output_size), device=self.device)
        self.image_urls, self.original_texts = [], []
        self.item_transform = transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
        ])

        loaded_from_disk = False
        if load_index_from_disk:
            loaded_from_disk = self._load_index_from_disk()

        if not loaded_from_disk and initial_dataset_df is not None and not initial_dataset_df.empty:
            print(f"DEBUG: LostFoundApp init, building index from DataFrame with {len(initial_dataset_df)} items.")
            idx_dataset = SimpleIndexDataset(initial_dataset_df, self.item_transform)
            self._build_index_from_dataset(idx_dataset)
            self._save_index_to_disk()
        elif not loaded_from_disk:
            print("DEBUG: LostFoundApp init, no initial dataset and no saved index found.")


    def _load_index_from_disk(self):
        if os.path.exists(IMAGE_INDEX_FEATURES_PATH) and os.path.exists(IMAGE_INDEX_METADATA_PATH):
            try:
                self.image_features = torch.load(IMAGE_INDEX_FEATURES_PATH, map_location=self.device)
                metadata = pd.read_pickle(IMAGE_INDEX_METADATA_PATH)
                self.image_urls = metadata['image_urls'].tolist()
                self.original_texts = metadata['original_texts'].tolist()
                print(f"DEBUG: Index loaded from disk. Features shape: {self.image_features.shape}, URLs: {len(self.image_urls)}")
                return True
            except Exception as e:
                print(f"DEBUG: Error loading index from disk: {e}. Will attempt to rebuild.")
                return False
        return False

    def _save_index_to_disk(self):
        try:
            torch.save(self.image_features, IMAGE_INDEX_FEATURES_PATH)
            metadata = pd.DataFrame({'image_urls': self.image_urls, 'original_texts': self.original_texts})
            metadata.to_pickle(IMAGE_INDEX_METADATA_PATH)
            print(f"DEBUG: Index saved to disk. Features: {IMAGE_INDEX_FEATURES_PATH}, Metadata: {IMAGE_INDEX_METADATA_PATH}")
        except Exception as e:
            print(f"DEBUG: Error saving index to disk: {e}")

    def _build_index_from_dataset(self, dataset, progress_callback=None):
        print(f"DEBUG: LostFoundApp._build_index_from_dataset with {len(dataset)} items.")
        loader_collate_fn = dataset.collate_fn if hasattr(dataset, 'collate_fn') else None
        loader = DataLoader(dataset, batch_size=INDEX_BATCH_SIZE, collate_fn=loader_collate_fn, num_workers=0)

        self.model.eval(); temp_features_list = []
        temp_urls_list, temp_texts_list = [], []

        with torch.no_grad():
            for i, batch in enumerate(loader):
                if not batch or batch['image'].numel()==0: continue
                images_on_device = batch['image'].to(self.device)
                features = self.model.image_encoder(images_on_device)
                if features.numel()>0:
                    temp_features_list.append(features.cpu())
                    temp_urls_list.extend(batch['image_url'])
                    temp_texts_list.extend(batch.get('original_text', ['N/A']*len(batch['image_url'])))
                if progress_callback: progress_callback(i+1, len(loader))

        if temp_features_list:
            self.image_features = torch.cat(temp_features_list).to(self.device)
            # Important: extend existing lists, don't overwrite if this method is called multiple times (though not typical for init)
            self.image_urls.extend(temp_urls_list)
            self.original_texts.extend(temp_texts_list)
        print(f"DEBUG: Index built. Features shape: {self.image_features.shape}, URLs: {len(self.image_urls)}")


    def add_new_item(self, image_url, item_title="N/A (Newly Added)"):
        self.model.eval()
        try:
            pil_image = self._load_image_from_url(image_url)
            if pil_image is None:
                return False, f"Could not load image from {image_url}"

            img_tensor = self.item_transform(pil_image).unsqueeze(0).to(self.device)
            with torch.no_grad(): new_feat = self.model.image_encoder(img_tensor)

            self.image_features = torch.cat([self.image_features, new_feat])
            self.image_urls.append(image_url)
            self.original_texts.append(item_title)
            self._save_index_to_disk()
            print(f"DEBUG: Item added. Index size: {len(self.image_urls)}")
            return True, f"Item added. Index size: {len(self.image_urls)}"
        except Exception as e:
            print(f"DEBUG: Error adding item {image_url}: {e}")
            return False, f"Error adding item: {e}"

    def _load_image_from_url(self, image_url):
        try:
            res = requests.get(image_url, timeout=10)
            res.raise_for_status()
            img_bytes = BytesIO(res.content)
            if img_bytes.getbuffer().nbytes > 0:
                return Image.open(img_bytes).convert('RGB')
        except requests.exceptions.RequestException as e:
            print(f"DEBUG: Network error loading image {image_url}: {e}")
        except UnidentifiedImageError:
            print(f"DEBUG: Cannot identify image file {image_url}")
        except Exception as e:
            print(f"DEBUG: Generic error loading image {image_url}: {e}")
        return None

    def _process_query_image(self, query_image_input):
        pil_image = None
        if isinstance(query_image_input, str):
            pil_image = self._load_image_from_url(query_image_input)
        elif isinstance(query_image_input, Image.Image):
            pil_image = query_image_input.convert('RGB')
        else:
            try:
                pil_image = Image.open(query_image_input).convert('RGB')
            except Exception as e:
                print(f"DEBUG: Could not process query image input: {e}")
                return None

        if pil_image is None:
            return None
        return self.item_transform(pil_image).unsqueeze(0).to(self.device)


    def search_by_text(self, description, top_k=5):
        self.model.eval()
        print(f"DEBUG: Text Search for '{description}', top_k={top_k}. Index size: {self.image_features.shape[0]}")
        if self.image_features.numel() == 0:
            return [], "Index is empty. Add items or check dataset."

        with torch.no_grad(): text_feat = self.model.text_encoder([description])
        if text_feat.numel() == 0:
            return [], "Failed to encode description."

        return self._find_similar(text_feat, top_k)

    def search_by_image(self, query_image_input, top_k=5):
        self.model.eval()
        print(f"DEBUG: Image Search, top_k={top_k}. Index size: {self.image_features.shape[0]}")
        if self.image_features.numel() == 0:
            return [], "Index is empty. Add items or check dataset."

        query_img_tensor = self._process_query_image(query_image_input)
        if query_img_tensor is None or query_img_tensor.numel() == 0:
            return [], "Failed to load or process query image."

        with torch.no_grad(): query_feat = self.model.image_encoder(query_img_tensor)
        if query_feat.numel() == 0:
            return [], "Failed to encode query image."

        return self._find_similar(query_feat, top_k)

    def _find_similar(self, query_feature_vector, top_k):
        if self.image_features.numel() == 0:
            return [], "Index is empty."

        scores = query_feature_vector @ self.image_features.T
        if scores.numel() == 0:
            return [], "No scores generated (empty index or query error)."

        k = min(top_k, self.image_features.shape[0])
        if k == 0:
            return [], "No items to rank (k=0)."

        top_scores, top_indices = torch.topk(scores.squeeze(), k)

        results = [{
            'image_url': self.image_urls[i.item()],
            'score': s.item(),
            'original_text': self.original_texts[i.item()]
        } for s, i in zip(top_scores, top_indices)]

        message = f"Found {len(results)} match(es)." if results else "No matches found."
        print(f"DEBUG: Search results: {len(results)} items. Message: {message}")
        return results, message