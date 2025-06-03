import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import requests
from PIL import Image, UnidentifiedImageError
from io import BytesIO
import re

from config import device

class LostFoundDataset(Dataset):
    def __init__(self, df, transform=None):
        self.transform = transform
        self.df = df.copy()
        if 'title' not in self.df.columns: self.df['title'] = "No Title"
        if 'imgUrl' not in self.df.columns: raise ValueError("DataFrame must contain 'imgUrl' column.")
        self.df['processed_text'] = self.df['title'].apply(self._clean_text)

    def _clean_text(self, text):
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = None
        try:
            response = requests.get(row['imgUrl'], timeout=10)
            response.raise_for_status()
            img_bytes = BytesIO(response.content)
            if img_bytes.getbuffer().nbytes > 0: img = Image.open(img_bytes).convert('RGB')
        except Exception: pass # Silently pass for now
        if img is None: img = Image.new('RGB', (224, 224), color='gray') # Default placeholder
        if self.transform: img = self.transform(img)
        return {'image': img, 'text': row['processed_text'], 'image_url': row['imgUrl'], 'original_text': row['title']}

    def collate_fn(self, batch):
        batch = [b for b in batch if b is not None]
        if not batch: return {'image': torch.empty(0, 3, 224, 224, device=device), 'text': [], 'image_url': [], 'original_text': []}
        images = torch.stack([item['image'] for item in batch]).to(device)
        return {'image': images, 'text': [item['text'] for item in batch], 'image_url': [item['image_url'] for item in batch], 'original_text': [item['original_text'] for item in batch]}


class SimpleIndexDataset(Dataset): # Used for building the initial index
    def __init__(self, df, transform):
        self.df, self.transform = df.copy(), transform
        if 'title' not in self.df.columns: self.df['title'] = "N/A" # Default title
        if 'imgUrl' not in self.df.columns: raise ValueError("'imgUrl' column missing in DataFrame for SimpleIndexDataset.")

    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]; img=None
        try:
            response = requests.get(row['imgUrl'], timeout=5)
            response.raise_for_status()
            img_bytes = BytesIO(res.content)
            if img_bytes.getbuffer().nbytes > 0:
                 img = Image.open(img_bytes).convert('RGB')
        except Exception: pass # Silently pass for now
        if img is None: img = Image.new('RGB', (224,224), 'lightgray') # Placeholder
        if self.transform: img = self.transform(img)
        # Return data needed for building the image index
        return {'image':img,'image_url':row['imgUrl'],'original_text':row.get('title','N/A')}

    def collate_fn(self, batch): # Custom collate for SimpleIndexDataset
        valid_items = [item for item in batch if item and item.get('image') is not None]
        if not valid_items:
            return {'image':torch.empty(0,3,224,224,device=device),'image_url':[],'original_text':[]}
        images = torch.stack([item['image'] for item in valid_items]).to(device)
        image_urls = [item['image_url'] for item in valid_items]
        original_texts = [item['original_text'] for item in valid_items]
        return {'image':images,'image_url':image_urls,'original_text':original_texts}