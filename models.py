
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import requests
from PIL import Image, UnidentifiedImageError
from io import BytesIO
import re
from sklearn.model_selection import train_test_split



device = torch.device('cpu')
MODEL_PATH = "trained_model.pth"



# --- ML Code: Dataset, Encoders, System, Minimal Training ---
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
        except Exception: pass
        if img is None: img = Image.new('RGB', (224, 224), color='gray')
        if self.transform: img = self.transform(img)
        return {'image': img, 'text': row['processed_text'], 'image_url': row['imgUrl'], 'original_text': row['title']}

    def collate_fn(self, batch):
        batch = [b for b in batch if b is not None]
        if not batch: return {'image': torch.empty(0, 3, 224, 224, device=device), 'text': [], 'image_url': [], 'original_text': []}
        images = torch.stack([item['image'] for item in batch]).to(device)
        return {'image': images, 'text': [item['text'] for item in batch], 'image_url': [item['image_url'] for item in batch], 'original_text': [item['original_text'] for item in batch]}

class ImageEncoder(nn.Module):
  def __init__(self, embed_size=256):
    super().__init__()
    try: self.cnn = torch.hub.load('pytorch/vision', 'resnet18', weights='ResNet18_Weights.DEFAULT' if hasattr(torchvision.models, 'ResNet18_Weights') else 'resnet18')
    except Exception: self.cnn = torch.hub.load('pytorch/vision', 'resnet18', weights=None) 
    self.cnn.fc = nn.Linear(self.cnn.fc.in_features, embed_size)
  def forward(self, images):
    if images is None or images.numel() == 0: return torch.empty((0, self.cnn.fc.out_features), device=device)
    return torch.nn.functional.normalize(self.cnn(images), p=2, dim=1)

class TextEncoder(nn.Module):
  def __init__(self, embed_size=256, vocab_size=10000, max_len=32):
    super().__init__()
    self.embedding = nn.Embedding(vocab_size, embed_size)
    self.rnn = nn.GRU(embed_size, embed_size, batch_first=True)
    self.vocab_size, self.max_len = vocab_size, max_len
  def forward(self, texts):
    dev = self.embedding.weight.device
    if not texts or all(not t for t in texts): return torch.empty((0, self.rnn.hidden_size), device=dev)
    
    processed_texts = []
    for t in texts:
        if t:
            word_ids = [hash(w) % self.vocab_size for w in str(t).split()[:self.max_len]]
            padded_word_ids = word_ids + [0] * (self.max_len - len(word_ids)) 
        else: 
            padded_word_ids = [0] * self.max_len
        processed_texts.append(padded_word_ids)

    ids = torch.tensor(processed_texts, dtype=torch.long).to(dev)
    _, h = self.rnn(self.embedding(ids))
    return torch.nn.functional.normalize(h[-1], p=2, dim=1)

class LostFoundSystem(nn.Module):
    def __init__(self, image_encoder, text_encoder):
        super().__init__(); self.image_encoder, self.text_encoder = image_encoder, text_encoder
        self.temperature = nn.Parameter(torch.ones([]) * 0.07)
    def forward(self, images, texts): return self.image_encoder(images), self.text_encoder(texts)

def minimal_train_model(model_to_train, train_loader, num_epochs=1, model_save_path="trained_model.pth", progress_callback=None):
    print(f"DEBUG: minimal_train_model called. Will save to {model_save_path}")
    model_to_train.to(device); criterion = nn.CrossEntropyLoss(); optimizer = torch.optim.Adam(model_to_train.parameters(), lr=0.0001)
    for epoch in range(num_epochs):
        model_to_train.train()
        total_loss = 0; num_batches = 0
        for batch in train_loader:
            if not batch or batch['image'].numel()==0: continue
            img, txt = batch['image'].to(device), batch['text']
            opt_img, opt_txt = model_to_train(img,txt)
            if opt_img.numel()==0 or opt_txt.numel()==0: continue
            logits = (opt_img @ opt_txt.T) * torch.exp(model_to_train.temperature)
            labels = torch.arange(len(img),device=device)
            loss = (criterion(logits,labels) + criterion(logits.T,labels))/2
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item(); num_batches += 1
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        if progress_callback: progress_callback(epoch+1, num_epochs, avg_loss)
        print(f"DEBUG: Minimal train epoch {epoch+1} avg loss: {avg_loss}")
    torch.save(model_to_train.state_dict(), model_save_path)
    print(f"DEBUG: Minimally trained model saved to {model_save_path}")
    return model_to_train

class LostFoundApp:
    def __init__(self, model, initial_dataset=None):
        self.model = model.to(device); self.device = device
        encoder_output_size = model.image_encoder.cnn.fc.out_features if hasattr(model.image_encoder.cnn, 'fc') else 256
        self.image_features = torch.empty((0, encoder_output_size), device=self.device)
        self.image_urls, self.original_texts = [], []
        self.item_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])
        if initial_dataset and len(initial_dataset) > 0: 
            print(f"DEBUG: LostFoundApp init, building index with {len(initial_dataset)} items.")
            self._build_index(initial_dataset)
        else:
            print("DEBUG: LostFoundApp init, no initial dataset to build index.")

    def _build_index(self, dataset, progress_callback=None):
        print(f"DEBUG: LostFoundApp._build_index with {len(dataset)} items.")
        loader_collate_fn = dataset.collate_fn if hasattr(dataset, 'collate_fn') else None
        loader = DataLoader(dataset, batch_size=16, collate_fn=loader_collate_fn)
        
        self.model.eval(); temp_features = []
        with torch.no_grad():
            for i, batch in enumerate(loader):
                if not batch or batch['image'].numel()==0: continue
                features = self.model.image_encoder(batch['image'].to(self.device))
                if features.numel()>0:
                    temp_features.append(features)
                    self.image_urls.extend(batch['image_url'])
                    self.original_texts.extend(batch.get('original_text', ['N/A']*len(batch['image_url'])))
                if progress_callback: progress_callback(i+1, len(loader))
        if temp_features: self.image_features = torch.cat(temp_features)
        print(f"DEBUG: Index built. Features shape: {self.image_features.shape}, URLs: {len(self.image_urls)}")

    def add_new_item(self, image_url):
        self.model.eval()
        try:
            res = requests.get(image_url, timeout=10); res.raise_for_status()
            img_bytes = BytesIO(res.content)
            if not img_bytes.getbuffer().nbytes > 0: return False, f"Empty image from {image_url}"
            img = Image.open(img_bytes).convert('RGB')
            img_tensor = self.item_transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad(): new_feat = self.model.image_encoder(img_tensor)
            self.image_features = torch.cat([self.image_features, new_feat])
            self.image_urls.append(image_url); self.original_texts.append("N/A (Newly Added)")
            print(f"DEBUG: Item added. Index size: {len(self.image_urls)}")
            return True, f"Item added. Index size: {len(self.image_urls)}"
        except Exception as e: 
            print(f"DEBUG: Error adding item {image_url}: {e}")
            return False, f"Error: {e}"

    def search_lost_item(self, description, top_k=5):
        self.model.eval()
        print(f"DEBUG: Searching for '{description}', top_k={top_k}. Index size: {self.image_features.shape[0]}")
        if self.image_features.numel() == 0: 
            print("DEBUG: Search failed: Index is empty.")
            return [], "Index is empty."
        with torch.no_grad(): text_feat = self.model.text_encoder([description])
        if text_feat.numel() == 0: 
            print("DEBUG: Search failed: Failed to encode description.")
            return [], "Failed to encode description."
        
        scores = text_feat @ self.image_features.T
        if scores.numel() == 0: 
            print("DEBUG: Search failed: No scores generated.")
            return [], "No scores generated."
        
        k = min(top_k, self.image_features.shape[0])
        if k == 0: 
            print("DEBUG: Search failed: No items to rank (k=0).")
            return [], "No items to rank."
            
        top_s, top_i = torch.topk(scores.squeeze(), k)
        results = [{'image_url': self.image_urls[i.item()],'score':s.item(),'original_text':self.original_texts[i.item()]} for s,i in zip(top_s,top_i)]
        message = "Search complete." if results else "No matches found."
        print(f"DEBUG: Search results: {len(results)} items. Message: {message}")
        return results, message

class SimpleIndexDataset(Dataset): 
    def __init__(self, df, transform):
        self.df, self.transform = df.copy(), transform
        if 'title' not in self.df.columns: self.df['title'] = "N/A"
        if 'imgUrl' not in self.df.columns: raise ValueError("'imgUrl' missing.")

    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]; img=None
        try:
            res = requests.get(row['imgUrl'], timeout=5); res.raise_for_status()
            img_bytes = BytesIO(res.content) 
            if img_bytes.getbuffer().nbytes > 0: 
                 img = Image.open(img_bytes).convert('RGB')
        except: pass 
        if img is None: img = Image.new('RGB', (224,224), 'lightgray')
        if self.transform: img = self.transform(img)
        return {'image':img,'image_url':row['imgUrl'],'original_text':row.get('title','N/A')}
        
    def collate_fn(self, batch):
        valid = [i for i in batch if i and i.get('image') is not None]
        if not valid: return {'image':torch.empty(0,3,224,224,device=device),'image_url':[],'original_text':[]}
        return {'image':torch.stack([i['image'] for i in valid]).to(device),'image_url':[i['image_url'] for i in valid],'original_text':[i['original_text'] for i in valid]}
