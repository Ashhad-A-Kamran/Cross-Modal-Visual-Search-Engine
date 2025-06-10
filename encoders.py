import torch
import torch.nn as nn
import torchvision
from sentence_transformers import SentenceTransformer # << NEW IMPORT

# from config import device, EMBED_SIZE, VOCAB_SIZE, MAX_TEXT_LEN # EMBED_SIZE might come from SBERT model
from config import device, EMBED_SIZE # Keep EMBED_SIZE for ImageEncoder for now, ensure consistency

class ImageEncoder(nn.Module):
  def __init__(self, embed_size=EMBED_SIZE): # Ensure this embed_size matches TextEncoder's output
    super().__init__()

    try:
        self.cnn = torch.hub.load('pytorch/vision', 'resnet18', weights='ResNet18_Weights.DEFAULT' if hasattr(torchvision.models, 'ResNet18_Weights') else 'resnet18', trust_repo=True)
    except Exception:
        self.cnn = torch.hub.load('pytorch/vision', 'resnet18', weights=None, trust_repo=True)
    
    # Check if embed_size needs to be adapted or if a projection layer is added
    # to match the TextEncoder's output dimension if they differ.
    self.cnn.fc = nn.Linear(self.cnn.fc.in_features, embed_size)

  def forward(self, images):
    current_device = images.device if images is not None and hasattr(images, 'device') else device
    if images is None or images.numel() == 0: return torch.empty((0, self.cnn.fc.out_features), device=current_device)
    return torch.nn.functional.normalize(self.cnn(images), p=2, dim=1)


class TextEncoder(nn.Module):
  def __init__(self, model_name='all-MiniLM-L6-v2'): # Example SBERT model
    super().__init__()
    self.sbert_model = SentenceTransformer(model_name, device=device) # Load SBERT model
    # The embedding dimension is now determined by the SBERT model
    self.embed_dim = self.sbert_model.get_sentence_embedding_dimension()
    print(f"DEBUG: TextEncoder initialized with {model_name}, embedding dimension: {self.embed_dim}")
    # Ensure EMBED_SIZE in config.py is set to this self.embed_dim for consistency with ImageEncoder


  def forward(self, texts):
    # SBERT expects a list of sentences (strings)
    if not texts or all(not t for t in texts):
        # Return an empty tensor with the correct embedding dimension
        return torch.empty((0, self.embed_dim), device=device) # Use self.sbert_model.device if model not explicitly moved to global device

    # Encode texts using SBERT. The model handles tokenization internally.
    # The output will be on the device the SBERT model is on.
    text_embeddings = self.sbert_model.encode(texts, convert_to_tensor=True, show_progress_bar=False)
    
    # Normalize embeddings
    text_embeddings = torch.nn.functional.normalize(text_embeddings, p=2, dim=1)
    return text_embeddings.to(device) # Ensure it's on the global device if needed elsewhere