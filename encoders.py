import torch
import torch.nn as nn
import torchvision

from config import device, EMBED_SIZE, VOCAB_SIZE, MAX_TEXT_LEN
class ImageEncoder(nn.Module):
  def __init__(self, embed_size=EMBED_SIZE):
    super().__init__()
    try: self.cnn = torch.hub.load('pytorch/vision', 'resnet18', weights='ResNet18_Weights.DEFAULT' if hasattr(torchvision.models, 'ResNet18_Weights') else 'resnet18', trust_repo=True)
    except Exception: self.cnn = torch.hub.load('pytorch/vision', 'resnet18', weights=None, trust_repo=True)
    self.cnn.fc = nn.Linear(self.cnn.fc.in_features, embed_size)
  def forward(self, images):
    # Ensure device consistency, use images.device if available, otherwise fallback to global device
    current_device = images.device if images is not None and hasattr(images, 'device') else device
    if images is None or images.numel() == 0: return torch.empty((0, self.cnn.fc.out_features), device=current_device)
    return torch.nn.functional.normalize(self.cnn(images), p=2, dim=1)

class TextEncoder(nn.Module):
  def __init__(self, embed_size=EMBED_SIZE, vocab_size=VOCAB_SIZE, max_len=MAX_TEXT_LEN): # Use configs
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
    embedded = self.embedding(ids)
    _, h = self.rnn(embedded)
    text_embeddings = torch.nn.functional.normalize(h[-1], p=2, dim=1)
    return text_embeddings