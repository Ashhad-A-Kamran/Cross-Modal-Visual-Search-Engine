import torch
import torch.nn as nn

from config import device, MODEL_PATH, TRAIN_EPOCHS, TRAIN_LR # Import necessary configs

class LostFoundSystem(nn.Module):
    def __init__(self, image_encoder, text_encoder):
        super().__init__(); self.image_encoder, self.text_encoder = image_encoder, text_encoder
        self.temperature = nn.Parameter(torch.ones([]) * 0.07) # Learnable temperature
    def forward(self, images, texts): return self.image_encoder(images), self.text_encoder(texts)

def minimal_train_model(model_to_train, train_loader, num_epochs=TRAIN_EPOCHS, model_save_path=MODEL_PATH, progress_callback=None):
    print(f"DEBUG: minimal_train_model called. Training for {num_epochs} epochs. Will save to {model_save_path}")
    model_to_train.to(device); criterion = nn.CrossEntropyLoss(); optimizer = torch.optim.Adam(model_to_train.parameters(), lr=TRAIN_LR)
    for epoch in range(num_epochs):
        model_to_train.train()
        total_loss = 0; num_batches = 0
        for batch_idx, batch in enumerate(train_loader):
            if not batch or batch['image'].numel()==0:
                print(f"DEBUG: Skipping empty batch in training epoch {epoch+1}, batch {batch_idx}")
                continue
            img, txt = batch['image'].to(device), batch['text']
            img_embeds, text_embeds = model_to_train(img,txt)

            if img_embeds.numel()==0 or text_embeds.numel()==0:
                print(f"DEBUG: Skipping batch due to empty embeddings in training epoch {epoch+1}, batch {batch_idx}")
                continue
            if img_embeds.shape[0] != text_embeds.shape[0]:
                print(f"DEBUG: Mismatch in embedding batch sizes: Img {img_embeds.shape[0]}, Txt {text_embeds.shape[0]}. Skipping batch.")
                continue
            if img_embeds.shape[0] == 0:
                continue

            logits = (img_embeds @ text_embeds.T) * torch.exp(model_to_train.temperature)
            labels = torch.arange(img_embeds.shape[0], device=device)
            loss = (criterion(logits,labels) + criterion(logits.T,labels))/2
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item(); num_batches += 1
        avg_loss = total_loss / num_batches if num_batches > 0 else float('nan')
        if progress_callback: progress_callback(epoch+1, num_epochs, avg_loss)
        print(f"DEBUG: Minimal train epoch {epoch+1}/{num_epochs} avg loss: {avg_loss:.4f}, Temperature: {model_to_train.temperature.item():.4f}")
    torch.save(model_to_train.state_dict(), model_save_path)
    print(f"DEBUG: Minimally trained model saved to {model_save_path}")
    return model_to_train