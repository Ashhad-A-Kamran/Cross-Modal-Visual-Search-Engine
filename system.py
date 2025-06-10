import os
import torch
import torch.nn as nn
import numpy as np # For checking NaN/inf if needed

# from config import device, MODEL_PATH, TRAIN_EPOCHS, TRAIN_LR # these will be passed as args
from config import device, EARLY_STOPPING_PATIENCE # Import specific needed configs

class LostFoundSystem(nn.Module):
    def __init__(self, image_encoder, text_encoder):
        super().__init__()
        self.image_encoder, self.text_encoder = image_encoder, text_encoder
        self.temperature = nn.Parameter(torch.ones([]) * 0.07) # Learnable temperature
    def forward(self, images, texts):
        return self.image_encoder(images), self.text_encoder(texts)

# Renamed and enhanced from minimal_train_model
def train_model(model_to_train, train_loader, val_loader, num_epochs, model_save_path, best_model_save_path, learning_rate, progress_callback=None):
    print(f"DEBUG: train_model called. Training for {num_epochs} epochs.")
    print(f"Will save final model to {model_save_path} and best model to {best_model_save_path}")

    model_to_train.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_to_train.parameters(), lr=learning_rate)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    history = {'train_loss': [], 'val_loss': []} # To store loss history

    for epoch in range(num_epochs):
        # --- Training Phase ---
        model_to_train.train()
        total_train_loss = 0
        num_train_batches = 0
        for batch_idx, batch in enumerate(train_loader):
            # Your existing batch skipping logic can remain
            if not batch or ('image' not in batch) or ('text' not in batch) or batch['image'].numel() == 0:
                print(f"DEBUG: Skipping empty/invalid batch in training epoch {epoch+1}, batch {batch_idx}")
                continue

            img, txt = batch['image'].to(device), batch['text']
            img_embeds, text_embeds = model_to_train(img, txt)

            if img_embeds.numel() == 0 or text_embeds.numel() == 0:
                print(f"DEBUG: Skipping batch due to empty embeddings in training epoch {epoch+1}, batch {batch_idx}")
                continue
            if img_embeds.shape[0] != text_embeds.shape[0] or img_embeds.shape[0] == 0:
                print(f"DEBUG: Mismatch or zero in embedding batch sizes. Img {img_embeds.shape[0]}, Txt {text_embeds.shape[0]}. Skipping batch.")
                continue

            logits = (img_embeds @ text_embeds.T) * torch.exp(model_to_train.temperature)
            labels = torch.arange(img_embeds.shape[0], device=device)
            loss = (criterion(logits, labels) + criterion(logits.T, labels)) / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            num_train_batches += 1
        
        avg_train_loss = total_train_loss / num_train_batches if num_train_batches > 0 else float('nan')
        history['train_loss'].append(avg_train_loss)

        # --- Validation Phase ---
        model_to_train.eval()
        total_val_loss = 0
        num_val_batches = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if not batch or ('image' not in batch) or ('text' not in batch) or batch['image'].numel() == 0:
                    print(f"DEBUG: Skipping empty/invalid batch in validation epoch {epoch+1}, batch {batch_idx}")
                    continue
                
                img, txt = batch['image'].to(device), batch['text']
                img_embeds, text_embeds = model_to_train(img, txt)

                if img_embeds.numel() == 0 or text_embeds.numel() == 0: continue
                if img_embeds.shape[0] != text_embeds.shape[0] or img_embeds.shape[0] == 0: continue

                logits = (img_embeds @ text_embeds.T) * torch.exp(model_to_train.temperature)
                labels = torch.arange(img_embeds.shape[0], device=device)
                loss = (criterion(logits, labels) + criterion(logits.T, labels)) / 2
                
                total_val_loss += loss.item()
                num_val_batches += 1

        avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else float('nan')
        history['val_loss'].append(avg_val_loss)

        if progress_callback:
            progress_callback(epoch + 1, num_epochs, avg_train_loss, avg_val_loss) # Pass val_loss too
        
        print(f"Epoch {epoch+1}/{num_epochs} -> Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Temp: {model_to_train.temperature.item():.4f}")

        # --- Early Stopping & Best Model Saving ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model_to_train.state_dict(), best_model_save_path)
            print(f"DEBUG: New best model saved to {best_model_save_path} (Val Loss: {avg_val_loss:.4f})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"DEBUG: Validation loss did not improve for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"DEBUG: Early stopping triggered after {epoch+1} epochs.")
            break
            
    # Save the model from the last epoch
    torch.save(model_to_train.state_dict(), model_save_path)
    print(f"DEBUG: Final model from last epoch saved to {model_save_path}")
    
    # Here you could plot the loss from 'history' dict if in a Jupyter notebook, or save to CSV
    # e.g., pd.DataFrame(history).to_csv("training_history.csv")

    # Load the best model to return if it exists and performed better
    if os.path.exists(best_model_save_path) and best_val_loss < avg_val_loss : # check if best was actually better than last
        print(f"DEBUG: Loading best model from {best_model_save_path} to return.")
        model_to_train.load_state_dict(torch.load(best_model_save_path, map_location=device))

    return model_to_train