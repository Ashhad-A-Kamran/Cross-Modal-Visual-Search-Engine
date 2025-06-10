import matplotlib.pyplot as plt
import numpy as np

# Data for Learning Rate 0.0001 (from Image 1)
epochs_lr_0001 = np.arange(1, 11)
train_loss_lr_0001 = [2.6022, 2.3320, 2.2672, 2.2214, 2.1810, 2.1497, 2.1153, 2.0851, 2.0631, 2.0434]
val_loss_lr_0001 = [2.3931, 2.3106, 2.2917, 2.2873, 2.2864, 2.2840, 2.2861, 2.2865, 2.2858, 2.2841]
best_val_loss_lr_0001 = 2.2840
best_epoch_lr_0001 = 6

# Data for Learning Rate 0.001 (from Image 2)
epochs_lr_001 = np.arange(1, 11)
train_loss_lr_001 = [3.1347, 3.0178, 2.9734, 2.9335, 2.8962, 2.8552, 2.8260, 2.7802, 2.7405, 2.7036]
val_loss_lr_001 = [3.1315, 3.0569, 2.9996, 2.9708, 2.9663, 2.9287, 2.9221, 2.8956, 2.8914, 2.8901]
best_val_loss_lr_001 = 2.8901
best_epoch_lr_001 = 10

# Create the plot
plt.figure(figsize=(12, 7))

# Plotting for LR = 0.0001
plt.plot(epochs_lr_0001, train_loss_lr_0001, 'o-', label='Train Loss (LR=0.0001)', color='royalblue', alpha=0.8)
plt.plot(epochs_lr_0001, val_loss_lr_0001, 's--', label='Validation Loss (LR=0.0001)', color='skyblue', alpha=0.8)
plt.scatter(best_epoch_lr_0001, best_val_loss_lr_0001, marker='*', s=150, color='gold', edgecolor='black', zorder=5, label=f'Best Val Loss (LR=0.0001): {best_val_loss_lr_0001:.4f}')

# Plotting for LR = 0.001
plt.plot(epochs_lr_001, train_loss_lr_001, 'o-', label='Train Loss (LR=0.001)', color='firebrick', alpha=0.8)
plt.plot(epochs_lr_001, val_loss_lr_001, 's--', label='Validation Loss (LR=0.001)', color='salmon', alpha=0.8)
plt.scatter(best_epoch_lr_001, best_val_loss_lr_001, marker='*', s=150, color='orange', edgecolor='black', zorder=5, label=f'Best Val Loss (LR=0.001): {best_val_loss_lr_001:.4f}')


# Adding labels and title
plt.title('Training and Validation Loss Comparison for Different Learning Rates', fontsize=16)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.xticks(epochs_lr_0001) # Ensure all epoch numbers are shown
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# Show plot
plt.show()
