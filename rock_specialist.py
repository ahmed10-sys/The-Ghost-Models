import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

# ==========================================
# 1. PC CONFIGURATION
# ==========================================
# Use the directories you have on your local machine
DATA_DIR = r'C:\Users\asjal\OneDrive\Documents\IBA Hackathon\dataset\Offroad_Segmentation_Training_Dataset'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 8  # Reduced for local GPU VRAM
EPOCHS = 5      # Short run to test the density theory

# ==========================================
# 2. ROCK-ONLY DATASET LOGIC
# ==========================================
class RockDataset(Dataset):
    def __init__(self, data_dir, subset='train', transform=None):
        self.img_dir = os.path.join(data_dir, subset, 'Color_Images')
        self.mask_dir = os.path.join(data_dir, subset, 'Segmentation')
        self.images = sorted([f for f in os.listdir(self.img_dir) if f.endswith('.png')])
        self.transform = transform

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])
        
        image = np.array(Image.open(img_path).convert("RGB"))
        # Focus strictly on Class ID 800 (Rocks) 
        full_mask = np.array(Image.open(mask_path))
        rock_mask = (full_mask == 800).astype(np.float32)
        
        if self.transform:
            augmented = self.transform(image=image, mask=rock_mask)
            image, rock_mask = augmented['image'], augmented['mask']
            
        return image, rock_mask.unsqueeze(0)

# ==========================================
# 3. TRAINING UTILS
# ==========================================
transform = A.Compose([
    A.Resize(320, 640), # Lower res for fast PC training
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2), # Sim-to-Real augmentation
    A.Normalize(),
    ToTensorV2(),
])

train_loader = DataLoader(RockDataset(DATA_DIR, 'train', transform), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(RockDataset(DATA_DIR, 'val', transform), batch_size=BATCH_SIZE)

# Using ResNet-18 Unet for speed and texture sensitivity
model = smp.Unet(encoder_name="resnet18", encoder_weights="imagenet", in_channels=3, classes=1).to(DEVICE)

# DICE Loss is excellent for scattered objects like rocks
criterion = smp.losses.DiceLoss(mode='binary')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ==========================================
# 4. PC TRAINING LOOP
# ==========================================
print(f"🚀 Starting Rock Specialist Training on {DEVICE}...")

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0
    for imgs, msks in tqdm(train_loader, desc=f"Epoch {epoch}"):
        imgs, msks = imgs.to(DEVICE), msks.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(imgs), msks)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Quick Validation
    model.eval()
    val_iou = 0
    with torch.no_grad():
        for imgs, msks in val_loader:
            imgs, msks = imgs.to(DEVICE), msks.to(DEVICE)
            preds = (torch.sigmoid(model(imgs)) > 0.5).float()
            inter = (preds * msks).sum()
            union = (preds + msks).sum() - inter
            val_iou += (inter + 1e-6) / (union + 1e-6)

    print(f"✅ Epoch {epoch} | Loss: {train_loss/len(train_loader):.4f} | Rock IoU: {val_iou/len(val_loader):.4f}")

# Save the specialist weights
torch.save(model.state_dict(), 'rock_specialist.pth')
print("💾 Specialist Model Saved!")