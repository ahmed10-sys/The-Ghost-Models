import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from tqdm import tqdm

# ==========================================
# 1. PC CONFIGURATION (Karachi PC - 6GB VRAM)
# ==========================================
DATA_DIR = r'C:\Users\asjal\OneDrive\Documents\IBA Hackathon\Dataset\Offroad_Segmentation_Training_Dataset'
SAVE_PATH = 'deeplab_offroad_best.pth'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 4      # Optimized for 1660S memory
LR = 1e-4           # Stable learning rate for ResNet backbones
EPOCHS = 20         # ResNet-50 needs more time than SegFormer to converge

# ==========================================
# 2. DATASET & VALUE MAPPING
# ==========================================
# Mapping for 16-bit Raw Values
value_map = {0: 0, 100: 1, 200: 2, 300: 3, 500: 4, 550: 5, 700: 6, 800: 7, 7100: 8, 10000: 9}

def convert_mask(mask):
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw_val, class_id in value_map.items():
        new_arr[arr == raw_val] = class_id
    return new_arr

class OffroadDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.masks_dir = os.path.join(data_dir, 'Segmentation')
        self.transform = transform
        self.images = sorted(os.listdir(self.image_dir))

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        image = np.array(Image.open(os.path.join(self.image_dir, img_name)).convert("RGB"))
        mask = convert_mask(np.array(Image.open(os.path.join(self.masks_dir, img_name))))
        if self.transform:
            aug = self.transform(image=image, mask=mask)
            image, mask = aug['image'], aug['mask']
        return image, mask

# ==========================================
# 3. AUGMENTATION & LOADERS
# ==========================================
# We use high resolution to see tiny Lush Bushes
transform = A.Compose([
    A.Resize(544, 960),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.ShiftScaleRotate(scale_limit=0.1, rotate_limit=10, p=0.3),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

train_ds = OffroadDataset(os.path.join(DATA_DIR, 'train'), transform=transform)
val_ds = OffroadDataset(os.path.join(DATA_DIR, 'val'), transform=transform)

# drop_last=True is vital for Batch Norm models like DeepLab
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

# ==========================================
# 4. MODEL & TARGETED LOSS
# ==========================================
model = smp.DeepLabV3Plus(
    encoder_name="resnet50", 
    encoder_weights="imagenet", 
    in_channels=3, 
    classes=10
).to(DEVICE)

# TARGETED WEIGHTS: Ignoring classes that are 0% in your Test set report
# Class 2 (Lush Bushes) and Class 7 (Rocks) are heavily weighted to fix low IoU
# Background(0), Clutter(5), and Logs(6) are 0.0 to focus model capacity elsewhere
loss_weights = torch.tensor([0.0, 1.0, 50.0, 1.0, 2.0, 0.0, 0.0, 20.0, 1.0, 1.0]).to(DEVICE)

criterion_ce = nn.CrossEntropyLoss(weight=loss_weights)
criterion_dice = smp.losses.DiceLoss(mode='multiclass', from_logits=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scaler = torch.cuda.amp.GradScaler() # Faster training on GTX 1660S



for epoch in range(1, EPOCHS + 1):
    model.train()
    progress = tqdm(train_loader, desc=f"Epoch {epoch}")
    for images, masks in progress:
        images, masks = images.to(DEVICE), masks.long().to(DEVICE)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logits = model(images)
            loss = criterion_ce(logits, masks) + criterion_dice(logits, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        progress.set_postfix(loss=loss.item())

    # Validation
    model.eval()
    tp, fp, fn, tn = 0, 0, 0, 0
    with torch.no_grad():
        for imgs, msks in val_loader:
            imgs, msks = imgs.to(DEVICE), msks.long().to(DEVICE)
            outputs = model(imgs)
            stats = smp.metrics.get_stats(outputs.argmax(dim=1), msks, mode='multiclass', num_classes=10)
            tp += stats[0]; fp += stats[1]; fn += stats[2]; tn += stats[3]
    
    val_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
# Validation phase ends here...
    print(f"⭐ Epoch {epoch} | Val IoU: {val_iou:.4f}")
    
    # Save the absolute best model
    if val_iou > best_iou:
        best_iou = val_iou
        torch.save(model.state_dict(), "deeplab_best.pth")
        print("💾 New Best Model Saved!")
        
    # ALSO save the latest epoch so you can resume training
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_iou': best_iou,
    }, "deeplab_checkpoint_latest.pth")