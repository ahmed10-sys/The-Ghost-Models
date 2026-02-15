import torch
import numpy as np
import cv2
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from tqdm import tqdm
from PIL import Image

# ==========================================
# 1. CONFIGURATION
# ==========================================
DATA_DIR = r'C:\Users\asjal\OneDrive\Documents\IBA Hackathon\Dataset\test_public_80'
IMG_DIR = os.path.join(DATA_DIR, 'Color_Images')
MASK_DIR = os.path.join(DATA_DIR, 'Segmentation')

MODEL_10_PATH = 'deeplab_best.pth'          
MODEL_BIN_PATH = 'binary_obstacle_model.pth' 
MODEL_ROCK_PATH = 'rock_specialist.pth'      

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
value_map = {0: 0, 100: 1, 200: 2, 300: 3, 500: 4, 550: 5, 700: 6, 800: 7, 7100: 8, 10000: 9}

# ==========================================
# 2. UTILITY FUNCTIONS
# ==========================================
def convert_mask(mask):
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw_val, class_id in value_map.items():
        new_arr[arr == raw_val] = class_id
    return new_arr

def calculate_smart_iou(pred_mask, true_mask):
    ious = []
    valid_indices = [1, 2, 3, 4, 7, 8, 9] 
    for cls in valid_indices:
        p, t = (pred_mask == cls), (true_mask == cls)
        union = (p | t).sum()
        ious.append((p & t).sum() / union if union > 0 else np.nan)
    return np.array(ious)

# ==========================================
# 3. LOAD MODELS
# ==========================================
print("⚙️ Initializing Triple-Model TTA Pipeline...")

def load_model(arch, weights, classes):
    if arch == 'deeplab':
        m = smp.DeepLabV3Plus(encoder_name="resnet50", encoder_weights=None, in_channels=3, classes=classes).to(DEVICE)
    else:
        m = smp.Unet(encoder_name="resnet18", encoder_weights=None, in_channels=3, classes=classes).to(DEVICE)
    m.load_state_dict(torch.load(weights, map_location=DEVICE, weights_only=True))
    m.eval()
    return m

model_10 = load_model('deeplab', MODEL_10_PATH, 10)
model_bin = load_model('unet', MODEL_BIN_PATH, 1)
model_rock = load_model('unet', MODEL_ROCK_PATH, 1)

# ==========================================
# 4. TTA INFERENCE LOGIC
# ==========================================
def get_probs(input_tensor):
    with torch.no_grad():
        p10 = torch.softmax(model_10(input_tensor), dim=1)
        pb = torch.sigmoid(model_bin(input_tensor))
        pr = torch.sigmoid(model_rock(input_tensor))
    return p10, pb, pr

def inference_step(image_rgb):
    H, W = image_rgb.shape[:2]
    transform = A.Compose([
        A.Resize(544, 960), 
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    # --- TTA 1: Original Image ---
    img1 = transform(image=image_rgb)['image'].unsqueeze(0).to(DEVICE)
    p10_1, pb_1, pr_1 = get_probs(img1)
    
    # --- TTA 2: Horizontal Flip ---
    img_flipped = cv2.flip(image_rgb, 1)
    img2 = transform(image=img_flipped)['image'].unsqueeze(0).to(DEVICE)
    p10_2, pb_2, pr_2 = get_probs(img2)
    
    # Flip predictions back to align with original [cite: 172]
    p10_2 = torch.flip(p10_2, [3])
    pb_2 = torch.flip(pb_2, [3])
    pr_2 = torch.flip(pr_2, [3])
    
    # Ensemble: Average Probabilities [cite: 38]
    p10 = (p10_1 + p10_2) / 2.0
    pb = (pb_1 + pb_2) / 2.0
    pr = (pr_1 + pr_2) / 2.0
    pb = pb.squeeze(1)
    pr = pr.squeeze(1)

    # --- HIERARCHICAL DUALITY LOGIC ---
    # Rock Specialist Correction [cite: 17, 171]
    is_rock_high_conf = (pr > 0.85).float()
    p10[:, 7, :, :] += (pr * 0.5) 
    
    # Obstacle Suppression for Landscape (8) and Sky (9) [cite: 19, 172]
    is_obstacle = (pb > 0.8).float()
    p10[:, [8, 9], :, :] *= (1.0 - (is_obstacle * 0.4))
    p10[:, 8, :, :] *= (1.0 - (is_rock_high_conf * 0.4))
    
    # Class Remapping for Competition [cite: 11]
    p10[:, 8, :, :] += p10[:, 0, :, :]
    p10[:, 0, :, :] = 0
    
    pred_mask = torch.argmax(p10, dim=1).cpu().numpy()[0]
    return cv2.resize(pred_mask.astype('uint8'), (W, H), interpolation=cv2.INTER_NEAREST)

# ==========================================
# 5. EXECUTION
# ==========================================
if not os.path.exists(IMG_DIR):
    print(f"❌ Path not found: {IMG_DIR}")
else:
    images = sorted([f for f in os.listdir(IMG_DIR) if f.endswith('.png')])
    total_ious = []
    print(f"🚀 Running TTA Evaluation on {len(images)} images...")

    for img_name in tqdm(images):
        img_path, mask_path = os.path.join(IMG_DIR, img_name), os.path.join(MASK_DIR, img_name)
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        pred_mask = inference_step(image)
        
        if os.path.exists(mask_path):
            gt_mask = convert_mask(np.array(Image.open(mask_path)))
            total_ious.append(calculate_smart_iou(pred_mask, gt_mask))

    all_ious = np.array(total_ious)
    mean_class_iou = np.nanmean(all_ious, axis=0)
    valid_names = ["Trees", "Lush Bushes", "Dry Grass", "Dry Bushes", "Rocks", "Landscape", "Sky"]

    print("\n" + "="*40 + "\n📊 FINAL TTA RESULTS\n" + "="*40)
    for name, iou in zip(valid_names, mean_class_iou):
        print(f"{name:<20}: {iou:.4f}")
    print("-" * 40 + f"\n🏆 FINAL ADJUSTED mIoU: {np.nanmean(mean_class_iou):.4f}")
