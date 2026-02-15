import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd

# ==========================================
# 1. PATH CONFIGURATION (Update for your PC)
# ==========================================
base_path = r'C:\Users\asjal\OneDrive\Documents\IBA Hackathon\Dataset' # Main dataset folder

paths = {
    "Training": os.path.join(base_path, 'Offroad_Segmentation_Training_Dataset', 'train', 'Segmentation'),
    "Validation": os.path.join(base_path, 'Offroad_Segmentation_Training_Dataset', 'val', 'Segmentation'),
    "Testing": os.path.join(base_path, 'test_public_80', 'Segmentation')
}

# Mapping and Names
value_map = {0: 0, 100: 1, 200: 2, 300: 3, 500: 4, 550: 5, 700: 6, 800: 7, 7100: 8, 10000: 9}
class_names = [
    "Background", "Trees", "Lush Bushes", "Dry Grass", "Dry Bushes",
    "Ground Clutter", "Logs", "Rocks", "Landscape", "Sky"
]

# ==========================================
# 2. ANALYSIS LOGIC
# ==========================================
def analyze_folder(folder_path, folder_name):
    if not os.path.exists(folder_path):
        return None
    
    files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    presence_count = np.zeros(10)
    pixel_count = np.zeros(10)
    
    print(f"🔍 Analyzing {folder_name} ({len(files)} files)...")
    for f in tqdm(files):
        mask = np.array(Image.open(os.path.join(folder_path, f)))
        # Map values to IDs
        mapped_mask = np.zeros_like(mask, dtype=np.uint8)
        for raw_v, cid in value_map.items():
            mapped_mask[mask == raw_v] = cid
            
        unique, counts = np.unique(mapped_mask, return_counts=True)
        for u, c in zip(unique, counts):
            if u < 10:
                presence_count[u] += 1
                pixel_count[u] += c
                
    results = []
    for i in range(10):
        results.append({
            "Class": class_names[i],
            f"{folder_name} Img %": (presence_count[i] / len(files)) * 100,
            f"{folder_name} Pixel Total": pixel_count[i]
        })
    return pd.DataFrame(results)

# ==========================================
# 3. RUN & COMPARE
# ==========================================
df_list = []
for name, path in paths.items():
    res = analyze_folder(path, name)
    if res is not None:
        df_list.append(res.set_index("Class"))

# Merge all results into one big table
if df_list:
    final_report = pd.concat(df_list, axis=1)
    print("\n" + "="*80)
    print("📊 CROSS-DATASET CLASS COMPARISON")
    print("="*80)
    print(final_report.to_string())
else:
    print("❌ Error: Could not find dataset folders. Check your paths!")