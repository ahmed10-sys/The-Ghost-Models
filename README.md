The Ghost Models - Offroad Autonomy Segmentation
This repository contains the final submission for the IBA Hackathon (Duality AI) by The Ghost Models. Our solution utilizes a Triple-Model Hierarchical Pipeline with Multi-Scale Test Time Augmentation (TTA) to achieve robust semantic segmentation in novel desert environments.

🛑 Action Required: Download Model Weights
Due to the high precision and size of our models, the .pth files exceed GitHub's 100MB limit.

[CLICK HERE TO DOWNLOAD MODELS](https://drive.google.com/drive/u/1/folders/1IQKnN4C5SCem53HjAhejYVagHg1rohSa)

Placement: Move the downloaded .pth files directly into the root directory of this project folder.

📊 Final Results
Adjusted mIoU: 0.3681

Inference Speed: ~50-55ms per image (Optimized for real-world deployment)

Key Innovation: Specialized Rock-Detection Micro-Model

🚀 Quick Start
1. Setup Environment
Ensure you have the competition environment ready:

Bash
conda activate EDU
2. Clone the Repository
Bash
git clone https://github.com/ahmed10-sys/The-Ghost-Models.git
cd The-Ghost-Models
🛠 Running the Project
Scoring & Evaluation
To reproduce our final leaderboard results and generate the Per-Class IoU breakdown, run the main testing script:

Bash
python test.py
This script applies our hierarchical logic and Pyramid TTA to the test_public_80 dataset.

Interactive Frontend
We have developed a frontend application to visualize the model's "Safe Path" planning in real-time:

Bash
python app.py
🧠 Methodology
Our pipeline consists of three specialized "expert" models:

General Classifier (DeepLabV3+): Handles large-scale context like Sky and Landscape.

Binary Gatekeeper (Unet-ResNet18): Identifies general obstacle zones for UGV safety.

Rock Specialist (Unet-ResNet18): Specifically optimized to detect high-density scattered rocks, solving the primary failure case of standard models.

📂 Package Contents
test.py: Core scoring script with TTA and Hierarchical logic.

app.py: Interactive visualization tool for the segmentation output.

Report.pdf: Detailed 8-page analysis of methodology and failure cases.

requirements.txt: List of specific dependencies for reproducibility.
