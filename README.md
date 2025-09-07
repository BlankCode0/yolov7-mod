# YOLOv7-Mod for Vehicle Detection

This repository contains an experimental modification of **YOLOv7** based on the paper *"Object Detection in Dense and Mixed Traffic for Autonomous Vehicles with Modified YOLO"*.  
The goal is to improve detection of **small and partially occluded vehicles** compared to the baseline YOLOv7.

---

## 🚀 Setup

```bash
# Install PyTorch with CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Verify GPU
python -c "import torch; print(torch.cuda.is_available())"

# Install dependencies
pip install matplotlib opencv-python Pillow PyYAML requests scipy tqdm tensorboard pandas seaborn ipython psutil thop
```

Verify GPU:

python -c "import torch; print(torch.cuda.is_available())"


Install other dependencies:

pip install matplotlib opencv-python Pillow PyYAML requests scipy tqdm tensorboard pandas seaborn ipython psutil thop


Clone this repository:

git clone https://github.com/Arunkj203/yolov7-mod.git
cd yolov7-mod

📂 Dataset

We used the Kaggle Vehicle Detection dataset (YOLO version).

Download in Colab:

from google.colab import files
import os, shutil

# Upload your kaggle.json from Kaggle account settings
uploaded = files.upload()
if 'kaggle.json' in uploaded:
    os.makedirs('/root/.kaggle', exist_ok=True)
    shutil.move('kaggle.json', '/root/.kaggle/kaggle.json')
    os.chmod('/root/.kaggle/kaggle.json', 0o600)

!pip install -q kaggle
!kaggle datasets download -d muhammadhananasghar/vehicle-detectionyolo-version -p /content/vehicle_ds --unzip

Creating an 800-image subset

A script was used to randomly sample 800 labeled images from the dataset:

Train: 640 images

Val: 160 images

Dataset structure after sampling:

vehicle_ds_800/
 ├── images/
 │    ├── train/
 │    └── val/
 └── labels/
      ├── train/
      └── val/


Custom YAML for YOLOv7:

train: /content/vehicle_ds_800/images/train
val:   /content/vehicle_ds_800/images/val

nc: 1
names: ['car']

🏋️ Training

Download official YOLOv7 pretrained weights:

wget "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt"


Train the baseline YOLOv7 and the modified YOLOv7:

# Baseline YOLOv7
python train.py --batch 8 --cfg cfg/training/yolov7.yaml \
  --epochs 30 --data data/custom.yaml \
  --weights /content/yolov7.pt --device 0 --name baseline_800

# Modified YOLOv7
python train.py --batch 8 --cfg cfg/training/yolov7-mod.yaml \
  --epochs 30 --data data/custom.yaml \
  --weights /content/yolov7.pt --device 0 --name mod_800

🔍 Inference

Run detection on a sample image:

python detect.py \
  --weights runs/train/mod_800/weights/best.pt \
  --img-size 640 --conf 0.36 \
  --source /content/vehicle_ds_800/images/val/10.jpg \
  --no-trace


Outputs are saved under runs/detect/.

## 📊 Results

- Trained on a custom subset of **800 images** (640 train / 160 val).  
- Training completed for ~100 epochs total.  
- **F1 Score**: **0.67** at confidence threshold **0.36**.  
- **Confusion Matrix**: ~71% accuracy for car detection.  
- Validation results show the model often detects **more cars than ground-truth labels**, indicating improved sensitivity to **small and partially occluded vehicles**.

### Visualizations

📈 **F1 Score Curve**  
![F1 Score Curve](results/f1_curve.png)

🔲 **Confusion Matrix**  
![Confusion Matrix](results/confusion_matrix.png)

🖼️ **Example Validation Batches**  
- Ground Truth  
  ![Ground Truth](results/test_batch_labels.jpg)  

- Predictions  
  ![Predictions](results/test_batch_pred.jpg)  

## ✅ Conclusion

The **YOLOv7-Mod** demonstrates improved ability to detect **small and partially occluded vehicles** compared to the baseline YOLOv7.

**Key takeaways:**
- Detects more vehicles in crowded and occluded scenes than baseline.  
- Achieved an **F1 score of 0.67** at confidence threshold **0.36**.  
- Confusion matrix shows ~71% accuracy for car detection.  
- Prediction results (`test_batch_pred.jpg`) often show more cars detected than the ground-truth labels (`test_batch_labels.jpg`), highlighting improved sensitivity.

**Future work:**
- Train on larger, occlusion-heavy datasets (e.g., **UA-DETRAC**, **BDD100K**).  
- Tune anchors and hyperparameters for better small-object detection.  
- Evaluate across diverse **weather** and **lighting conditions** for real-world robustness.  
