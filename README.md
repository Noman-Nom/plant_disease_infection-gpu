# GreenClinic — End-to-End Plant Disease AI (23 Classes, Explainable)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Practical, explainable plant disease diagnosis** — A state-of-the-art plant disease detection system using a 23-class EfficientNet-B0 classifier achieving **~99.6% test accuracy** + **weakly-supervised lesion maps** (Grad-CAM → U-Net). This project provides a complete pipeline from dataset hygiene to training, evaluation, visualization, and segmentation, making it ready for real-world deployment (ONNX/INT8 planned).

## 🌟 Key Features

- Advanced deep learning model for plant disease detection
- Support for 23 different plant diseases across various crops
- High accuracy and reliable predictions
- Real-time visualization of disease-affected areas
- Extensive data preprocessing and validation pipeline
- Model explainability through Grad-CAM
- Weakly-supervised segmentation for precise disease localization

---

<p align="center">
  <img src="docs/hero_confmat_placeholder.png" alt="Confusion Matrix" width="65%"/>
</p>

<p align="center">
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-project-structure">Structure</a> •
  <a href="#-dataset--preprocessing">Data</a> •
  <a href="#-classification-training">Training</a> •
  <a href="#-evaluation--results">Results</a> •
  <a href="#-explainability--segmentation">Explainability</a> •
  <a href="#-deployment-roadmap">Deployment</a>
</p>

---

## ✨ Highlights

- **Dataset:** 24,417 images across **23 classes** (`train: 17,082 | val: 3,651 | test: 3,684`)
- **Model:** `timm` **EfficientNet-B0** (224×224), class-weighted, mixed precision (AMP)
- **Hygiene:** Exact duplicate & cross-split leakage scan → **no pixel-identical overlap**
- **Accuracy:** **99.59%** (test, strict, no leakage)
- **Explainability:** **Grad-CAM** overlays for per-image disease evidence
- **Segmentation (weakly-supervised):** Grad-CAM pseudo-masks → **U-Net** trained, **Val IoU ≈ 0.563**
- **Reproducible paths:** Project root at `/home/muhammad-noman/projects/plant_disease-gpu-env`

---

## 🚀 Quick Start

```bash
# 1) Activate your environment
cd ~/projects/plant_disease-gpu-env
source .plants-gpu/bin/activate          # or: conda activate .plants-gpu

# 2) Install deps
pip install -r requirements.txt

# 3) Expected data layout (23 folders per split)
#   data/
#     train/<ClassName>/*.jpg
#     val/<ClassName>/*.jpg
#     test/<ClassName>/*.jpg

# 4) (Optional) Data sanity & class listing
python -m src.tools.check_data

# 5) Train classifier (EfficientNet-B0)
python -m src.train.classify

# 6) Evaluate on test set
python -m src.eval.classify_test


## 📁 Project Structure

```
plant_disease-gpu-env/
├─ data/
│  ├─ organized_dataset/          # Main dataset with 23 disease classes
│  ├─ train/                      # Training split (17,082 images)
│  ├─ val/                        # Validation split (3,651 images)
│  └─ test/                       # Test split (3,684 images)
├─ models/
│  ├─ best_model.pth             # Best performing classifier (EffNet-B0)
│  ├─ seg_unet_best.pth          # U-Net for lesion segmentation
│  ├─ class_names.json           # Disease class mapping
│  └─ training_history.csv       # Training metrics history
├─ outputs/
│  ├─ gradcam/                   # Visualization outputs
│  ├─ pseudo_lesion_masks_strict/# Generated lesion masks
│  ├─ quarantine_dups/           # Quarantined duplicate images
│  └─ test_classification_report.txt  # Detailed evaluation metrics
├─ notebooks/                    # Interactive development notebooks
├─ src/
│  ├─ datasets/                  # Data loading and preprocessing
│  └─ utils/                     # Utility functions and helpers
├─ requirements.txt              # Project dependencies
└─ README.md                     # Project documentation
```

## 🧼 Dataset & Preprocessing

### Dataset Overview
- **Total Images**: 24,417
- **Split Distribution**:
  - Training: 17,082 images
  - Validation: 3,651 images
  - Test: 3,684 images
- **Number of Classes**: 23

### Supported Plant Diseases
1. Apple: Apple Scab, Black Rot, Cedar Apple Rust, Healthy
2. Blueberry: Healthy
3. Cherry: Powdery Mildew, Healthy
4. Corn (Maize): Cercospora Leaf Spot, Common Rust, Northern Leaf Blight, Healthy
5. Grape: Black Rot, Esca (Black Measles), Leaf Blight, Healthy
6. Pepper: Healthy
7. Potato: Early Blight, Late Blight, Healthy
8. Tomato: Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites

### Preprocessing Pipeline
1. **Data Integrity**:
   - Verified class distribution across splits
   - Confirmed image format and corruption checks
   
2. **Duplicate Detection**:
   - Content-based image hashing
   - Cross-split leakage prevention
   - Within-split duplicate quarantine
   
3. **Image Transformations**:
   - Training: 
     - Resize to 224×224
     - Random flips and rotations
     - Color jittering
     - Affine transformations
     - ImageNet normalization
   - Validation/Test:
     - Resize to 224×224
     - ImageNet normalization

4. **Class Balance**:
   - Computed class weights for balanced training
   - Applied weighted sampling strategy


## 🧠 Model Architecture & Training

### Model Architecture
```
EfficientNet-B0
├─ Backbone: Pretrained ImageNet weights
├─ Custom Head:
│  ├─ Dropout(0.2)
│  ├─ Linear(1280 → 512)
│  ├─ ReLU
│  ├─ BatchNorm1d
│  ├─ Dropout(0.2)
│  └─ Linear(512 → 23)
```

### Training Configuration
- **Framework**: PyTorch with `timm` library
- **Hardware**: GPU-accelerated training
- **Optimizer**: AdamW
  - Learning rate: 1e-4
  - Weight decay: 1e-2
- **Loss Function**: CrossEntropyLoss with class weights
- **Learning Rate Scheduler**: ReduceLROnPlateau
  - Mode: max (validation accuracy)
  - Patience: 3
  - Factor: 0.1

### Training Details
- **Batch Size**: 32
- **GPU Optimization**:
  - Automatic Mixed Precision (AMP)
  - Pin memory enabled
  - Optimized num_workers for data loading
- **Early Stopping**:
  - Patience: 7 epochs
  - Monitor: Validation accuracy
  - Best model checkpoint saved
- **Reproducibility**:
  - Fixed random seeds (Python, NumPy, PyTorch)
  - Deterministic CUDA operations

## ✅ Model Performance & Results

### Classification Metrics
- **Validation Accuracy**: 99.45%
- **Test Accuracy**: 99.59%
- **Test Set Performance**:
  - Precision: > 99% for most classes
  - Recall: > 99% for most classes
  - F1-Score: > 99% for most classes

### Key Performance Indicators
- Robust against various lighting conditions
- High accuracy across all plant species
- Minimal false positives for healthy plants
- Reliable disease differentiation

### Output Artifacts
- **Model Checkpoints**: 
  - `models/best_model.pth`: Best performing model
  - `models/seg_unet_best.pth`: Segmentation model
- **Performance Reports**:
  - Detailed confusion matrix
  - Per-class accuracy metrics
  - Training history and learning curves
- **Validation Reports**:
  - Cross-validation results
  - Error analysis
  - Performance visualization


## 🔎 Explainability & Visualization

### Grad-CAM Visualization
Our model provides transparent decision-making through Grad-CAM (Gradient-weighted Class Activation Mapping):

1. **Implementation**:
   - Hooks into the final convolutional layer
   - Computes gradient-weighted feature maps
   - Generates class-specific activation heatmaps

2. **Visualization Features**:
   - Highlights disease-affected regions
   - Side-by-side comparison with original images
   - Color-coded intensity mapping
   - Interactive visualization notebooks

### Weakly-Supervised Segmentation
We implemented a U-Net based segmentation model trained on Grad-CAM generated pseudo-masks:

1. **Architecture**:
   - Compact U-Net design
   - Skip connections for fine detail preservation
   - Optimized for real-time inference

2. **Training Details**:
   - Loss: Combined Dice + Binary Cross Entropy
   - Automatic Mixed Precision training
   - Early stopping based on Validation IoU
   - Best Validation IoU: 0.563

3. **Applications**:
   - Precise disease localization
   - Affected area measurement
   - Severity assessment
   - Treatment planning support

### Output Directory Structure
```
outputs/
├─ gradcam/                 # Grad-CAM visualization results
├─ pseudo_lesion_masks_strict/  # Generated segmentation masks
└─ visualization_reports/   # Analysis and validation reports
```

## 📦 Deployment & Future Work

### Model Optimization
1. **Quantization**:
   - INT8 quantization for efficient deployment
   - Post-Training Static Quantization (PTSQ)
   - Calibration using fbgemm on x86
   - Optional Quantization-Aware Training (QAT)

2. **Model Export**:
   - ONNX format export (opset 12)
   - Support for both FP32 and INT8
   - Dynamic batch size handling
   - TensorRT optimization (planned)

### Performance Optimization
1. **Benchmarking**:
   - Comprehensive accuracy validation
   - CPU/GPU latency measurements
   - FPS benchmarking
   - Memory usage profiling

2. **Deployment Options**:
   - Lightweight inference script
   - Docker containerization (CPU/GPU)
   - REST API implementation
   - Interactive Streamlit demo

### Installation & Usage
```bash
# Clone the repository
git clone https://github.com/Noman-Nom/plant_disease_infection-gpu.git
cd plant_disease_infection-gpu

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.\.venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run inference
python src/inference.py --image path/to/image.jpg
```

### Future Roadmap
1. Mobile deployment optimization
2. Real-time processing capabilities
3. Integration with agricultural IoT systems
4. Extended disease class support
5. Multi-crop disease detection
6. Seasonal model adaptation

## 📫 Contact & Support

For questions and support, please:
- Open an issue in the GitHub repository
- Contact: [Your Contact Information]
- Visit: [Project Website/Documentation]

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Plant Village Dataset
- PyTorch Team
- timm library contributors
- Agricultural domain experts
