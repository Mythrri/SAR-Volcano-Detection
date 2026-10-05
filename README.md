# 🌋 SAR Volcano Deformation Detection

An AI-powered system to detect volcanic deformation using Sentinel-1 SAR (Synthetic Aperture Radar) interferograms with Deep Learning.

### 🎯 Problem Statement
Traditional volcano monitoring is risky and slow. This project uses satellite SAR images to automatically detect ground deformation that indicates volcanic activity.

### 🧠 Tech Stack
- Python, PyTorch, ResNet18
- Computer Vision, Transfer Learning
- Sentinel-1 SAR Data

### 📁 Project Structure
- `model.py` - ResNet18 model architecture
- `train.py` - Model training pipeline
- `predict.py` - Real-time prediction
- `dataset/` - deformation / non_deformation interferograms

### 🚀 How to Run
```bash
pip install -r requirements.txt
python train.py
python predict.py

**📊Results**
Model Accuracy: ∼90%+
Correctly classifies:✅ Non-Deformation (Normal)
🌋 Volcano Deformation Detected

**🔬Future Scope**
-Real-time satellite data integration
-Multi-volcano early warning system
