# 🧠 Tumor Detection using ResNet18

This repository contains a deep learning-based approach for classifying brain tumor types using grayscale MRI images. The model leverages a fine-tuned ResNet-18 convolutional neural network.

## 📁 Repository Structure

```
├── Data/
│   ├── Training/    # Training images organized by class
│   └── Testing/     # Testing images organized by class
├── dependencies_installation.ipynb  # Notebook to install dependencies
├── requirements.txt                 # Python dependencies
├── resnet_model.py                  # Core classification model
```

## 📦 Dataset

The dataset used for training and testing is publicly available on Kaggle:

👉 [Brain Tumor MRI Dataset on Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

Download the dataset and place it in the `Data/` directory, structured as follows:

```
Data/
├── Training/
│   ├── glioma_tumor/
│   ├── meningioma_tumor/
│   ├── notumor/
│   └── pituitary_tumor/
├── Testing/
    ├── glioma_tumor/
    ├── meningioma_tumor/
    ├── notumor/
    └── pituitary_tumor/
```

## 🔧 Setup Instructions

1. **Clone the repository**

```bash
git clone https://github.com/Abdullah-Shahid01/tumor-detection.git
cd tumor-detection
```

2. **Install dependencies**

You can either:

- Run the setup notebook:

```bash
jupyter notebook dependencies_installation.ipynb
```

OR

- Use the requirements file:

```bash
pip install -r requirements.txt
```

> 📌 **Python version**: 3.10

## 🧠 Model Overview

- **Architecture**: Pretrained ResNet-18 from PyTorch
- **Input**: Grayscale MRI images, resized to 224×224
- **Output**: 4-class classification
- **Transfer Learning**:
  - Earlier layers are frozen
  - Last block (`layer4`) and final `fc` layer are unfrozen for fine-tuning

## 🏋️ Training

- Epochs: 10  
- Loss Function: CrossEntropyLoss  
- Optimizer: Adam (lr=0.0001)  
- Batch size: 32  

```python
for epoch in range(num_epochs):
    model.train()
    ...
```

## 📊 Evaluation

After training, the model is evaluated on the test set and outputs:
- Classification report (precision, recall, F1-score)
- Confusion matrix heatmap

## 📌 Notes

- Ensure the dataset folder structure matches what the model expects.
- The model uses grayscale images; channel conversion is handled internally.


---

## ✨ Acknowledgements

- Dataset by [Masoud Nickparvar on Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
