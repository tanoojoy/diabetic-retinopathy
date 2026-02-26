# Diabetic Retinopathy Classification Pipeline

A hybrid machine learning pipeline for Diabetic Retinopathy (DR) classification using:

- ResNet18 (ImageNet pretrained) for deep feature extraction
- PCA for dimensionality reduction
- Random Forest classifier
- Artificial Neural Network (ANN) classifier

This project combines transfer learning + classical ML + deep learning into a structured and reproducible workflow.

## Installation
1. Create Virtual Environment
```bash 
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

2. Install Dependencies
```bash
pip install -r requirements.txt
```

## How To Run
1. Download the dataset first
 ```bash
 python dataset.py
 ```

 This will create a new directory called `the_dataset`

2. Run the training
```bash
python training.py
```
This will:
- Load dataset
- Balance class distribution
- Split into train & validation
- Extract deep features using ResNet18
- Apply PCA (100 components)
- Train Random Forest
- Train ANN
- Save trained models
- Print evaluation metrics
- Display confusion matrix

## Pipeline Overview
### Step 1 — Dataset Balancing

The dataset is balanced using bootstrapped resampling:

`resample(subset, replace=True, n_samples=max_size)`

This prevents class imbalance bias.

### Step 2 — Deep Feature Extraction
- Uses pretrained ResNet18 (ImageNet weights)
- Final fully connected layer removed
- Outputs 512-dimensional feature vectors
- Operates in evaluation mode (no fine-tuning)

Image preprocessing:
```python
transforms.Resize((224, 224))
transforms.ToTensor()
transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
```

### Step 3 — PCA Dimensionality Reduction

Reduces 512 features → 100 components:

`PCA(n_components=100)`

Benefits:

- Removes noise

- Reduces overfitting

- Improves training speed

Saved as: `pca.pkl`

### Step 4A — Random Forest Classifier
```python
RandomForestClassifier(
    n_estimators=300,
    max_depth=25,
    class_weight='balanced',
    random_state=42
)
```

Outputs:

- Accuracy
- Weighted F1 Score
- Weighted Precision
- Top 10 PCA feature importance plot

Saved as: `model.pkl`

### Step 4B — Artificial Neural Network (ANN)

Architecture:
```
Input (100)
  → Linear(128)
  → ReLU
  → Dropout(0.3)
  → Linear(64)
  → ReLU
  → Linear(5)
```

Training configuration:
- Loss: CrossEntropyLoss
- Optimizer: Adam
- Learning rate: 0.001
- Epochs: 10
- Batch size: 64

Saved as: `ann_model.pt`

## Evaluation Metrics

- Accuracy
- Weighted F1 Score
- Weighted Precision
- Full Classification Report
- Confusion Matrix (heatmap)

Saved Models
|File|Description|
|:--:| :--:
|pca.pkl	|Trained PCA transformer|
|model.pkl	|Random Forest model|
|ann_model.pt|	Trained ANN weights|

## Inference Example
### Load PCA + Random Forest
```python
import joblib
import torch

pca = joblib.load("pca.pkl")
rf_model = joblib.load("model.pkl")

features = extract_features("new_image.jpg")
features_pca = pca.transform([features])

prediction = rf_model.predict(features_pca)
print(prediction)
```

### Load ANN
```python
ann_model = SimpleANN(100, 5)
ann_model.load_state_dict(torch.load("ann_model.pt"))
ann_model.eval()

with torch.no_grad():
    tensor = torch.tensor(features_pca, dtype=torch.float32)
    output = ann_model(tensor)
    pred = torch.argmax(output, dim=1)
    print(pred)
```