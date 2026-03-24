# 🩺 Multi-Class Chest X-Ray Pathology Classification 

## 📌 Overview

This project focuses on building a robust deep learning system for **multi-class classification of chest X-ray images** across **20 thoracic pathologies**. The work was carried out as part of a competitive Kaggle-based assignment, where the objective was to optimize performance under a **cost-sensitive medical evaluation metric**.

🚀 **Achievement:** Secured **Rank 1** on the leaderboard by designing an effective ensemble of CNNs and Vision Transformers with strong generalization and recall.

---

## 🎯 Problem Statement

* Classify each chest X-ray image into one of **20 pathology classes** (including *No Finding*)
* Handle:

  * Severe **class imbalance**
  * **Asymmetric misclassification cost** (False Negatives heavily penalized)
* Optimize for a **custom cost-based evaluation metric** rather than standard accuracy

---

## ⚙️ Methodology

### 🔹 Model Architectures

Implemented and trained multiple state-of-the-art models:

* **EfficientNet-B0** (CNN-based feature extractor)
* **ConvNeXt-Tiny** (modern convolutional architecture)
* **DeiT3-Small** (Vision Transformer)

---

### 🔹 Training Strategy

* Stratified train-validation split to preserve class distribution
* Data augmentation:

  * Random horizontal flips
  * Rotation
  * Color jitter
* Optimization:

  * AdamW optimizer
  * Cosine Annealing Learning Rate Scheduler
* Evaluation:

  * **Macro F1-score** (to handle imbalance)

---

### 🔹 Handling Class Imbalance

* Used **macro-averaged metrics** to ensure equal importance across all classes
* Designed training to improve **recall for rare diseases**

---

### 🔹 Cost-Sensitive Learning

The evaluation penalized:

* False Negatives (−5) ❗
* False Positives (−1)

👉 Model was optimized to:

* Maximize **True Positives**
* Minimize **False Negatives** (critical in medical diagnosis)

---

## 🔗 Model Ensemble (Key Innovation)

Final predictions were generated using a **weighted ensemble**:

* ConvNeXt → 50% weight
* EfficientNet → 25% weight
* DeiT → 25% weight

This improved:

* Robustness
* Generalization
* Performance on rare classes

---

## 🔄 Pipeline

```
Input X-ray Image
        ↓
Preprocessing & Augmentation
        ↓
Model Predictions (EffNet / ConvNeXt / DeiT)
        ↓
Weighted Ensemble
        ↓
Softmax Probability Outputs
        ↓
Final Class Prediction (One-hot)
```

---

## 📊 Results

* 🏆 **Rank 1 on Kaggle leaderboard**
* Strong performance on:

  * Rare pathologies
  * Cost-sensitive metric
* High **macro F1-score**
* Significant reduction in **false negatives**

---

## 📁 Project Structure

```
├── train.py          # Model training pipeline
├── predict.py        # Inference & submission generation
├── model.py          # Model architectures
├── requirements.txt  # Dependencies
├── checkpoint/       # Saved model weights
```

---

## 🚀 How to Run

### Train Models

```
python train.py --data_path path/to/train.csv --model_out_path checkpoint/
```

### Generate Predictions

```
python predict.py --data_path path/to/test/images \
--model_path checkpoint/ \
--output submission.csv
```

---

## 🔗 Kaggle Notebook

👉 https://www.kaggle.com/code/hsiiestiitd27/notebookbf02f18d91

---

## 🛠 Tech Stack

* PyTorch
* torchvision
* timm (Vision Transformers)
* NumPy, Pandas, scikit-learn

---

## 💡 Key Learnings

* Handling **imbalanced medical datasets**
* Designing **cost-sensitive models**
* Combining CNNs + Transformers effectively
* Importance of **ensemble learning** in competitions
* Building **reproducible ML pipelines**

---

## 📌 Future Improvements

* Incorporate **attention-based localization (Grad-CAM)**
* Use **self-supervised pretraining**
* Optimize **threshold tuning per class**
* Explore **multi-label classification setting**

---

## 👤 Author

Hrithik Sharma
IIT Delhi

---



