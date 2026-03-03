# Sleep Breathing Irregularity Detection
---

# 📌 Project Overview

This project develops an end‑to‑end machine learning pipeline to detect abnormal breathing events during sleep using physiological signals.

The system processes overnight polysomnography (PSG) data and trains a deep learning model to classify normal vs abnormal breathing windows.

---

# 🧠 Problem Statement

Given overnight recordings for each participant containing:

* Nasal Airflow (32 Hz)
* Thoracic Movement (32 Hz)
* SpO₂ (4 Hz)
* Annotated breathing events
* Sleep profile

The goal is to:

1. Visualize full‑night signals
2. Preprocess signals
3. Create labeled windows
4. Train a 1D CNN
5. Evaluate using Leave‑One‑Participant‑Out (LOPO) CV

---

# 📂 Project Structure

```
Project Root/
│
├── Data/                     # Raw participant data (AP01–AP05)
├── Dataset/
│   └── windows_file.pkl           # Processed window dataset
├── Models/
│   ├── cnn_test_APxx.h5      # LOPO fold models
│   └── final_cnn.h5          # Final deployment model
├── Visualizations/
│   └── APxx_report.pdf     # Multi‑page PSG plots
└── scripts/
    ├── vis.py                # Visualization pipeline
    ├── create_dataset.py     # Preprocessing & windowing
    └── train_model.py        # CNN training & evaluation
```

---

# 🔍 Part 1 — Data Visualization

## Objective

To visually inspect overnight signals and verify annotated breathing events.

## Key Features Implemented

* Multi‑page full‑night visualization
* Clinical PSG‑style layout
* Timestamp with day and seconds
* Event overlay with color coding
* Automatic file detection

## Signals Plotted

* Nasal Airflow
* Thoracic Movement
* SpO₂

## Output

Saved to:

```
Visualizations/APxx_report.pdf
```

---

# ⚙️ Part 2 — Signal Preprocessing & Dataset Creation

## 🎯 Goal

Convert raw physiological signals into machine‑learning ready labeled windows.

---

## Step 1: Bandpass Filtering

Human breathing frequency range:

```
0.17 – 0.4 Hz (10–24 breaths/min)
```

A Butterworth bandpass filter was applied to:

* Nasal airflow
* Thoracic movement

**Purpose:** remove high‑frequency noise and retain breathing patterns.

---

## Step 2: Sliding Window Segmentation

Parameters used:

* Window length: **30 seconds**
* Overlap: **50%**
* Stride: **15 seconds**

This converts continuous signals into fixed‑length segments suitable for CNN input.

---

## Step 3: Window Labeling

Labeling rule:

* If event overlap > 50% of window → assign event label
* Otherwise → label as **Normal**

This ensures medically meaningful supervision.

---

## Step 4: Dataset Construction

Each window stored as:

```
{
  participant,
  start_time,
  label,
  airflow (960 samples),
  thoracic (960 samples),
  spo2 (120 samples)
}
```

Dataset saved as:

```
Dataset/windows_file.pkl
```

---

# 📊 Dataset Statistics

Observed class distribution:

* Normal: 8038
* Hypopnea: 593
* Obstructive Apnea: 164
* Body event: 3
* Mixed Apnea: 2

## Important Observation

The dataset is **highly imbalanced**.
---
# linear interpolation

* airflow   → (960,)
* thoracic  → (960,)
* spo2      → (120,)
* linear interpolation is used foer SpO₂ to  match length
---

# 🔄 Binary Label Strategy

To handle extreme imbalance and match the project goal (detect abnormal breathing), labels were converted to binary:

* **Normal → 0**
* **Any abnormal event → 1**

This improves model stability and medical relevance.

---

# 🤖 Part 3 — Deep Learning Model

## Model Type

**1D Convolutional Neural Network (CNN)**

## Input Design

Three signals are used as channels:

* Airflow
* Thoracic
* SpO₂ (resampled to match length)

Final input shape:

```
(length=960, channels=3)
```

---

## Network Architecture

* Conv1D → ReLU → MaxPool
* Conv1D → ReLU → MaxPool
* Conv1D → Global Average Pool
* Dense →
* Sigmoid output

---

# 🔁 Evaluation Strategy

## Leave‑One‑Participant‑Out (LOPO)

Procedure:

* Train on 4 participants
* Test on remaining participant
* Repeat for all participants

---

# 📈 Metrics Reported

For each fold:

* Accuracy
* Precision
* Recall
* Confusion Matrix

Final performance reported as mean across LOPO folds.

---

# 💾 Model Saving Strategy

Two types of models are saved:

## 1. LOPO Fold Models (for evaluation)

```
Models/cnn_test_APxx.keras
```

Purpose:

* reproducibility
* academic evaluation

## 2. Final Model ⭐

After LOPO, model is trained on full dataset and saved as:

```
Models/final_cnn.keras
```

This is the model intended for real‑world use.

---

# 🚀 How to Run

## Step 1 — Visualization

```
python scripts/vis.py -name "Data/AP01"
```

## Step 2 — Create Dataset

```
python scripts/create_dataset.py -in_dir "Data" -out_dir "Dataset"
```

## Step 3 — Train Model

```
python scripts/train_model.py
```

---

# 🧪 Environment

* Python 3.10
* TensorFlow 2.10 (GPU enabled)
* NumPy
* SciPy
* Pandas
* Matplotlib
* scikit‑learn

---

# 🔮 Possible Improvements

Future enhancements may include:

* Class‑weighted loss
* Focal loss for imbalance
* Deeper CNN / ResNet1D
* Attention mechanisms
* Real‑time inference pipeline

---

# 🏆 Conclusion

This project successfully implements a complete end‑to‑end pipeline for sleep breathing abnormality detection, including:

* clinical visualization
* biomedical signal filtering
* robust window labeling
* subject‑independent evaluation
* deep learning classification.

---

**End of README**
