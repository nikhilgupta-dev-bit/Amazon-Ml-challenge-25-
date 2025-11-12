# 🧠 Price Category Prediction using Ensemble Learning (XGBoost)

An end-to-end **machine learning project** that classifies products into **Low**, **Medium**, and **High** price categories using their textual descriptions.  
This project leverages **TF-IDF**, **SVD (Dimensionality Reduction)**, and advanced **Ensemble Learning** techniques such as **XGBoost** to achieve competitive classification performance.

---

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [Project Framework](#project-framework)
- [Model Workflow](#model-workflow)
- [Implementation Steps](#implementation-steps)
- [Results](#results)
- [Visualizations](#visualizations)
- [Future Scope](#future-scope)
- [How to Run](#how-to-run)
- [Contributors](#contributors)

---

## 🚀 Overview
E-commerce platforms list thousands of products, and determining an item’s **price category** based on its description helps in automated pricing, marketing, and recommendation systems.

This project aims to:
> Predict whether a product belongs to the **Low**, **Medium**, or **High** price range using **machine learning** models trained on product catalog data.

### ✅ Key Features
- Text-based and numeric feature extraction from `catalog_content`
- Preprocessing pipeline with **TF-IDF** and **TruncatedSVD**
- Ensemble learning models comparison (Naive Bayes, Random Forest, AdaBoost, XGBoost)
- Final optimized **XGBoost** pipeline with 59% validation accuracy
- Exported trained model and predictions for test dataset

---

## 🧾 Dataset
The dataset consists of two CSV files:

| File | Description |
|------|--------------|
| `train.csv` | Training data with product descriptions and prices |
| `test.csv`  | Test data without price categories (to predict) |

### Columns:
| Column | Description |
|---------|-------------|
| `sample_id` | Unique identifier for each product |
| `catalog_content` | Product textual description |
| `image_link` | Image URL (not used in model) |
| `price` | Numerical product price |
| `price_class` | Derived target (Low / Medium / High) |

---

## 🧰 Tech Stack

| Category | Tools/Packages |
|-----------|----------------|
| **Language** | Python 3.10+ |
| **Data Handling** | Pandas, NumPy |
| **ML Frameworks** | Scikit-learn, XGBoost |
| **Visualization** | Matplotlib, Seaborn |
| **Model Persistence** | Joblib |
| **IDE** | VS Code / Jupyter Notebook |

---

## 🧱 Project Framework


---

## 🔍 Model Workflow

1. **Data Loading** – Import training and testing datasets  
2. **Data Cleaning** – Handle missing values and normalize text  
3. **Feature Engineering**  
   - Extract numeric text features (length, word count, etc.)  
   - Generate TF-IDF vectors + SVD components  
4. **Model Selection** – Compare baseline models  
5. **Training** – Train tuned **XGBoost Classifier**  
6. **Evaluation** – Accuracy, precision, recall, F1-score  
7. **Model Saving** – Export trained model using `joblib`  
8. **Prediction** – Generate final predictions for test dataset  

---
###🧑‍💻 Contributors
Nikhil Gupta
📧 [nikhilunofficial123@gmail.com]
💼 [www.linkedin.com/in/nikhil-gupta-76a6a12a6]
💻 [https://github.com/nikhilgupta-dev-bit]

## ⚙️ Implementation Steps

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/price-category-prediction.git
cd price-category-prediction
pip install -r requirements.txt
jupyter notebook best_model_pipeline.ipynb
test_xgb_price_class_predictions.csv
<img width="863" height="569" alt="Screenshot 2025-11-13 at 12 00 15 AM" src="https://github.com/user-attachments/assets/551c71fb-1d34-4ef4-b3e9-c918dea6b1c9" />

---

Would you like me to make a **matching `requirements.txt`** (with all exact dependencies and versions) to go with this README? It’ll make your repo instantly runnable by anyone.

