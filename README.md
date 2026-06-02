# Credit-card-fraud-detection
Machine learning-based credit card fraud detection system using Random Forest, SMOTE, and Streamlit for real-time transaction prediction.

## Project Overview
This project uses machine learning techniques to detect fraudulent credit card transactions. The model is trained on a highly imbalanced dataset and deployed through a Streamlit web application for real-time predictions.

## Features
- Exploratory Data Analysis (EDA)
- Fraud detection using Random Forest
- SMOTE oversampling for class imbalance
- Batch CSV prediction
- Fraud probability scoring
- Streamlit web application

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Streamlit

## Dataset
Credit Card Fraud Detection Dataset from Kaggle.

## How to Run

1. Install dependencies

```bash
pip install -r requirements.txt
```

2. Run the application

```bash
streamlit run streamlit_app.py
```

## Project Structure

```text
credit-card-fraud-detection/
│
├── fraud_detection_final.ipynb
├── streamlit_app.py
├── README.md
└── models/
    └── final_pipeline.joblib
```
